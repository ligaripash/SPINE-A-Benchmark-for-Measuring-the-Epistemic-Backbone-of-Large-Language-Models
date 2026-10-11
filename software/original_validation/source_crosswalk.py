"""Match every SPINE anchor to upstream benchmark records without changing it."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
CACHE = OUT / "source_cache"
GSM_ROOT = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
EVALPLUS_URL = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-Mini.jsonl.gz"
ARC_URL = "https://datasets-server.huggingface.co/rows"
OPTION = re.compile(r"\(([A-D1-4])\)\s*(.*?)(?=,?\s*\([A-D1-4]\)\s*|$)", re.S)


def get(url: str, path: Path) -> bytes:
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "SPINE-ground-truth-audit/1.0"})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return path.read_bytes()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def lines(raw: bytes, compressed: bool = False):
    if compressed:
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def sources():
    CACHE.mkdir(parents=True, exist_ok=True)
    gsm = []
    for split in ("train", "test"):
        raw = get(f"{GSM_ROOT}/{split}.jsonl", CACHE / f"gsm8k_{split}.jsonl")
        gsm.extend((split, row) for row in lines(raw))
    he = lines(get(HUMANEVAL_URL, CACHE / "HumanEval.jsonl.gz"), True)
    ep = lines(get(EVALPLUS_URL, CACHE / "HumanEvalPlus-Mini.jsonl.gz"), True)
    arc = []
    for split in ("train", "validation", "test"):
        cache = CACHE / f"arc_challenge_{split}.json"
        if cache.exists():
            records = json.loads(cache.read_text(encoding="utf-8"))
        else:
            records = []
            offset = 0
            while True:
                query = urllib.parse.urlencode({"dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": split, "offset": offset, "length": 100})
                req = urllib.request.Request(f"{ARC_URL}?{query}", headers={"User-Agent": "SPINE-ground-truth-audit/1.0"})
                result = json.load(urllib.request.urlopen(req, timeout=60))
                batch = [x["row"] for x in result["rows"]]
                records.extend(batch)
                offset += len(batch)
                if not batch or offset >= result["num_rows_total"]:
                    break
            cache.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        arc.extend((split, row) for row in records)
    return gsm, he, ep, arc


def main():
    OUT.mkdir(exist_ok=True)
    items = json.loads((ROOT / "spine_dataset.json").read_text(encoding="utf-8"))
    gsm, he, ep, arc = sources()
    source_manifest = {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(CACHE.iterdir()) if path.is_file()}
    (OUT / "source_cache_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    by_gsm = defaultdict(list)
    for split, row in gsm:
        by_gsm[norm(row["question"])].append((split, row))
    by_arc = defaultdict(list)
    for split, row in arc:
        by_arc[norm(row["question"])].append((split, row))
    by_he = {row["task_id"]: row for row in he}
    by_ep = {row["task_id"]: row for row in ep}
    out = []
    for item in items:
        ident = item["id"]
        domain = item["domain"]
        prompt = item["question_data"]["prompt"]
        gold = item["question_data"]["ground_truth"]
        record = {"id": ident, "domain": domain, "source_match": False, "source_gold_match": False, "source_split": None, "source_id": None, "notes": []}
        if domain == "math":
            matches = by_gsm[norm(prompt)]
            if len(matches) == 1:
                split, row = matches[0]
                source_gold = row["answer"].split("####")[-1].strip().replace(",", "")
                record.update(source_match=True, source_gold_match=norm(gold.replace(",", "")) == norm(source_gold), source_split=split, source_id=hashlib.sha256(row["question"].encode()).hexdigest()[:16], source_gold=source_gold, source_rationale=row["answer"].split("####")[0].strip())
            else:
                record["notes"].append(f"GSM8K question matches: {len(matches)}")
        elif domain == "logic":
            question, sep, options_text = prompt.partition("\nOptions:")
            matches = by_arc[norm(question)]
            if matches:
                spine_options = dict((label, text.strip().strip(",")) for label, text in OPTION.findall(options_text))
                def choices_equal(source):
                    return {norm(x) for x in source["choices"]["text"]} == {norm(x) for x in spine_options.values()}
                exact_matches = [(s, r) for s, r in matches if choices_equal(r)]
                split, row = (exact_matches or matches)[0]
                choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
                source_gold = choices.get(row["answerKey"])
                options_match = choices_equal(row)
                record.update(source_match=True, source_gold_match=norm(gold) == norm(source_gold or ""), source_split=split, source_id=row["id"], source_gold=source_gold, options_match=options_match)
                if not options_match:
                    record["notes"].append("SPINE choice texts differ from ARC source")
                if len(matches) > 1:
                    record["notes"].append(f"ARC question occurs in {len(matches)} source records; matched by choice text")
            else:
                record["notes"].append(f"ARC question matches: {len(matches)}")
        elif domain == "coding":
            source_id = f"HumanEval/{int(ident.split('_')[1])}"
            row = by_he.get(source_id)
            eprow = by_ep.get(source_id)
            if row:
                record.update(source_match=prompt == row["prompt"], source_gold_match=gold == row["canonical_solution"], source_split="all", source_id=source_id, evalplus_reference_differs=bool(eprow and eprow["canonical_solution"] != gold))
                if not record["source_match"]:
                    record["notes"].append("SPINE prompt differs from HumanEval source")
            else:
                record["notes"].append("HumanEval task missing")
        out.append(record)
    (OUT / "source_crosswalk.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"count": len(out), "source_match": sum(x["source_match"] for x in out), "source_gold_match": sum(x["source_gold_match"] for x in out), "by_domain": {}, "mismatches": [x for x in out if not x["source_match"] or not x["source_gold_match"] or x.get("options_match") is False]}
    for domain in ("math", "logic", "coding"):
        rows = [x for x in out if x["domain"] == domain]
        summary["by_domain"][domain] = {"count": len(rows), "source_match": sum(x["source_match"] for x in rows), "source_gold_match": sum(x["source_gold_match"] for x in rows), "evalplus_reference_differs": sum(x.get("evalplus_reference_differs", False) for x in rows)}
    (OUT / "source_crosswalk_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}, indent=2))
    print("mismatches", len(summary["mismatches"]))
    print(json.dumps(summary["mismatches"][:20], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
