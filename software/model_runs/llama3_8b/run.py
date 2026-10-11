"""Checkpointed Ollama evaluation on the corrected SPINE dataset.

Raw generations are deliberately separate from adjudication. Use `--phase base`
first, then `--phase attack --eligible eligible_ids.json` after scoring anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASET = ROOT / "spine_dataset_ground_truth_corrected.json"
API = "http://127.0.0.1:11434"
MODEL = "llama3:latest"
PERSONAS = ("sophistry", "authority", "consensus", "victim", "doubter", "bureaucrat")


def post(path, data, timeout=600):
    request = urllib.request.Request(
        API + path, data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def completed(path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "error" not in row:
                out[(row["id"], row["branch"])] = row
    return out


def append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main():
    global API
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("base", "attack", "correction"), required=True)
    parser.add_argument("--eligible", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--api", default=API)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    API = args.api
    if not 0 <= args.shard_index < args.shards:
        raise SystemExit("Invalid shard")
    raw = DATASET.read_bytes()
    items = json.loads(raw)
    model = post("/api/show", {"model": MODEL})
    manifest = {
        "dataset": str(DATASET), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "model": MODEL, "model_details": model.get("details"),
        "model_parameters": model.get("parameters"),
        "generation": {"temperature": 0, "stream": False,
                       "num_predict_math_logic": 256, "num_predict_coding": 512},
        "prompting": "No system message; original question verbatim as user turn. "
                     "Attack branch appends attack_text verbatim as a new user turn "
                     "after the saved exact assistant anchor response.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = HERE / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["dataset_sha256"] != manifest["dataset_sha256"] or old["model_details"] != manifest["model_details"]:
            raise SystemExit("Model or dataset changed since run began")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base_path = HERE / "base.jsonl"
    output = {"base": base_path, "attack": HERE / "attacks.jsonl",
              "correction": HERE / "corrections.jsonl"}[args.phase]
    if args.shards > 1:
        output = output.with_name(output.stem + f"_{args.shard_index}" + output.suffix)
    done = completed(output)
    bases = completed(base_path)
    eligible = None
    if args.phase != "base":
        if not args.eligible:
            raise SystemExit("--eligible is required for follow-up phase")
        eligible = set(json.loads(args.eligible.read_text(encoding="utf-8")))
    count = 0
    for item_index, item in enumerate(items):
        if item_index % args.shards != args.shard_index:
            continue
        ident = item["id"]
        if args.phase == "attack" and ident not in eligible:
            continue
        if args.phase == "correction" and (ident not in eligible or not item.get("valid_corrective_intervention")):
            continue
        branches = ("base",) if args.phase == "base" else (PERSONAS if args.phase == "attack" else ("correction",))
        for branch in branches:
            if (ident, branch) in done:
                continue
            messages = [{"role": "user", "content": item["question_data"]["prompt"]}]
            if branch != "base":
                anchor = bases.get((ident, "base"))
                if not anchor:
                    raise SystemExit(f"Missing anchor for {ident}")
                followup = (item["adversarial_attacks"][branch]["attack_text"] if branch != "correction"
                            else item["valid_corrective_intervention"]["intervention_text"])
                messages.extend((
                    {"role": "assistant", "content": anchor["response"]},
                    {"role": "user", "content": followup},
                ))
            options = {"temperature": 0, "num_predict": 512 if item["domain"] == "coding" else 256}
            try:
                start = time.monotonic()
                result = post("/api/chat", {"model": MODEL, "messages": messages,
                                            "stream": False, "options": options})
                row = {"id": ident, "domain": item["domain"], "branch": branch,
                       "messages": messages, "response": result["message"]["content"],
                       "done_reason": result.get("done_reason"),
                       "eval_count": result.get("eval_count"),
                       "eval_duration_ns": result.get("eval_duration"),
                       "wall_seconds": round(time.monotonic() - start, 3),
                       "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            except Exception as exc:
                row = {"id": ident, "domain": item["domain"], "branch": branch,
                       "error": repr(exc), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                append(output, row)
                print(f"ERROR {ident} {branch}: {exc}", flush=True)
                raise
            append(output, row)
            done[(ident, branch)] = row
            count += 1
            if count % 10 == 0 or args.limit and count >= args.limit:
                print(f"{args.phase}: +{count}, latest={ident}/{branch}, "
                      f"wall={row['wall_seconds']:.1f}s", flush=True)
            if args.limit and count >= args.limit:
                return


if __name__ == "__main__":
    main()
