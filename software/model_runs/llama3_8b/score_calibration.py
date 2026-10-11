"""Compare Qwen answer extraction with deterministic explicit-choice parsing."""
import json
from pathlib import Path

from score_objective import DATASET, HERE, OPTION, norm

items = {x["id"]: x for x in json.loads(DATASET.read_text(encoding="utf-8"))}
rows = [json.loads(line) for line in (HERE / "logic_judge_calibration.jsonl").read_text(encoding="utf-8").splitlines()]
agreement = []
unresolved = []
for row in rows:
    item = items[row["id"]]
    options = dict(OPTION.findall(item["question_data"]["prompt"].split("Options:", 1)[-1]))
    gold = [label for label, text in options.items() if norm(text) == norm(item["question_data"]["ground_truth"])]
    if len(gold) != 1 or row["choice"] not in options:
        unresolved.append({"id": row["id"], "branch": row["branch"], "choice": row["choice"]})
        continue
    extracted_correct = row["choice"] == gold[0]
    agreement.append({"id": row["id"], "branch": row["branch"], "choice": row["choice"],
                      "extracted_correct": extracted_correct, "parser_correct": row["parser_correct"]})
summary = {"sampled": len(rows), "resolved": len(agreement), "unresolved": len(unresolved),
           "agreed": sum(x["extracted_correct"] == x["parser_correct"] for x in agreement),
           "disagreements": [x for x in agreement if x["extracted_correct"] != x["parser_correct"]],
           "unresolved_rows": unresolved}
(HERE / "logic_judge_calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
