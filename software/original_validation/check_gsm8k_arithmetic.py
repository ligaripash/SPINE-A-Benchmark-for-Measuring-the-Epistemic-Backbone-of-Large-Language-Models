"""Check all explicit <<calculation=result>> steps in the 300 GSM8K rationales."""
from __future__ import annotations

import ast
import json
import re
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = re.compile(r"<<([^<>]+)>>")


def arithmetic(node):
    if isinstance(node, ast.Expression):
        return arithmetic(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -arithmetic(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = arithmetic(node.left), arithmetic(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right.numerator) < 20:
            return left ** right.numerator
    raise ValueError("unsupported expression")


def check_step(step: str):
    if "=" not in step:
        return "unparseable", None
    lhs, rhs = step.rsplit("=", 1)
    lhs = lhs.strip().replace(",", "").replace("×", "*").replace("÷", "/").replace("^", "**")
    rhs = rhs.strip().replace(",", "")
    try:
        expected = Fraction(rhs)
        actual = arithmetic(ast.parse(lhs, mode="eval"))
        if actual == expected or abs(actual - expected) < Fraction(1, 1000000):
            return "correct", None
        return "incorrect", f"{lhs} = {actual}, rationale says {rhs}"
    except Exception as exc:
        return "unparseable", f"{type(exc).__name__}: {str(exc)[:100]}"


def main():
    rows = json.loads((HERE / "source_crosswalk.json").read_text(encoding="utf-8"))
    out = []
    for item in rows:
        if item["domain"] != "math":
            continue
        steps = []
        for step in STEP.findall(item["source_rationale"]):
            status, note = check_step(step)
            steps.append({"expression": step, "status": status, "note": note})
        out.append({"id": item["id"], "gold": item["source_gold"], "step_count": len(steps), "checked_steps": sum(x["status"] != "unparseable" for x in steps), "incorrect_steps": sum(x["status"] == "incorrect" for x in steps), "steps": steps})
    (HERE / "gsm8k_arithmetic.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"tasks": len(out), "tasks_with_steps": sum(x["step_count"] > 0 for x in out), "steps": sum(x["step_count"] for x in out), "checked_steps": sum(x["checked_steps"] for x in out), "incorrect_steps": sum(x["incorrect_steps"] for x in out), "incorrect_ids": [x["id"] for x in out if x["incorrect_steps"]]}
    (HERE / "gsm8k_arithmetic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
