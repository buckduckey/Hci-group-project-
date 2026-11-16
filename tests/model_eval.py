"""Runs form inputs through the real /api/infer route and checks the answers.

Used by test_model_live.py, and runnable on its own to compare models:

    python -m tests.model_eval --models llama3.2:1b llama3.2:3b --runs 3

Needs Ollama running with the models installed.
"""
import argparse
import json
import statistics
import urllib.request

import LLM
from app import app
from tests.grounding import check_answer

# What the page sends when the form is submitted untouched.
DEFAULT_FORM = {
    "q1": "6", "q2": True, "q3": "Shopping", "q4": "Deliberate",
    "q5": True, "q6": True, "q7": True, "q8": True,
    "q9": "None", "q10": True, "q11": True,
}

# Chosen to differ from the prompt's worked example in different ways, so an
# answer that copies the example instead of reading the input gets caught.
CASES = {
    "default form": DEFAULT_FORM,
    "default, geolocation off": dict(DEFAULT_FORM, q11=False),
    "everything off": {
        "q1": "2", "q2": False, "q3": "News", "q4": "Steady",
        "q5": False, "q6": False, "q7": False, "q8": False,
        "q9": "VPN", "q10": False, "q11": False,
    },
    "opposite of the example": {
        "q1": "1", "q2": False, "q3": "Shopping", "q4": "Deliberate",
        "q5": False, "q6": True, "q7": False, "q8": True,
        "q9": "VPN", "q10": False, "q11": True,
    },
    "heavy use, well protected": {
        "q1": "14", "q2": True, "q3": "News", "q4": "Rapid",
        "q5": False, "q6": False, "q7": False, "q8": False,
        "q9": "VPN", "q10": False, "q11": False,
    },
    "light use, unprotected": {
        "q1": "3", "q2": False, "q3": "Social", "q4": "Steady",
        "q5": True, "q6": False, "q7": True, "q8": False,
        "q9": "AdBlock", "q10": True, "q11": True,
    },
    "same hours as the example": {
        "q1": "9", "q2": False, "q3": "News", "q4": "Deliberate",
        "q5": False, "q6": True, "q7": False, "q8": True,
        "q9": "AdBlock", "q10": False, "q11": True,
    },
    "only geolocation on": {
        "q1": "5", "q2": False, "q3": "Shopping", "q4": "Steady",
        "q5": False, "q6": False, "q7": False, "q8": False,
        "q9": "None", "q10": False, "q11": True,
    },
}


def ollama_available():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3).close()
        return True
    except OSError:
        return False


def ask(model, form):
    """Send one form through the route. Returns (answer, error message)."""
    previous = LLM.MODEL
    LLM.MODEL = model
    try:
        response = app.test_client().post("/api/infer", json=form)
    finally:
        LLM.MODEL = previous
    body = response.get_json()
    if response.status_code != 200:
        return None, body.get("error", f"HTTP {response.status_code}")
    return body, None


def evaluate(model, runs):
    """Ask every case `runs` times. Returns one record per case."""
    records = []
    for name, form in CASES.items():
        answers, errors = [], []
        for _ in range(runs):
            answer, error = ask(model, form)
            if error:
                errors.append(error)
            else:
                answers.append(answer)
        distinct = {json.dumps(answer, sort_keys=True) for answer in answers}
        records.append({
            "case": name,
            "form": form,
            "errors": errors,
            "answers": answers,
            "deterministic": not errors and len(distinct) == 1,
            "problems": [check_answer(answer, form) for answer in answers],
        })
    return records


def summarise(model, records):
    evidence_lines = bad_lines = 0
    kinds = {}
    confidences = []
    for record in records:
        for answer, problems in zip(record["answers"], record["problems"]):
            evidence_lines += len(answer["key_info"])
            bad_lines += len({index for index, *_ in problems})
            for _, kind, _, _ in problems:
                kinds[kind] = kinds.get(kind, 0) + 1
            confidences += [item["confidence"] for item in answer["key_info"]]

    print(f"\n=== {model} ===")
    print(f"failed requests:        {sum(len(r['errors']) for r in records)}")
    print(f"deterministic cases:    {sum(r['deterministic'] for r in records)}/{len(records)}")
    print(f"flagged evidence lines: {bad_lines}/{evidence_lines}")
    print(f"problems by kind:       {kinds or 'none'}")
    if confidences:
        print(
            f"confidence:             min {min(confidences)}, max {max(confidences)}, "
            f"mean {statistics.mean(confidences):.0f}, "
            f"distinct values {sorted(set(confidences))}"
        )
    for record in records:
        for error in record["errors"][:1]:
            print(f"  [{record['case']}] request failed: {error}")
        if record["answers"] and not record["deterministic"] and not record["errors"]:
            print(f"  [{record['case']}] answers differed between runs")
        for index, kind, detail, evidence in (record["problems"] or [[]])[0]:
            print(f"  [{record['case']}] {kind}: {detail}\n      evidence: {evidence}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=[LLM.MODEL])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", help="write every answer to this JSON file")
    args = parser.parse_args()

    if not ollama_available():
        raise SystemExit("Ollama is not running on localhost:11434.")

    results = {}
    for model in args.models:
        results[model] = evaluate(model, args.runs)
        summarise(model, results[model])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=1)
        print(f"\nAll answers written to {args.out}")


if __name__ == "__main__":
    main()
