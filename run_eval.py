"""run_eval.py — evaluate the Wikipedia agent with an LLM judge.

For every case in eval_cases.jsonl this runs agent.answer_question, then grades
the result with a *different* Anthropic model using the rubric in
rater_prompt.txt (loaded from disk, never hard-coded). Each run writes a
timestamped report to eval_run/run_<N>.json, where N increments per run.

Usage:
    python run_eval.py                      # grade every case
    python run_eval.py --limit 5            # first 5 cases (quick smoke)
    python run_eval.py --cases c.jsonl --rater r.txt --out-dir eval_run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

import agent

# --- Config -----------------------------------------------------------------

# The judge must differ from the agent's model (agent.MODEL) so a model doesn't
# grade its own work. Change this to a judge model you have access to.
JUDGE_MODEL = "claude-opus-4-8"
JUDGE_MAX_TOKENS = 1024
# Sampling control: Opus 4.7+ (including claude-opus-4-8) DEPRECATED
# temperature/top_p/top_k — sending any of them returns HTTP 400, so the field
# must be omitted. Leave JUDGE_TEMPERATURE=None for Opus 4.8. If you switch
# JUDGE_MODEL to a family that still supports sampling params (Sonnet/Haiku),
# set JUDGE_TEMPERATURE=0 for steadier grading. Note: temperature=0 was never a
# hard determinism guarantee even where supported — the strict rubric does more.
JUDGE_TEMPERATURE = None  # None => omit the field entirely

CASES_PATH = "eval_cases.jsonl"
RATER_PATH = "rater_prompt.txt"
REPORT_DIR = "eval_run"

# Boolean verdict fields the rater returns; also the metrics we report.
CRITERIA = [
    "overall_correctness",
    "groundedness",
    "completeness",
    "search_calibration",
    "tool_use_fidelity",
]

# rater_prompt.txt placeholders we fill. We replace these explicitly (rather
# than str.format) so the literal { } in the rubric's JSON example are left
# untouched.
CASE_PLACEHOLDERS = [
    "question", "category", "reference_answer",
    "gold_keywords", "expected_search", "expected_tool_calls",
]


# --- Loading ----------------------------------------------------------------

def load_cases(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rater_template(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# --- Prompt rendering -------------------------------------------------------

def _as_text(value):
    """Render a case field for insertion into the prompt."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def render_rater_prompt(template, case, agent_result):
    """Fill the rater template with case fields + the agent's full result."""
    out = template
    for key in CASE_PLACEHOLDERS:
        out = out.replace("{" + key + "}", _as_text(case.get(key)))
    out = out.replace("{agent_result}", json.dumps(agent_result, ensure_ascii=False, indent=2))
    return out


# --- Judge ------------------------------------------------------------------

def _extract_json(text):
    """Parse the judge's JSON, tolerating code fences or surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)  # first {...} block
        if not match:
            raise
        return json.loads(match.group(0))


def make_judge(client):
    """Return judge(prompt) -> verdict dict, calling the judge model."""
    def judge(prompt):
        kwargs = {
            "model": JUDGE_MODEL,
            "max_tokens": JUDGE_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Only send temperature when set — Opus 4.7+ reject it (see config note).
        if JUDGE_TEMPERATURE is not None:
            kwargs["temperature"] = JUDGE_TEMPERATURE
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return _extract_json(text)
    return judge


# --- Grading ----------------------------------------------------------------

def _tool_summary(tool_calls):
    """Compact tool-call view for the report (drops the full retrieved_content)."""
    return [
        {
            "tool": tc.get("tool"),
            "input": tc.get("input"),
            "wiki_url": tc.get("wiki_url"),
            "retrieved_content_preview": tc.get("retrieved_content_preview"),
        }
        for tc in tool_calls
    ]


def build_record(case, agent_result, verdict, judge_error):
    grades = {c: (verdict.get(c) if isinstance(verdict, dict) else None) for c in CRITERIA}
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": case.get("question"),
        "reference_answer": case.get("reference_answer"),
        "gold_keywords": case.get("gold_keywords", []),
        "expected_search": case.get("expected_search"),
        "expected_tool_calls": case.get("expected_tool_calls", []),
        "agent_answer": agent_result.get("answer"),
        "searched": agent_result.get("searched"),
        "stop_reason": agent_result.get("stop_reason"),
        "num_iterations": agent_result.get("num_iterations"),
        "tool_calls": _tool_summary(agent_result.get("tool_calls", [])),
        "usage": agent_result.get("usage"),
        "grades": grades,
        "judge_reasoning": verdict.get("judge_reasoning") if isinstance(verdict, dict) else None,
        "judge_error": judge_error,
    }


def grade_case(case, system_prompt, rater_template, answer_fn, judge_fn):
    agent_result = answer_fn(case.get("question", ""), system_prompt)
    prompt = render_rater_prompt(rater_template, case, agent_result)
    verdict, judge_error = None, None
    try:
        verdict = judge_fn(prompt)
    except Exception as e:  # never let one bad grade abort the whole run
        judge_error = f"{type(e).__name__}: {e}"
    return build_record(case, agent_result, verdict, judge_error)


def summarize(records):
    """Percentage of cases passing each criterion (over cases with a bool verdict)."""
    metrics = {}
    for c in CRITERIA:
        vals = [r["grades"][c] for r in records if isinstance(r["grades"][c], bool)]
        metrics[c] = round(100 * sum(vals) / len(vals), 1) if vals else None
    return metrics


def run(cases, system_prompt, rater_template, answer_fn=None, judge_fn=None, progress=True):
    answer_fn = answer_fn or agent.answer_question
    judge_fn = judge_fn or make_judge(agent._client())
    records = []
    for i, case in enumerate(cases, 1):
        if progress:
            print(f"[{i}/{len(cases)}] {case.get('id','?')}: {case.get('question','')[:60]}",
                  file=sys.stderr)
        records.append(grade_case(case, system_prompt, rater_template, answer_fn, judge_fn))
    return records, summarize(records)


# --- Reporting --------------------------------------------------------------

def next_report_path(report_dir):
    os.makedirs(report_dir, exist_ok=True)
    nums = [int(m.group(1)) for name in os.listdir(report_dir)
            if (m := re.fullmatch(r"run_(\d+)\.json", name))]
    n = (max(nums) + 1) if nums else 1
    return os.path.join(report_dir, f"run_{n:03d}.json"), n


def build_report(run_number, system_prompt, records, metrics):
    now = dt.datetime.now().astimezone()
    return {
        "run_number": run_number,
        "run_at": now.isoformat(timespec="seconds"),
        "run_at_readable": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "agent_model": agent.MODEL,
        "judge_model": JUDGE_MODEL,
        "system_prompt": system_prompt,
        "num_cases": len(records),
        "judge_errors": sum(1 for r in records if r["judge_error"]),
        "metrics_percent": metrics,
        "results": records,
    }


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# --- CLI --------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the Wikipedia agent with an LLM judge.")
    p.add_argument("--cases", default=CASES_PATH)
    p.add_argument("--rater", default=RATER_PATH)
    p.add_argument("--out-dir", default=REPORT_DIR)
    p.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    args = p.parse_args(argv)

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    rater_template = load_rater_template(args.rater)
    system_prompt = agent._load_system_prompt()

    records, metrics = run(cases, system_prompt, rater_template)
    path, n = next_report_path(args.out_dir)
    write_report(path, build_report(n, system_prompt, records, metrics))

    print(f"\nRun {n}: graded {len(records)} cases (agent={agent.MODEL}, judge={JUDGE_MODEL})")
    for c in CRITERIA:
        v = metrics[c]
        print(f"  {c:20s} {v:5.1f}%" if v is not None else f"  {c:20s}    n/a")
    errs = sum(1 for r in records if r["judge_error"])
    if errs:
        print(f"  ({errs} case(s) had judge errors — see report)")
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
