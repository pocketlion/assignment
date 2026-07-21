"""run_eval.py — evaluate the Wikipedia agent with an LLM judge.

For every case in eval_cases.jsonl this runs agent.answer_question, then grades
the result with a *different* Anthropic model using the rubric in
rater_prompt.txt (loaded from disk, never hard-coded). Each run writes a
timestamped report to eval_run/run_<N>.json, where N increments per run.

Usage:
    python run_eval.py                      # grade every case
    python run_eval.py --limit 5            # first 5 cases (quick smoke)
    python run_eval.py --id sf-01 mh-02     # only these case ids
    python run_eval.py --cases c.jsonl --rater r.txt --out-dir eval_run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from statistics import mean

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
CRITERIA_SHORT = {
    "overall_correctness": "correct",
    "groundedness": "grounded",
    "completeness": "complete",
    "search_calibration": "search",
    "tool_use_fidelity": "tools",
}
# Criteria surfaced in console logging (latency/token metrics stay in the report only).
LOG_CRITERIA = ["overall_correctness", "groundedness", "completeness",
                "search_calibration", "tool_use_fidelity"]

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


def select_cases(cases, ids):
    """Filter `cases` to those whose id is requested, preserving file order.

    `ids` is a list of tokens, each of which may itself be comma-separated
    (so --id sf-01 mh-02 and --id sf-01,mh-02 both work). Returns
    (selected_cases, sorted_missing_ids).
    """
    wanted = {i.strip() for tok in ids for i in tok.split(",") if i.strip()}
    selected = [c for c in cases if c.get("id") in wanted]
    missing = sorted(wanted - {c.get("id") for c in cases})
    return selected, missing


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
            "latency_s": tc.get("latency_s"),
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
        "model_call_latencies_s": agent_result.get("model_call_latencies_s", []),
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


def _percentile(sorted_vals, p):
    """Linear-interpolation percentile (numpy 'linear' method)."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _stats(values):
    """mean / p90 / p95 (plus count) over non-None numeric values."""
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "p90": None, "p95": None}
    s = sorted(values)
    return {
        "count": len(values),
        "mean": round(mean(values), 3),
        "p90": round(_percentile(s, 90), 3),
        "p95": round(_percentile(s, 95), 3),
    }


def performance_summary(records):
    """Latency & token distributions across all graded cases.

    Latencies are pooled over individual calls (every model call / every tool
    call across all cases); token counts are per-case totals.
    """
    model_lat, tool_lat, in_tok, out_tok = [], [], [], []
    for r in records:
        model_lat += r.get("model_call_latencies_s") or []
        tool_lat += [tc["latency_s"] for tc in r.get("tool_calls", [])
                     if tc.get("latency_s") is not None]
        u = r.get("usage") or {}
        if u.get("input_tokens") is not None:
            in_tok.append(u["input_tokens"])
        if u.get("output_tokens") is not None:
            out_tok.append(u["output_tokens"])
    return {
        "model_call_latency_s": _stats(model_lat),
        "tool_call_latency_s": _stats(tool_lat),
        "input_tokens_per_case": _stats(in_tok),
        "output_tokens_per_case": _stats(out_tok),
    }


def _mark(v):
    return "✓" if v is True else ("✗" if v is False else "?")


def log_result(record, index, total, stream=sys.stderr):
    """One compact line per case: id + pass/fail for the logged criteria."""
    g = record["grades"]
    marks = "  ".join(f"{_mark(g.get(c))} {CRITERIA_SHORT[c]}" for c in LOG_CRITERIA)
    print(f"[{index}/{total}] {record['id']:<7} {marks}", file=stream)


def failing_ids(records, criteria=None):
    """(id, [failed criteria]) for cases failing any logged criterion (None=judge error, not a fail)."""
    criteria = criteria or LOG_CRITERIA
    out = []
    for r in records:
        failed = [c for c in criteria if r["grades"].get(c) is False]
        if failed:
            out.append((r["id"], failed))
    return out


def run(cases, system_prompt, rater_template, answer_fn=None, judge_fn=None, progress=True):
    answer_fn = answer_fn or agent.answer_question
    judge_fn = judge_fn or make_judge(agent._client())
    records = []
    total = len(cases)
    for i, case in enumerate(cases, 1):
        record = grade_case(case, system_prompt, rater_template, answer_fn, judge_fn)
        records.append(record)
        if progress:
            log_result(record, i, total)   # one compact line, emitted immediately
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
        "performance": performance_summary(records),
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
    p.add_argument("--id", nargs="+", default=None, metavar="ID",
                   help="only run cases with these ids (space- or comma-separated, "
                        "e.g. --id sf-01 mh-02  or  --id sf-01,mh-02)")
    p.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    args = p.parse_args(argv)

    cases = load_cases(args.cases)
    if args.id:
        cases, missing = select_cases(cases, args.id)
        if missing:
            print(f"warning: no case(s) matched id(s): {', '.join(missing)}", file=sys.stderr)
        if not cases:
            print("No cases to run after --id filter. Exiting.", file=sys.stderr)
            return
    if args.limit:
        cases = cases[: args.limit]
    rater_template = load_rater_template(args.rater)
    system_prompt = agent._load_system_prompt()

    records, metrics = run(cases, system_prompt, rater_template)
    path, n = next_report_path(args.out_dir)
    report = build_report(n, system_prompt, records, metrics)
    write_report(path, report)

    print(f"\nRun {n}: {len(records)} cases (agent={agent.MODEL}, judge={JUDGE_MODEL})")
    for c in LOG_CRITERIA:
        v = metrics[c]
        print(f"  {c:20s} {v:5.1f}%" if v is not None else f"  {c:20s}    n/a")

    fails = failing_ids(records)
    if fails:
        print(f"\n  failing ({len(fails)}):")
        for cid, failed in fails:
            print(f"    {cid}: {', '.join(failed)}")
    else:
        print("\n  no failing cases")

    errored = [r["id"] for r in records if r.get("judge_error")]
    if errored:
        print(f"\n  unevaluated (judge error): {', '.join(errored)}")

    print(f"\nFull metrics, latency & per-case details in: {path}")


if __name__ == "__main__":
    main()
