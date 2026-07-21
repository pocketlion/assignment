# Running the Prototype

A small Wikipedia tool-use agent (`agent.py` + `wiki.py`) with an LLM-judged
evaluation harness (`run_eval.py`).

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic requests
```

Add your API key to a `.env` file in the project root (never committed —
already in `.gitignore`):

```
ANTHROPIC_API_KEY=sk-ant-...
```

`agent.py` loads `.env` automatically; a real exported env var always takes
precedence over the file.

## 2. Demo mode

Run one question:

```bash
python agent.py "What year did Marie Curie win her first Nobel Prize?"
```

Add `--json` to get the full structured result (answer, tool calls, token
usage, stop reason) instead of the human-readable trace:

```bash
python agent.py "What year did Marie Curie win her first Nobel Prize?" --json
```

Run the built-in set of demo questions (simple factual, multi-hop, no-search,
nonexistent-topic) back to back:

```bash
python agent.py --demo
```

## 3. Running the eval set

`run_eval.py` runs every case in `eval_cases.jsonl` through `agent.answer_question`,
grades each result with a separate judge model (`claude-opus-4-8`) using the
rubric in `rater_prompt.txt`, and writes a timestamped report to
`eval_run/run_<N>.json`.

Grade the full eval set:

```bash
python run_eval.py
```

Quick smoke test on the first few cases only:

```bash
python run_eval.py --limit 5
```

Point at a different cases file, rubric, or output directory:

```bash
python run_eval.py --cases my_cases.jsonl --rater my_rubric.txt --out-dir my_runs
```

Each run prints a pass-rate summary per criterion
(`overall_correctness`, `groundedness`, `completeness`, `search_calibration`,
`tool_use_fidelity`) and writes the full per-case report (agent answer, tool
calls, judge verdict + reasoning) to `eval_run/run_<N>.json`.

## 4. Running the tests

```bash
pip install pytest pytest-mock
python -m pytest -q
```

All API calls in the test suite are mocked, so no `ANTHROPIC_API_KEY` is
required. `test_wiki.py` also has a few live-network smoke tests that are
skipped by default — set `WIKI_LIVE=1` to enable them.
