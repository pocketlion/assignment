# Running the Prototype

A small question answering agent (`agent.py` + `wiki.py`) with an LLM-judged
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

Sample demo run:
```bash
(.venv) hysonli@Hysons-MacBook-Air Assignment % python agent.py --demo
Q: In what year did Marie Curie win her first Nobel Prize?
🔍 search_wikipedia: "Marie Curie Nobel Prize"
A: Marie Curie won her first Nobel Prize in **1903**, when she shared the Nobel Prize in Physics with her husband Pierre Curie for their joint research on radioactivity.
   [answered, 2 iter, 2950 tokens]

Q: What nationality was the scientist that curium is named after?
🔍 search_wikipedia: "curium element named after"
📄 get_article: "Marie Curie"
📄 get_article: "Pierre Curie"
A: Curium is named after both **Marie Curie** and **Pierre Curie**. Marie Curie was **Polish and naturalised French**, while Pierre Curie was **French**.
   [answered, 3 iter, 10825 tokens]

Q: What is 17 times 4?
A: 17 times 4 is **68**. This is straightforward arithmetic and requires no external lookup.
   [answered, 1 iter, 1266 tokens]

Q: Tell me about Flibberjabberwock, the ancient lost city of Mars.
🔍 search_wikipedia: "Flibberjabberwock ancient lost city Mars"
A: "Flibberjabberwock" does not appear to exist in any Wikipedia records, and there is no such thing as an "ancient lost city of Mars" — no ancient cities have ever been discovered on Mars. This appears to be a completely fictional or fabricated concept. I can't provide information about it, as there is nothing factual to share.

If you're interested in real topics, I'd be happy to help with things like **Mars exploration**, **real ancient lost cities on Earth** (such as Pompeii or Petra), or **fictional Martian civilizations in literature and film**!
   [answered, 2 iter, 2865 tokens]
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

Sample eval run:
```bash
(.venv) hysonli@Hysons-MacBook-Air Assignment % python run_eval.py 
[1/20] sf-01   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[2/20] sf-02   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[3/20] sf-03   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[4/20] sf-04   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[5/20] sf-05   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[6/20] sf-06   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[7/20] mh-01   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[8/20] mh-02   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[9/20] mh-03   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[10/20] ns-01   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[11/20] ns-02   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[12/20] ns-03   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[13/20] ae-01   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[14/20] ae-02   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[15/20] un-01   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[16/20] un-02   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[17/20] fp-01   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[18/20] fp-02   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[19/20] la-01   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools
[20/20] la-02   ✓ correct  ✓ grounded  ✓ complete  ✓ search  ✓ tools

Run 26: 20 cases (agent=claude-sonnet-4-6, judge=claude-opus-4-8)
  overall_correctness  100.0%
  groundedness         100.0%
  completeness         100.0%
  search_calibration   100.0%
  tool_use_fidelity    100.0%

  no failing cases
```

Quick smoke test on the first few cases only:

```bash
python run_eval.py --limit 5
```

Quick smoke test on certain cases with test case ids specified:

```bash
python run_eval.py --id sf-01 sf-02 
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
