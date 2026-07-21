"""Tests for run_eval.py — fully offline (fake agent + fake judge).

    python -m unittest test_run_eval -v
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import run_eval as R

TEMPLATE = (
    "Q: {question}\nCategory: {category}\nRef: {reference_answer}\n"
    "KW: {gold_keywords}\nExpectSearch: {expected_search}\n"
    "ExpectTools: {expected_tool_calls}\n\nResult:\n{agent_result}\n\n"
    'Output ONLY: {"overall_correctness": bool, "groundedness": bool}'
)

CASE = {
    "id": "t-01", "category": "simple_factual", "question": "Capital of France?",
    "reference_answer": "Paris.", "gold_keywords": ["paris"],
    "expected_search": True, "expected_tool_calls": ["search_wikipedia"],
}

AGENT_RESULT = {
    "question": "Capital of France?", "answer": "Paris.", "searched": True,
    "tool_calls": [{"tool": "search_wikipedia", "input": {"query": "France capital"},
                    "wiki_url": None, "retrieved_content": "Paris is the capital.",
                    "retrieved_content_preview": "Paris is the capital."}],
    "num_iterations": 2, "stop_reason": "answered", "usage": {"total_tokens": 42},
}


class RenderTests(unittest.TestCase):
    def test_fills_placeholders_and_preserves_rubric_json(self):
        out = R.render_rater_prompt(TEMPLATE, CASE, AGENT_RESULT)
        for ph in R.CASE_PLACEHOLDERS + ["agent_result"]:
            self.assertNotIn("{" + ph + "}", out)  # every placeholder filled
        self.assertIn("Capital of France?", out)
        self.assertIn('"query": "France capital"', out)  # agent_result embedded
        # The rubric's literal example JSON must survive untouched.
        self.assertIn('{"overall_correctness": bool, "groundedness": bool}', out)

    def test_none_reference_renders_as_null(self):
        case = dict(CASE, reference_answer=None)
        out = R.render_rater_prompt(TEMPLATE, case, AGENT_RESULT)
        self.assertIn("Ref: null", out)


class ExtractJsonTests(unittest.TestCase):
    def test_bare(self):
        self.assertEqual(R._extract_json('{"a": true}'), {"a": True})

    def test_fenced(self):
        self.assertEqual(R._extract_json('```json\n{"a": false}\n```'), {"a": False})

    def test_prose_wrapped(self):
        self.assertEqual(R._extract_json('Verdict:\n{"a": true, "b": false}\ndone')["b"], False)


class SummarizeTests(unittest.TestCase):
    def test_percentages_skip_none(self):
        records = [
            {"grades": {c: True for c in R.CRITERIA}},
            {"grades": {c: (False if c == "groundedness" else True) for c in R.CRITERIA}},
            {"grades": {c: None for c in R.CRITERIA}},  # judge error -> excluded
        ]
        m = R.summarize(records)
        self.assertEqual(m["overall_correctness"], 100.0)  # 2/2 (None skipped)
        self.assertEqual(m["groundedness"], 50.0)          # 1/2

    def test_all_none_gives_none(self):
        records = [{"grades": {c: None for c in R.CRITERIA}}]
        self.assertIsNone(R.summarize(records)["groundedness"])


class ReportPathTests(unittest.TestCase):
    def test_increments(self):
        d = tempfile.mkdtemp()
        p1, n1 = R.next_report_path(d)
        self.assertEqual((os.path.basename(p1), n1), ("run_001.json", 1))
        open(p1, "w").close()  # now exists
        p2, n2 = R.next_report_path(d)
        self.assertEqual((os.path.basename(p2), n2), ("run_002.json", 2))

    def test_ignores_non_matching_files(self):
        d = tempfile.mkdtemp()
        open(os.path.join(d, "notes.txt"), "w").close()
        open(os.path.join(d, "run_005.json"), "w").close()
        _, n = R.next_report_path(d)
        self.assertEqual(n, 6)


class RunEndToEndTests(unittest.TestCase):
    def test_run_with_fakes(self):
        cases = [
            CASE,
            dict(CASE, id="t-02", question="Unknowable?", reference_answer="Unknown.",
                 expected_search=False, expected_tool_calls=[], gold_keywords=[]),
        ]

        def fake_answer(q, sp):
            return dict(AGENT_RESULT, question=q, answer="Paris.")

        def fake_judge(prompt):
            if "Unknowable?" in prompt:
                raise RuntimeError("boom")  # exercise judge-error path
            return {c: True for c in R.CRITERIA} | {"judge_reasoning": "ok"}

        records, metrics = R.run(cases, "SYS", TEMPLATE,
                                 answer_fn=fake_answer, judge_fn=fake_judge, progress=False)

        self.assertEqual(len(records), 2)
        self.assertIsNone(records[0]["judge_error"])
        self.assertTrue(records[1]["judge_error"].startswith("RuntimeError"))
        self.assertTrue(all(v is None for v in records[1]["grades"].values()))
        self.assertEqual(metrics["overall_correctness"], 100.0)  # only the 1 valid case counts

        report = R.build_report(1, "SYS", records, metrics)
        for key in ("run_at", "run_at_readable", "agent_model", "judge_model",
                    "metrics_percent", "performance", "results", "num_cases", "judge_errors"):
            self.assertIn(key, report)
        self.assertEqual(report["judge_errors"], 1)


class JudgeCallTests(unittest.TestCase):
    """make_judge must omit temperature when None (Opus 4.8) and send it otherwise."""
    def _recording_client(self):
        rec = {}
        block = type("Blk", (), {"type": "text", "text": '{"overall_correctness": true}'})()
        resp = type("Resp", (), {"content": [block]})()

        class Client:
            def create(self, **kwargs):
                rec["kwargs"] = kwargs
                return resp
        c = Client()
        c.messages = c
        return c, rec

    def test_omits_temperature_when_none(self):
        c, rec = self._recording_client()
        with patch.object(R, "JUDGE_TEMPERATURE", None):
            verdict = R.make_judge(c)("prompt")
        self.assertNotIn("temperature", rec["kwargs"])
        self.assertTrue(verdict["overall_correctness"])

    def test_includes_temperature_when_set(self):
        c, rec = self._recording_client()
        with patch.object(R, "JUDGE_TEMPERATURE", 0):
            R.make_judge(c)("prompt")
        self.assertEqual(rec["kwargs"]["temperature"], 0)


class StatsTests(unittest.TestCase):
    def test_percentiles_linear(self):
        s = R._stats(list(range(1, 11)))  # 1..10
        self.assertEqual(s["count"], 10)
        self.assertEqual(s["mean"], 5.5)
        self.assertEqual(s["p90"], 9.1)    # numpy 'linear' method
        self.assertEqual(s["p95"], 9.55)

    def test_single_and_empty(self):
        self.assertEqual(R._stats([2.0])["p95"], 2.0)
        empty = R._stats([])
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["mean"])
        self.assertIsNone(empty["p90"])


class PerformanceTests(unittest.TestCase):
    def test_pools_latencies_and_tokens(self):
        records = [
            {"model_call_latencies_s": [0.1, 0.3], "tool_calls": [{"latency_s": 0.5}],
             "usage": {"input_tokens": 100, "output_tokens": 10}},
            {"model_call_latencies_s": [0.2],
             "tool_calls": [{"latency_s": 0.7}, {"latency_s": 0.9}],
             "usage": {"input_tokens": 200, "output_tokens": 20}},
        ]
        perf = R.performance_summary(records)
        self.assertEqual(perf["model_call_latency_s"]["count"], 3)   # pooled over calls
        self.assertAlmostEqual(perf["model_call_latency_s"]["mean"], 0.2)
        self.assertEqual(perf["tool_call_latency_s"]["count"], 3)
        self.assertAlmostEqual(perf["tool_call_latency_s"]["mean"], 0.7)
        self.assertEqual(perf["input_tokens_per_case"]["count"], 2)  # per-case
        self.assertAlmostEqual(perf["input_tokens_per_case"]["mean"], 150)
        self.assertAlmostEqual(perf["output_tokens_per_case"]["mean"], 15)

    def test_missing_fields_are_safe(self):
        perf = R.performance_summary([{"usage": {}}])  # no latencies, no tokens
        self.assertEqual(perf["model_call_latency_s"]["count"], 0)
        self.assertIsNone(perf["input_tokens_per_case"]["mean"])


class LogResultTests(unittest.TestCase):
    def test_writes_one_compact_line(self):
        import io
        rec = {"id": "t-1", "grades": {c: True for c in R.CRITERIA}}
        buf = io.StringIO()
        R.log_result(rec, 1, 3, stream=buf)
        out = buf.getvalue()
        self.assertIn("t-1", out)
        for c in R.LOG_CRITERIA:                       # all logged criteria shown
            self.assertIn(R.CRITERIA_SHORT[c], out)
        self.assertIn("grounded", out)                 # groundedness now included
        self.assertEqual(out.count("\n"), 1)           # still exactly one line

    def test_marks_reflect_grades(self):
        import io
        grades = {c: True for c in R.CRITERIA}
        grades["completeness"] = False
        buf = io.StringIO()
        R.log_result({"id": "t-2", "grades": grades}, 2, 3, stream=buf)
        out = buf.getvalue()
        self.assertIn("✗ complete", out)
        self.assertIn("✓ correct", out)


class FailingIdsTests(unittest.TestCase):
    def _rec(self, cid, **overrides):
        grades = {c: True for c in R.CRITERIA}
        grades.update(overrides)
        return {"id": cid, "grades": grades}

    def test_lists_failures_with_criteria(self):
        records = [
            self._rec("a"),
            self._rec("b", completeness=False),
            self._rec("c", overall_correctness=False, tool_use_fidelity=False),
            self._rec("d", **{c: None for c in R.CRITERIA}),  # judge error -> not a "fail"
        ]
        fails = R.failing_ids(records)
        self.assertEqual([cid for cid, _ in fails], ["b", "c"])
        self.assertEqual(dict(fails)["c"], ["overall_correctness", "tool_use_fidelity"])

    def test_groundedness_counted(self):
        # groundedness is now in LOG_CRITERIA, so a groundedness failure is flagged.
        records = [self._rec("a", groundedness=False)]
        self.assertEqual(R.failing_ids(records), [("a", ["groundedness"])])


class SelectCasesTests(unittest.TestCase):
    CASES = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    def test_filters_preserving_file_order(self):
        selected, missing = R.select_cases(self.CASES, ["c", "a"])
        self.assertEqual([c["id"] for c in selected], ["a", "c"])  # file order, not arg order
        self.assertEqual(missing, [])

    def test_comma_separated_token(self):
        selected, _ = R.select_cases(self.CASES, ["a,b"])
        self.assertEqual([c["id"] for c in selected], ["a", "b"])

    def test_mixed_and_whitespace(self):
        selected, _ = R.select_cases(self.CASES, [" a , c ", "b"])
        self.assertEqual([c["id"] for c in selected], ["a", "b", "c"])

    def test_reports_missing(self):
        selected, missing = R.select_cases(self.CASES, ["a", "zzz", "qqq"])
        self.assertEqual([c["id"] for c in selected], ["a"])
        self.assertEqual(missing, ["qqq", "zzz"])  # sorted


if __name__ == "__main__":
    unittest.main(verbosity=2)
