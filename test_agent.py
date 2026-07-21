"""Tests for agent.py.

No network and no real Anthropic SDK required: we patch agent._client with a
fake client that returns canned responses, and pass a fake tools_impl.

    python -m unittest test_agent -v
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import agent


# --- Fakes mimicking the Anthropic SDK response objects ---------------------

class Block:
    """Stand-in for a content block (text or tool_use)."""
    def __init__(self, type, text=None, name=None, input=None, id="tu_1"):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


def text_block(text):
    return Block("text", text=text)


def tool_block(name, input, id="tu_1"):
    return Block("tool_use", name=name, input=input, id=id)


class Usage:
    def __init__(self, input_tokens=10, output_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class Response:
    def __init__(self, content, stop_reason, usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or Usage()


class FakeClient:
    """Pops queued responses in order. client.messages.create -> self.create."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class LoopingClient:
    """Always returns the same response (used to force max-iterations)."""
    def __init__(self, response):
        self._response = response
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


FAKE_TOOLS = {
    "search_wikipedia": lambda query: [{"title": "Marie Curie", "snippet": "physicist"}],
    "get_article": lambda title: f"Extract of {title}.",
}


# --- Tests ------------------------------------------------------------------

class AgentTests(unittest.TestCase):

    def test_uses_tool_then_answers(self):
        client = FakeClient([
            Response([tool_block("search_wikipedia", {"query": "marie curie"})], "tool_use"),
            Response([text_block("Marie Curie won her first Nobel Prize in 1903.")], "end_turn"),
        ])

        with patch.object(agent, "_client", return_value=client):
            res = agent.answer_question("When did Marie Curie win?", "sys", FAKE_TOOLS)

        self.assertEqual(res["stop_reason"], "answered")
        self.assertTrue(res["searched"])
        self.assertEqual(res["num_iterations"], 2)
        self.assertEqual(res["answer"], "Marie Curie won her first Nobel Prize in 1903.")

        self.assertEqual(len(res["tool_calls"]), 1)
        call = res["tool_calls"][0]
        self.assertEqual(call["tool"], "search_wikipedia")
        self.assertEqual(call["input"], {"query": "marie curie"})
        self.assertEqual(call["result_retrieved"], [{"title": "Marie Curie", "snippet": "physicist"}])

        # Usage summed across both API calls (10+5 each).
        self.assertEqual(res["usage"]["input_tokens"], 20)
        self.assertEqual(res["usage"]["output_tokens"], 10)
        self.assertEqual(res["usage"]["total_tokens"], 30)

        # Second API call must include the tool_result we sent back.
        second_messages = client.calls[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "user")
        self.assertEqual(second_messages[-1]["content"][0]["type"], "tool_result")

    def test_max_iterations_cutoff(self):
        # Model never stops asking for a tool.
        looping = LoopingClient(
            Response(
                [text_block("still working"), tool_block("search_wikipedia", {"query": "x"})],
                "tool_use",
            )
        )

        with patch.object(agent, "_client", return_value=looping):
            res = agent.answer_question("loop forever?", "sys", FAKE_TOOLS)

        self.assertEqual(res["stop_reason"], "max_iterations")
        self.assertEqual(res["num_iterations"], agent.MAX_ITERATIONS)
        self.assertEqual(len(res["tool_calls"]), agent.MAX_ITERATIONS)
        self.assertTrue(res["searched"])
        # Whatever text the model last produced is kept.
        self.assertEqual(res["answer"], "still working")

    def test_tool_exception_surfaced_gracefully(self):
        def boom(query):
            raise RuntimeError("wikipedia is down")

        client = FakeClient([
            Response([tool_block("search_wikipedia", {"query": "anything"})], "tool_use"),
        ])
        tools = {"search_wikipedia": boom, "get_article": lambda title: "x"}

        with patch.object(agent, "_client", return_value=client):
            res = agent.answer_question("trigger a tool error", "sys", tools)  # must not raise

        self.assertEqual(res["stop_reason"], "error")
        self.assertIn("wikipedia is down", res["answer"])
        self.assertEqual(res["num_iterations"], 1)


class LoadDotenvTests(unittest.TestCase):
    def _write(self, text):
        tf = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        tf.write(text)
        tf.close()
        self.addCleanup(os.unlink, tf.name)
        return tf.name

    def test_parses_lines(self):
        path = self._write(
            "# a comment\n"
            "\n"
            "export ANT_TEST_A=plain\n"
            'ANT_TEST_B="quoted value"\n'
            "ANT_TEST_C = spaced \n"
        )
        for k in ("ANT_TEST_A", "ANT_TEST_B", "ANT_TEST_C"):
            self.addCleanup(os.environ.pop, k, None)

        agent._load_dotenv(path)

        self.assertEqual(os.environ["ANT_TEST_A"], "plain")
        self.assertEqual(os.environ["ANT_TEST_B"], "quoted value")
        self.assertEqual(os.environ["ANT_TEST_C"], "spaced")

    def test_does_not_override_existing(self):
        os.environ["ANT_TEST_EXISTING"] = "real"
        self.addCleanup(os.environ.pop, "ANT_TEST_EXISTING", None)
        path = self._write("ANT_TEST_EXISTING=from_file\n")

        agent._load_dotenv(path)

        self.assertEqual(os.environ["ANT_TEST_EXISTING"], "real")

    def test_missing_file_is_noop(self):
        agent._load_dotenv("/no/such/file/.env")  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
