"""Tests for wiki.py.

All tests mock wiki.SESSION.get, so they run fully offline and are
deterministic — no real Wikipedia calls, no flakiness.

An optional live smoke test hits the real API; it is skipped unless you set
WIKI_LIVE_TESTS=1 in the environment (and have network access):

    WIKI_LIVE_TESTS=1 python -m unittest test_wiki -v
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import wiki


def fake_response(payload):
    """Build a stand-in for a requests.Response returning `payload` from .json()."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


# --- search_wikipedia -------------------------------------------------------

class SearchWikipediaTests(unittest.TestCase):
    @patch.object(wiki.SESSION, "get")
    def test_returns_stripped_snippets(self, mock_get):
        mock_get.return_value = fake_response(
            {
                "query": {
                    "search": [
                        {
                            "title": "Python (programming language)",
                            "snippet": 'A <span class="searchmatch">Python</span> '
                            "interpreter &amp; runtime",
                        },
                        {
                            "title": "Monty Python",
                            "snippet": "British <b>comedy</b> troupe",
                        },
                    ]
                }
            }
        )

        results = wiki.search_wikipedia("python")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Python (programming language)")
        # Tags removed, &amp; unescaped to &.
        self.assertEqual(results[0]["snippet"], "A Python interpreter & runtime")
        self.assertEqual(results[1]["snippet"], "British comedy troupe")
        # No leftover HTML anywhere.
        for r in results:
            self.assertNotIn("<", r["snippet"])
            self.assertNotIn(">", r["snippet"])

    @patch.object(wiki.SESSION, "get")
    def test_sends_correct_query_params(self, mock_get):
        mock_get.return_value = fake_response({"query": {"search": []}})

        wiki.search_wikipedia("quantum entanglement")

        _, kwargs = mock_get.call_args
        params = kwargs["params"]
        self.assertEqual(params["action"], "query")
        self.assertEqual(params["list"], "search")
        self.assertEqual(params["srsearch"], "quantum entanglement")
        self.assertEqual(params["srlimit"], 5)
        self.assertEqual(params["format"], "json")

    @patch.object(wiki.SESSION, "get")
    def test_no_results_returns_message(self, mock_get):
        mock_get.return_value = fake_response({"query": {"search": []}})

        result = wiki.search_wikipedia("asdkjfhaksjdfhqwoeiu")

        self.assertIsInstance(result, str)
        self.assertEqual(
            result,
            "No results for 'asdkjfhaksjdfhqwoeiu'. "
            "Try broader or alternative search terms.",
        )

    @patch.object(wiki.SESSION, "get")
    def test_missing_query_key_treated_as_no_results(self, mock_get):
        mock_get.return_value = fake_response({})  # e.g. odd/empty response
        result = wiki.search_wikipedia("edge case")
        self.assertIsInstance(result, str)
        self.assertIn("No results for 'edge case'", result)


# --- get_article ------------------------------------------------------------

class GetArticleTests(unittest.TestCase):
    @patch.object(wiki.SESSION, "get")
    def test_plain_article(self, mock_get):
        mock_get.return_value = fake_response(
            {
                "query": {
                    "pages": {
                        "12345": {
                            "pageid": 12345,
                            "title": "Photosynthesis",
                            "extract": "Photosynthesis is a process used by plants.",
                        }
                    }
                }
            }
        )

        result = wiki.get_article("Photosynthesis")

        self.assertEqual(result, "Photosynthesis is a process used by plants.")
        self.assertNotIn("[Redirected to:", result)
        self.assertNotIn("[Article truncated]", result)

    @patch.object(wiki.SESSION, "get")
    def test_sends_correct_query_params(self, mock_get):
        mock_get.return_value = fake_response(
            {"query": {"pages": {"1": {"title": "X", "extract": "hi"}}}}
        )

        wiki.get_article("Some Title")

        _, kwargs = mock_get.call_args
        params = kwargs["params"]
        self.assertEqual(params["action"], "query")
        self.assertEqual(params["prop"].split("|")[0], "extracts")
        self.assertIn("extracts", params["prop"])
        self.assertEqual(params["explaintext"], 1)
        self.assertEqual(params["redirects"], 1)
        self.assertEqual(params["titles"], "Some Title")

    @patch.object(wiki.SESSION, "get")
    def test_redirect_prefix(self, mock_get):
        mock_get.return_value = fake_response(
            {
                "query": {
                    "redirects": [{"from": "NYC", "to": "New York City"}],
                    "pages": {
                        "645042": {
                            "title": "New York City",
                            "extract": "New York City is the most populous city in the US.",
                        }
                    },
                }
            }
        )

        result = wiki.get_article("NYC")

        self.assertTrue(result.startswith("[Redirected to: New York City]"))
        self.assertIn("most populous city", result)

    @patch.object(wiki.SESSION, "get")
    def test_disambiguation_page(self, mock_get):
        mock_get.return_value = fake_response(
            {
                "query": {
                    "pages": {
                        "18978754": {
                            "title": "Mercury",
                            "extract": "Mercury may refer to:",
                            "pageprops": {"disambiguation": ""},
                            "links": [
                                {"ns": 0, "title": "Mercury (element)"},
                                {"ns": 0, "title": "Mercury (planet)"},
                                {"ns": 0, "title": "Mercury (mythology)"},
                            ],
                        }
                    }
                }
            }
        )

        result = wiki.get_article("Mercury")

        self.assertTrue(result.startswith("'Mercury' is a disambiguation page."))
        self.assertIn("Options:", result)
        self.assertIn("Mercury (element)", result)
        self.assertIn("Mercury (planet)", result)
        self.assertIn("Mercury (mythology)", result)

    @patch.object(wiki.SESSION, "get")
    def test_missing_page(self, mock_get):
        mock_get.return_value = fake_response(
            {
                "query": {
                    "pages": {
                        "-1": {
                            "ns": 0,
                            "title": "Thisdefinitelydoesnotexist12345",
                            "missing": "",
                        }
                    }
                }
            }
        )

        result = wiki.get_article("Thisdefinitelydoesnotexist12345")

        self.assertEqual(
            result,
            "No article found for 'Thisdefinitelydoesnotexist12345'. "
            "Use search_wikipedia to find the correct title.",
        )

    @patch.object(wiki.SESSION, "get")
    def test_empty_pages_returns_not_found(self, mock_get):
        mock_get.return_value = fake_response({"query": {}})
        result = wiki.get_article("Whatever")
        self.assertIn("No article found for 'Whatever'", result)

    @patch.object(wiki.SESSION, "get")
    def test_truncation(self, mock_get):
        long_extract = "A" * (wiki.MAX_EXTRACT_CHARS + 5000)
        mock_get.return_value = fake_response(
            {
                "query": {
                    "pages": {
                        "1": {"title": "Long", "extract": long_extract}
                    }
                }
            }
        )

        result = wiki.get_article("Long")

        self.assertTrue(result.endswith("[Article truncated]"))
        # Body (minus the marker) should be about the cap, never far over it.
        body = result.replace("\n\n[Article truncated]", "")
        self.assertLessEqual(len(body), wiki.MAX_EXTRACT_CHARS)
        self.assertGreater(len(body), wiki.MAX_EXTRACT_CHARS - 100)

    @patch.object(wiki.SESSION, "get")
    def test_no_truncation_when_short(self, mock_get):
        mock_get.return_value = fake_response(
            {"query": {"pages": {"1": {"title": "Short", "extract": "brief"}}}}
        )
        result = wiki.get_article("Short")
        self.assertEqual(result, "brief")

    @patch.object(wiki.SESSION, "get")
    def test_redirect_and_truncation_combined(self, mock_get):
        long_extract = "B" * (wiki.MAX_EXTRACT_CHARS + 1)
        mock_get.return_value = fake_response(
            {
                "query": {
                    "redirects": [{"from": "Old", "to": "New"}],
                    "pages": {"1": {"title": "New", "extract": long_extract}},
                }
            }
        )
        result = wiki.get_article("Old")
        self.assertTrue(result.startswith("[Redirected to: New]"))
        self.assertTrue(result.endswith("[Article truncated]"))


# --- helper unit tests ------------------------------------------------------

class StripHtmlTests(unittest.TestCase):
    def test_removes_tags_and_unescapes(self):
        self.assertEqual(
            wiki._strip_html('x <span class="s">y</span> &amp; z'),
            "x y & z",
        )

    def test_empty(self):
        self.assertEqual(wiki._strip_html(""), "")
        self.assertEqual(wiki._strip_html(None), "")


# --- retry behaviour in _api_get -------------------------------------------

class ApiRetryTests(unittest.TestCase):
    @patch.object(wiki.time, "sleep")  # don't actually wait
    @patch.object(wiki.SESSION, "get")
    def test_retries_then_succeeds(self, mock_get, mock_sleep):
        # Two transient failures, then a good response.
        mock_get.side_effect = [
            wiki.requests.exceptions.Timeout("timeout"),
            wiki.requests.exceptions.ConnectionError("conn reset"),
            fake_response({"query": {"search": []}}),
        ]

        result = wiki.search_wikipedia("anything")

        self.assertIsInstance(result, str)  # zero-results message; call succeeded
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # backoff between the 3 attempts

    @patch.object(wiki.time, "sleep")
    @patch.object(wiki.SESSION, "get")
    def test_gives_up_after_max_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = wiki.requests.exceptions.Timeout("always down")

        with self.assertRaises(wiki.requests.exceptions.RequestException):
            wiki.get_article("Whatever")

        self.assertEqual(mock_get.call_count, wiki.MAX_RETRIES)

    @patch.object(wiki.time, "sleep")
    @patch.object(wiki.SESSION, "get")
    def test_success_on_first_try_does_not_sleep(self, mock_get, mock_sleep):
        mock_get.return_value = fake_response(
            {"query": {"pages": {"1": {"title": "X", "extract": "hi"}}}}
        )
        wiki.get_article("X")
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()


# --- optional live smoke test (skipped by default) --------------------------

@unittest.skipUnless(
    os.environ.get("WIKI_LIVE_TESTS") == "1",
    "set WIKI_LIVE_TESTS=1 to run live network tests",
)
class LiveSmokeTests(unittest.TestCase):
    def test_search_live(self):
        results = wiki.search_wikipedia("Alan Turing")
        self.assertIsInstance(results, list)
        self.assertTrue(any("Turing" in r["title"] for r in results))

    def test_article_live(self):
        text = wiki.get_article("Alan Turing")
        self.assertIn("Turing", text)

    def test_redirect_live(self):
        text = wiki.get_article("NYC")
        self.assertTrue(text.startswith("[Redirected to:"))

    def test_missing_live(self):
        text = wiki.get_article("Zzxqwv Nonexistent Article 99999")
        self.assertIn("No article found", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
