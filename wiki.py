"""wiki.py — a minimal client for the live MediaWiki API (English Wikipedia).

Public functions
----------------
search_wikipedia(query) -> list[dict] | str
    Search article titles/snippets. Returns up to 5 hits, or a helpful
    message string when there are no results.

get_article(title) -> str
    Fetch the plain-text extract of an article, transparently following
    redirects and giving useful messages for disambiguation / missing pages.

Only the standard library and `requests` are used.
"""

from __future__ import annotations

import html
import re
import time

import requests

# --- Configuration ----------------------------------------------------------

API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia asks every client to send a descriptive User-Agent.
# https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "wiki.py/1.0 (educational example; contact: user@example.com) python-requests"

# "Roughly 16,000 chars" per the spec.
MAX_EXTRACT_CHARS = 16_000

# Retry transient network failures (timeouts, connection drops, HTTP 5xx)
# up to MAX_RETRIES times with a simple linear backoff.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0

# One shared session => connection reuse + one place to set headers.
# Tests patch SESSION.get, so no real network calls happen there.
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

_TAG_RE = re.compile(r"<[^>]+>")


# --- Helpers ----------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities (search snippets contain both)."""
    if not text:
        return ""
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _api_get(params: dict) -> dict:
    """GET the MediaWiki API and return parsed JSON.

    Retries up to MAX_RETRIES times on transient network errors (timeouts,
    connection drops, HTTP 5xx) with a simple linear backoff. Non-transient
    failures (and the final attempt) propagate to the caller.
    """
    query = {"format": "json", **params}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(API_URL, params=query, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)


# --- Public API -------------------------------------------------------------

def search_wikipedia(query: str):
    """Search Wikipedia.

    Returns a list of ``{"title": str, "snippet": str}`` dicts (max 5), with
    HTML stripped from the snippets. If there are no results, returns a
    guidance string instead of an empty list.
    """
    data = _api_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
        }
    )

    results = data.get("query", {}).get("search", [])
    if not results:
        return (
            f"No results for '{query}'. "
            f"Try broader or alternative search terms."
        )

    return [
        {
            "title": item.get("title", ""),
            "snippet": _strip_html(item.get("snippet", "")),
        }
        for item in results
    ]


def get_article(title: str) -> str:
    """Fetch the plain-text extract for an article title.

    Behaviour:
      * Follows redirects. If the requested title redirected elsewhere, the
        output is prefixed with ``[Redirected to: <resolved title>]``.
      * Disambiguation pages return a list of the linked article titles.
      * Non-existent pages return a "not found" message.
      * Extracts longer than ~16,000 chars are truncated, with a trailing
        ``[Article truncated]`` marker.
    """
    data = _api_get(
        {
            "action": "query",
            "prop": "extracts|pageprops|links",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            # Only namespace-0 (article) links, so disambiguation "Options"
            # don't include Help:/Category:/Wikipedia: noise.
            "plnamespace": 0,
            "pllimit": "max",
        }
    )

    query = data.get("query", {})
    pages = query.get("pages", {})
    if not pages:
        return (
            f"No article found for '{title}'. "
            f"Use search_wikipedia to find the correct title."
        )

    # A titles=<single title> query returns exactly one page.
    page = next(iter(pages.values()))

    if "missing" in page:
        return (
            f"No article found for '{title}'. "
            f"Use search_wikipedia to find the correct title."
        )

    resolved_title = page.get("title", title)

    # Disambiguation pages carry the "disambiguation" pageprop.
    if "disambiguation" in page.get("pageprops", {}):
        options = [ln.get("title", "") for ln in page.get("links", [])]
        options_str = ", ".join(o for o in options if o)
        return (
            f"'{title}' is a disambiguation page. "
            f"Options: {options_str}"
        )

    extract = page.get("extract", "") or ""

    # A real redirect happened only if the API reports one (normalization of
    # capitalization/spacing does NOT count as a redirect).
    prefix = ""
    if query.get("redirects"):
        prefix = f"[Redirected to: {resolved_title}]\n\n"

    truncated = False
    if len(extract) > MAX_EXTRACT_CHARS:
        extract = extract[:MAX_EXTRACT_CHARS].rstrip()
        truncated = True

    result = prefix + extract
    if truncated:
        result += "\n\n[Article truncated]"
    return result


# --- Tiny CLI for manual/live use ------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3 or sys.argv[1] not in {"search", "article"}:
        print("Usage:")
        print('  python wiki.py search  "your query"')
        print('  python wiki.py article "Article Title"')
        raise SystemExit(1)

    command, argument = sys.argv[1], " ".join(sys.argv[2:])

    if command == "search":
        hits = search_wikipedia(argument)
        if isinstance(hits, str):
            print(hits)
        else:
            for i, hit in enumerate(hits, 1):
                print(f"{i}. {hit['title']}")
                print(f"   {hit['snippet']}")
    else:  # article
        print(get_article(argument))
