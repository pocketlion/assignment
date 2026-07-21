"""agent.py — a small Wikipedia tool-use agent on the Anthropic SDK.

    answer_question(question, system_prompt, tools_impl=None) -> dict

CLI:  python agent.py "question" [--json]   |   python agent.py --demo
"""
from __future__ import annotations
import json, os, sys
import wiki

# --- Config -----------------------------------------------------------------
MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 6           # model turns before stopping with "max_iterations"
TEMPERATURE = 0
MAX_TOKENS = 1024

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant with access to Wikipedia tools "
    "(search_wikipedia, get_article) to help answer questions from the user."
)

# --- Tool descriptions (edit these freely while iterating) -------------------
SEARCH_WIKIPEDIA_DESCRIPTION = (
    "Search English Wikipedia and return up to 5 candidate articles as "
    "{title, snippet}. Use this first when you are unsure of the exact article "
    "title, need to disambiguate a name, or want to find related topics."
)
GET_ARTICLE_DESCRIPTION = (
    "Fetch the plain-text extract of a Wikipedia article by its exact title. "
    "Follows redirects; returns guidance if the title is a disambiguation page "
    "or does not exist. Use after search_wikipedia to read the best match."
)

TOOLS = [
    {"name": "search_wikipedia", "description": SEARCH_WIKIPEDIA_DESCRIPTION,
     "input_schema": {"type": "object", "required": ["query"],
                      "properties": {"query": {"type": "string", "description": "Search terms."}}}},
    {"name": "get_article", "description": GET_ARTICLE_DESCRIPTION,
     "input_schema": {"type": "object", "required": ["title"],
                      "properties": {"title": {"type": "string", "description": "Exact article title."}}}},
]
DEFAULT_TOOLS = {"search_wikipedia": wiki.search_wikipedia, "get_article": wiki.get_article}

# --- Client -----------------------------------------------------------------
def _load_dotenv(path=".env"):
    """Load KEY=VALUE lines from a .env file into os.environ.

    Existing environment variables are never overwritten (a real exported var
    wins); a missing file is a silent no-op. Supports blank lines, comments (#),
    an optional `export ` prefix, and single/double-quoted values.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line[7:] if line.startswith("export ") else line
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)

def _client():
    """Return an Anthropic client (reads ANTHROPIC_API_KEY, e.g. from .env).

    Imported lazily and patched in tests. The SDK retries transient API errors
    itself; Wikipedia calls are retried inside wiki.py.
    """
    import anthropic
    _load_dotenv()
    return anthropic.Anthropic()

# --- Core loop --------------------------------------------------------------
def answer_question(question, system_prompt, tools_impl=None):
    tools_impl = DEFAULT_TOOLS if tools_impl is None else tools_impl
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    result = {"question": question, "answer": "", "searched": False, "tool_calls": [],
              "num_iterations": 0, "stop_reason": "answered", "model": MODEL, "usage": usage}
    messages = [{"role": "user", "content": question}]
    try:
        client = _client()
        for i in range(MAX_ITERATIONS):
            result["num_iterations"] = i + 1
            resp = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                system=system_prompt, tools=TOOLS, messages=messages)
            u = getattr(resp, "usage", None)
            if u is not None:
                usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

            text = "".join(b.text for b in resp.content if b.type == "text")
            if text:
                result["answer"] = text
            if resp.stop_reason != "tool_use":
                result["stop_reason"] = "answered"
                return result

            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                result["searched"] = True
                raw = tools_impl[block.name](**block.input)   # tool errors -> outer except
                content = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                result["tool_calls"].append(
                    {"tool": block.name, "input": block.input, "result_retrieved": raw})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content})
            messages.append({"role": "user", "content": tool_results})

        result["stop_reason"] = "max_iterations"
        return result
    except Exception as e:
        result["answer"] = f"Error: {e}"
        result["stop_reason"] = "error"
        return result

# --- CLI --------------------------------------------------------------------
def _load_system_prompt(path="system_prompt.txt"):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or DEFAULT_SYSTEM_PROMPT
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT

_ICONS = {"search_wikipedia": "🔍", "get_article": "📄"}
def _trace(tc):
    arg = next(iter(tc["input"].values()), "")
    print(f'{_ICONS.get(tc["tool"], "🔧")} {tc["tool"]}: "{arg}"')

DEMO_QUESTIONS = [
    "In what year did Marie Curie win her first Nobel Prize?",           # simple factual
    "What nationality was the scientist that curium is named after?",    # multi-hop
    "What is 17 times 4?",                                               # no search needed
    "Tell me about Flibberjabberwock, the ancient lost city of Mars.",  # nonexistent topic
]

def main(argv):
    as_json = "--json" in argv
    args = [a for a in argv[1:] if a not in ("--json", "--demo")]
    system_prompt = _load_system_prompt()
    if "--demo" in argv:
        questions = DEMO_QUESTIONS
    elif args:
        questions = [args[0]]
    else:
        print('Usage: python agent.py "question" [--json]  |  python agent.py --demo')
        return
    for q in questions:
        res = answer_question(q, system_prompt)
        if as_json:
            print(json.dumps(res, indent=2, ensure_ascii=False)); continue
        print(f"Q: {q}")
        for tc in res["tool_calls"]:
            _trace(tc)
        print(f"A: {res['answer']}")
        print(f"   [{res['stop_reason']}, {res['num_iterations']} iter, "
              f"{res['usage']['total_tokens']} tokens]\n")

if __name__ == "__main__":
    main(sys.argv)
