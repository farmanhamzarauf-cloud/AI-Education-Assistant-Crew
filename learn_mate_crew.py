import os
import inspect
import litellm  # import BEFORE crewai so our patch is in place first

# ---------------------------------------------------------------------------
# Groq fix: CrewAI adds a "cache_breakpoint" field to messages (used for prompt
# caching on other providers). Groq rejects it, so we remove it before sending.
# ---------------------------------------------------------------------------
_UNSUPPORTED_KEYS = {"cache_breakpoint", "cache_control"}


def _clean(obj):
    """Recursively remove unsupported keys from messages."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in _UNSUPPORTED_KEYS}
    if isinstance(obj, list):
        return [_clean(item) for item in obj]
    return obj


def _clean_call(args, kwargs):
    args = list(args)
    if len(args) > 1:                       # litellm.completion(model, messages, ...)
        args[1] = _clean(args[1])
    if kwargs.get("messages"):
        kwargs["messages"] = _clean(kwargs["messages"])
    return tuple(args), kwargs


def _wrap(fn):
    if inspect.iscoroutinefunction(fn):
        async def async_wrapper(*args, **kwargs):
            args, kwargs = _clean_call(args, kwargs)
            return await fn(*args, **kwargs)
        async_wrapper._cleaned = True
        return async_wrapper

    def wrapper(*args, **kwargs):
        args, kwargs = _clean_call(args, kwargs)
        return fn(*args, **kwargs)
    wrapper._cleaned = True
    return wrapper


for _name in ("completion", "acompletion"):
    _fn = getattr(litellm, _name, None)
    if _fn is not None and not getattr(_fn, "_cleaned", False):
        setattr(litellm, _name, _wrap(_fn))
# ---------------------------------------------------------------------------

from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from ddgs import DDGS

# The "groq/" prefix tells CrewAI to use Groq. The rest is the model name.
GROQ_MODEL = "groq/openai/gpt-oss-120b"


@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web with DuckDuckGo. Returns titles, links and snippets."""
    try:
        results = DDGS().text(query, max_results=6)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(
            f"Title: {r.get('title')}\n"
            f"URL: {r.get('href')}\n"
            f"Snippet: {r.get('body')}\n"
        )
    return "\n".join(lines)


def run_research(topic: str, api_key: str) -> str:
    """Runs the single-agent crew and returns the report as markdown text."""
    os.environ["GROQ_API_KEY"] = api_key

    llm = LLM(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.3,
    )

    researcher = Agent(
        role="Learn Mate Research Assistant",
        goal=f"Research '{topic}' and write a clear, accurate report for students.",
        backstory=(
            "You are Learn Mate, a friendly education assistant. You search the web, "
            "compare several sources, and explain topics in simple, organized language."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        allow_delegation=False,
        max_iter=6,
        verbose=True,
    )

    task = Task(
        description=(
            f"Research the topic: {topic}\n"
            "Use the search tool 2 to 4 times with different queries, "
            "then write a report for students based on what you found."
        ),
        expected_output=(
            "A markdown report with these sections: Title, Introduction, Key Concepts, "
            "Important Facts and Recent Developments, Real-World Examples, Conclusion, "
            "and a Sources list with URLs."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return result.raw
