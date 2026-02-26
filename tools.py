"""
Tools for Personal Chat: semantic search, weather, web search, GitHub, LinkedIn.
Used by the LLM via OpenAI function/tool calling. Resources (model, index, chunks)
are injected via init_resources() from app.py.
"""

import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

# Injected by init_resources(); used by semantic_search_personal
_model = None
_index = None
_chunks = None


def init_resources(model: Any, index: Any, chunks: List[str]) -> None:
    """Set the FAISS model, index, and chunks for semantic_search_personal."""
    global _model, _index, _chunks
    _model = model
    _index = index
    _chunks = chunks


# --- Tool implementations ---


def semantic_search_personal(query: str, k: int = 4) -> str:
    """
    Search Roxy's resume and personal data for relevant information.
    Call this with different queries when you need to look up her background, skills, or experience.
    """
    global _model, _index, _chunks
    if _index is None or _chunks is None or _model is None:
        return "Semantic search is not available (resources not loaded)."
    try:
        k = max(1, min(k, len(_chunks)))
        query_embedding = _model.encode([query]).astype("float32")
        distances, indices = _index.search(query_embedding, k)
        relevant = [_chunks[i] for i in indices[0]]
        return "\n\n---\n\n".join(relevant)
    except Exception as e:
        return f"Search failed: {str(e)}"


def get_weather(location: str) -> str:
    """
    Get current weather and today's forecast for a city or place.
    Use for small talk about weather (e.g. "What's the weather in NYC?").
    """
    try:
        import urllib.request
        import urllib.parse

        # Geocode location via Open-Meteo (no API key)
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": location.strip(), "count": 1, "language": "en", "format": "json"}
        req = urllib.request.Request(
            f"{geocode_url}?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": "PersonalRAGChatbot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        if not data.get("results"):
            return f"Could not find location: {location}. Try a city name (e.g. New York, London)."
        r = data["results"][0]
        lat, lon = r["latitude"], r["longitude"]
        name = r.get("name", location)

        # Forecast for today
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": r.get("timezone", "auto"),
            "forecast_days": 1,
        }
        req = urllib.request.Request(
            f"{forecast_url}?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": "PersonalRAGChatbot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        cur = data.get("current", {})
        daily = data.get("daily", {})
        temp = cur.get("temperature_2m")
        code = cur.get("weather_code", 0)
        # WMO codes: 0=clear, 1-3=mainly clear/cloudy, 45/48=fog, 51-67=rain, 71-77=snow, 80-82=showers, 95-99=thunderstorm
        if code == 0:
            cond = "clear"
        elif code in (1, 2, 3):
            cond = "mainly clear to partly cloudy"
        elif code in (45, 48):
            cond = "foggy"
        elif 51 <= code <= 67:
            cond = "rainy"
        elif 71 <= code <= 77:
            cond = "snowy"
        elif 80 <= code <= 82:
            cond = "rain showers"
        elif 95 <= code <= 99:
            cond = "thunderstorms"
        else:
            cond = "variable"
        t_max = daily.get("temperature_2m_max", [None])[0]
        t_min = daily.get("temperature_2m_min", [None])[0]
        prec = daily.get("precipitation_sum", [0])[0] or 0
        parts = [
            f"Weather in {name}: {temp}°C ({cond}).",
            f"Today: high {t_max}°C, low {t_min}°C." if t_max is not None else "",
            f"Precipitation: {prec} mm." if prec else "",
        ]
        return " ".join(p for p in parts if p).strip()
    except Exception as e:
        return f"Weather lookup failed: {str(e)}. Try a city name like 'New York' or 'London'."


def web_search(query: str) -> str:
    """
    Search the web for current information. Use for general lookup, news, or when
    the user asks about something outside your resume (e.g. weather via search, recent events).
    """
    # Prefer Tavily if key is set and package installed
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            result = client.search(query, max_results=5, search_depth="basic")
            if not result.get("results"):
                return "No results found."
            parts = []
            for r in result["results"][:5]:
                title = r.get("title", "")
                content = r.get("content", "")[:500]
                url = r.get("url", "")
                parts.append(f"[{title}]({url})\n{content}")
            return "\n\n".join(parts)
        except ImportError:
            pass  # fall through to DuckDuckGo
        except Exception as e:
            return f"Tavily search failed: {str(e)}. You can answer from general knowledge."
    # Fallback: DuckDuckGo (no API key)
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        parts = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")[:400]
            href = r.get("href", "")
            parts.append(f"{title}\n{body}\nSource: {href}")
        return "\n\n".join(parts)
    except ImportError:
        return "Web search is unavailable (install duckduckgo-search or set TAVILY_API_KEY)."
    except Exception as e:
        return f"Web search failed: {str(e)}."


def github_search(query: str) -> str:
    """
    Search GitHub for repositories across all of GitHub. To find a specific person's
    projects, include "user:USERNAME" in the query (e.g. "user:roxystory" or "user:manishbhaiii python").
    """
    token = os.getenv("GITHUB_TOKEN")
    try:
        import urllib.request
        import urllib.parse
        q = urllib.parse.quote(query.strip())
        url = f"https://api.github.com/search/repositories?q={q}&per_page=5&sort=relevance"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        req.add_header("User-Agent", "PersonalRAGChatbot/1.0")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("items", [])
        if not items:
            return "No GitHub repositories found for that query."
        parts = []
        for r in items[:5]:
            full_name = r.get("full_name", "")
            owner = r.get("owner", {}).get("login", "")
            desc = r.get("description") or "No description"
            stars = r.get("stargazers_count", 0)
            url = r.get("html_url", "")
            parts.append(f"{full_name} (owner: {owner}) ({stars} stars)\n{desc}\n{url}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"GitHub search failed: {str(e)}. (Optional: set GITHUB_TOKEN for higher rate limits.)"


# --- OpenAI tool definitions and registry ---

def _make_tool(
    name: str,
    description: str,
    params: Dict[str, Any],
    handler: Callable[..., str],
    required: Optional[List[str]] = None,
) -> Tuple[Dict, Callable]:
    req = required if required is not None else list(params.keys())
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": params, "required": req},
        },
    }
    return schema, handler


def _semantic_search_handler(query: str, k: int = 4) -> str:
    return semantic_search_personal(query, k)


TOOL_DEFINITIONS: List[Tuple[Dict, Callable]] = [
    _make_tool(
        "semantic_search_personal",
        "Search Roxy's resume and personal data for relevant facts. Call with different queries to look up her background, skills, experience, or preferences. Use this when answering questions about her.",
        {"query": {"type": "string", "description": "Natural language search query"}, "k": {"type": "integer", "description": "Number of chunks to return (default 4)"}},
        _semantic_search_handler,
        required=["query"],
    ),
    _make_tool(
        "get_weather",
        "Get current weather and today's forecast for a city. Use for small talk about weather.",
        {"location": {"type": "string", "description": "City or place name (e.g. New York, London)"}},
        get_weather,
    ),
    _make_tool(
        "web_search",
        "Search the web for current information. Use when the user asks about something outside your resume (e.g. news, weather by search, general facts).",
        {"query": {"type": "string", "description": "Search query"}},
        web_search,
    ),
    _make_tool(
        "github_search",
        "Search all of GitHub for repositories. To find a specific person's projects use 'user:USERNAME' in the query (e.g. user:roxystory or user:someone repo-name). Use for Roxy's GitHub or anyone else's when the recruiter provides an account name.",
        {"query": {"type": "string", "description": "Search query; include user:USERNAME to limit to that account (e.g. user:roxystory)"}},
        github_search,
    ),
]


def get_openai_tools() -> List[Dict]:
    """Return list of OpenAI tool schemas (for chat.completions.create tools=)."""
    return [schema for schema, _ in TOOL_DEFINITIONS]


def get_tool_handlers() -> Dict[str, Callable]:
    """Return name -> handler for executing tool calls."""
    return {td[0]["function"]["name"]: td[1] for td in TOOL_DEFINITIONS}


def run_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool by name with given arguments. Returns string result."""
    handlers = get_tool_handlers()
    if name not in handlers:
        return f"Unknown tool: {name}"
    fn = handlers[name]
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"Tool error (wrong arguments): {str(e)}"
