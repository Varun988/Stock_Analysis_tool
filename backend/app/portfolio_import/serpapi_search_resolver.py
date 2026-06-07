import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.config import settings


MAX_SEARCH_RESULTS_PER_QUERY = 8
MAX_QUERIES_PER_HOLDING = 4


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def build_instrument_search_queries(holding: dict[str, Any]) -> list[str]:
    """Build targeted search queries for resolving an uploaded holding.

    This intentionally avoids hardcoded ISIN-to-symbol mappings. The goal is to
    gather enough web-search context for Gemini to convert into structured
    instrument identity JSON.
    """
    isin = _clean_text(holding.get("isin"))
    name = _clean_text(holding.get("instrument_name"))
    instrument_type = _clean_text(holding.get("instrument_type"))

    queries: list[str] = []

    if isin:
        queries.extend(
            [
                f"{isin} NSE symbol ETF mutual fund",
                f"{isin} Yahoo Finance NSE",
                f"{isin} AMFI scheme code",
                f"{isin} instrument name benchmark",
            ]
        )

    if name:
        queries.extend(
            [
                f"{name} NSE symbol",
                f"{name} Yahoo Finance",
                f"{name} AMFI scheme code benchmark",
            ]
        )

    if isin and name:
        queries.insert(0, f"{isin} {name}")

    if instrument_type and name:
        queries.append(f"{name} {instrument_type} benchmark")

    seen = set()
    unique_queries = []
    for query in queries:
        normalized_query = " ".join(query.split())
        if normalized_query and normalized_query not in seen:
            seen.add(normalized_query)
            unique_queries.append(normalized_query)

    return unique_queries[:MAX_QUERIES_PER_HOLDING]


def fetch_serpapi_google_results(query: str) -> dict[str, Any]:
    """Fetch Google results through SerpAPI using the backend settings."""
    if not settings.serpapi_api_key:
        raise RuntimeError(
            "SerpAPI API key is not configured. Set SERPAPI_API_KEY in backend .env."
        )

    params = {
        "engine": "google",
        "q": query,
        "api_key": settings.serpapi_api_key,
        "hl": settings.research_language,
        "gl": settings.research_country,
        "num": MAX_SEARCH_RESULTS_PER_QUERY,
    }

    url = f"{settings.serpapi_base_url}?{urlencode(params)}"

    request = Request(
        url,
        headers={
            "accept": "application/json",
            "User-Agent": "StockAnalysisTool/0.1",
        },
    )

    try:
        with urlopen(request, timeout=settings.research_request_timeout_seconds) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return json.loads(response_body)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SerpAPI HTTP error {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"SerpAPI connection error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("SerpAPI returned invalid JSON") from exc


def extract_search_result_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only compact, safe, useful fields from SerpAPI results."""
    items: list[dict[str, Any]] = []

    organic_results = payload.get("organic_results", [])
    if isinstance(organic_results, list):
        for result in organic_results[:MAX_SEARCH_RESULTS_PER_QUERY]:
            if not isinstance(result, dict):
                continue

            item = {
                "title": _clean_text(result.get("title")),
                "snippet": _clean_text(result.get("snippet")),
                "link": _clean_text(result.get("link")),
                "source": _clean_text(result.get("source")),
                "displayed_link": _clean_text(result.get("displayed_link")),
            }
            if item["title"] or item["snippet"] or item["link"]:
                items.append(item)

    knowledge_graph = payload.get("knowledge_graph")
    if isinstance(knowledge_graph, dict):
        item = {
            "title": _clean_text(knowledge_graph.get("title")),
            "snippet": _clean_text(knowledge_graph.get("description")),
            "link": _clean_text(knowledge_graph.get("source", {}).get("link"))
            if isinstance(knowledge_graph.get("source"), dict)
            else "",
            "source": "knowledge_graph",
            "displayed_link": "",
        }
        if item["title"] or item["snippet"]:
            items.append(item)

    answer_box = payload.get("answer_box")
    if isinstance(answer_box, dict):
        item = {
            "title": _clean_text(answer_box.get("title")),
            "snippet": _clean_text(answer_box.get("snippet") or answer_box.get("answer")),
            "link": _clean_text(answer_box.get("link")),
            "source": "answer_box",
            "displayed_link": "",
        }
        if item["title"] or item["snippet"]:
            items.append(item)

    return items


def search_instrument_context_with_serpapi(holding: dict[str, Any]) -> dict[str, Any]:
    """Return compact search context for Gemini instrument identity extraction."""
    queries = build_instrument_search_queries(holding)
    search_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for query in queries:
        try:
            payload = fetch_serpapi_google_results(query)
            items = extract_search_result_items(payload)
            for item in items:
                search_results.append({"query": query, **item})
        except Exception as exc:
            errors.append(f"{query}: {exc}")

    # Deduplicate by link + title while preserving order.
    deduped_results: list[dict[str, Any]] = []
    seen = set()
    for result in search_results:
        key = (result.get("link"), result.get("title"))
        if key in seen:
            continue
        seen.add(key)
        deduped_results.append(result)

    return {
        "queries_used": queries,
        "search_results": deduped_results[:20],
        "errors": errors,
    }
