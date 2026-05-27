import json
from json import JSONDecodeError

from google import genai

from app.config import settings
from app.research.schemas import RawResearchResult


DEFAULT_RESEARCH_RISK_NOTE = (
    "Research context is for education and decision support only. It is not "
    "financial advice, does not guarantee returns, and should not be used as a "
    "direct buy/sell signal."
)


def _clean_json_text(text: str) -> str:
    cleaned_text = text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.removeprefix("```json").strip()

    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.removeprefix("```").strip()

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text.removesuffix("```").strip()

    return cleaned_text


def _build_sources_text(results: list[RawResearchResult]) -> str:
    if not results:
        return "No search results were provided."

    lines: list[str] = []

    for result in results:
        lines.append(
            "\n".join(
                [
                    f"Position: {result.position}",
                    f"Title: {result.title}",
                    f"Source: {result.source or result.displayed_link or 'Unknown'}",
                    f"Date: {result.date or 'Unknown'}",
                    f"URL: {result.link or 'Not available'}",
                    f"Snippet: {result.snippet or 'No snippet'}",
                ]
            )
        )

    return "\n\n---\n\n".join(lines)


def _build_research_summary_prompt(
    query: str,
    subject_type: str,
    results: list[RawResearchResult],
) -> str:
    sources_text = _build_sources_text(results)

    prompt_lines = [
        "You are a cautious investment research summarization assistant.",
        "",
        "Your task:",
        "- Summarize only the supplied search results/snippets.",
        "- Do not invent facts, prices, NAVs, returns, ratings, dates, or events.",
        "- Do not provide direct buy/sell advice.",
        "- Do not predict short-term market movement.",
        "- Explain the context in beginner-friendly language.",
        "- Mention uncertainty when results are generic, limited, or mixed.",
        "- Keep the output educational and calm.",
        "",
        f"Research query: {query}",
        f"Subject type: {subject_type}",
        "",
        "Search results:",
        sources_text,
        "",
        "Return only valid JSON with exactly these keys:",
        "{",
        '  "summary": "short beginner-friendly research summary",',
        '  "key_points": ["point 1", "point 2", "point 3"],',
        '  "risk_note": "clear note that research is informational only and not a buy/sell signal"',
        "}",
    ]

    return "\n".join(prompt_lines)


def summarize_research_with_gemini(
    query: str,
    subject_type: str,
    results: list[RawResearchResult],
) -> dict:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "Gemini API key is not configured. Set GEMINI_API_KEY before using research summarization."
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = _build_research_summary_prompt(query, subject_type, results)

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini research summarization failed: {exc}") from exc
    finally:
        client.close()

    response_text = getattr(response, "text", None)

    if not response_text:
        raise RuntimeError("Gemini returned an empty research summary response.")

    cleaned_text = _clean_json_text(response_text)

    try:
        parsed = json.loads(cleaned_text)
    except JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid research JSON: {response_text}") from exc

    required_keys = {"summary", "key_points", "risk_note"}
    missing_keys = required_keys - set(parsed.keys())

    if missing_keys:
        raise RuntimeError(f"Gemini research summary missing keys: {sorted(missing_keys)}")

    if not isinstance(parsed["key_points"], list):
        raise RuntimeError("Gemini research key_points must be a list.")

    return parsed


def build_rule_based_research_summary(
    query: str,
    subject_type: str,
    results: list[RawResearchResult],
) -> dict:
    if not results:
        return {
            "summary": (
                "No research results were available for this query. Treat this as "
                "insufficient research context rather than a market signal."
            ),
            "key_points": [
                "No external research results were returned.",
                "Use portfolio allocation, risk profile, and market data as the primary decision inputs.",
                "Try refining the research query if needed.",
            ],
            "risk_note": DEFAULT_RESEARCH_RISK_NOTE,
        }

    top_titles = [result.title for result in results[:3]]

    return {
        "summary": (
            f"Research results were found for '{query}'. These results provide "
            "background context only and should not override portfolio/risk-based recommendations."
        ),
        "key_points": [
            f"Top result: {top_titles[0]}",
            "Review source links directly before relying on any article or headline.",
            "Use research context as supplementary information, not as a direct buy/sell signal.",
        ],
        "risk_note": DEFAULT_RESEARCH_RISK_NOTE,
    }
