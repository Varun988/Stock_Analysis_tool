import json
from json import JSONDecodeError

from google import genai

from app.ai_engine.providers.base import AIExplanationProvider
from app.ai_engine.schemas import AIExplanationRequest, AIExplanationResponse
from app.config import settings


class GeminiAIExplanationProvider(AIExplanationProvider):
    """Gemini-powered AI explanation provider."""

    def _ensure_configured(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set GEMINI_API_KEY in the environment before using Gemini."
            )

    def _build_prompt(self, request: AIExplanationRequest) -> str:
        reason_codes_text = ", ".join(request.reason_codes)

        if request.allocation_plan:
            allocation_plan_text = "\n".join(
                [
                    (
                        f"- {item.instrument_type}: INR {item.amount}. "
                        f"Reason: {item.reason}"
                    )
                    for item in request.allocation_plan
                ]
            )
        else:
            allocation_plan_text = "No allocation plan was provided."

        if request.score_breakdown is not None:
            score_breakdown_text = (
                "Diversification score: "
                f"{request.score_breakdown.diversification_score}/100\n"
                "Risk suitability score: "
                f"{request.score_breakdown.risk_suitability_score}/100\n"
                "Preference match score: "
                f"{request.score_breakdown.preference_match_score}/100"
            )
        else:
            score_breakdown_text = "No score breakdown was provided."

        prompt_lines = [
            "You are an educational investment explanation assistant.",
            "",
            "Your job:",
            "- Explain the backend-generated recommendation in beginner-friendly language.",
            "- Explain the suggested allocation plan if one is provided.",
            "- Explain the score breakdown if one is provided.",
            "- Do not invent new prices, NAVs, returns, risk values, or financial facts.",
            "- Do not give direct buy/sell advice.",
            "- Do not override the backend recommendation.",
            "- Keep the tone clear, calm, and educational.",
            "- Make it clear this is not financial advice.",
            "",
            "Backend recommendation data:",
            f"Recommendation ID: {request.recommendation_id}",
            f"Suggested action: {request.suggested_action}",
            f"Suggested amount: INR {request.suggested_amount}",
            f"Summary: {request.summary}",
            f"Reason codes: {reason_codes_text}",
            f"Risk note: {request.risk_note}",
            f"Disclaimer: {request.disclaimer}",
            "",
            "Suggested allocation plan:",
            allocation_plan_text,
            "",
            "Score breakdown:",
            score_breakdown_text,
            "",
            "Return only valid JSON with exactly these keys:",
            "{",
            '  "beginner_summary": "short beginner-friendly summary",',
            '  "explanation": "detailed but simple explanation including the allocation plan and score breakdown",',
            '  "risk_explanation": "simple explanation of the risk note and risk suitability score"',
            "}",
        ]

        return "\n".join(prompt_lines)
        
    def _clean_json_text(self, text: str) -> str:
        cleaned_text = text.strip()

        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text.removeprefix("```json").strip()

        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.removeprefix("```").strip()

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text.removesuffix("```").strip()

        return cleaned_text

    def _parse_response_text(self, response_text: str) -> dict:
        cleaned_text = self._clean_json_text(response_text)

        try:
            parsed = json.loads(cleaned_text)
        except JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned invalid JSON: {response_text}"
            ) from exc

        required_keys = {
            "beginner_summary",
            "explanation",
            "risk_explanation",
        }

        missing_keys = required_keys - set(parsed.keys())

        if missing_keys:
            raise RuntimeError(
                f"Gemini response missing required keys: {sorted(missing_keys)}"
            )

        return parsed

    def generate_explanation(
        self,
        request: AIExplanationRequest,
    ) -> AIExplanationResponse:
        self._ensure_configured()

        client = genai.Client(api_key=settings.gemini_api_key)
        prompt = self._build_prompt(request)

        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Gemini explanation generation failed: {exc}"
            ) from exc
        finally:
            client.close()

        response_text = getattr(response, "text", None)

        if not response_text:
            raise RuntimeError("Gemini returned an empty response.")

        parsed_response = self._parse_response_text(response_text)

        return AIExplanationResponse(
            provider="GEMINI",
            explanation=parsed_response["explanation"],
            beginner_summary=parsed_response["beginner_summary"],
            risk_explanation=parsed_response["risk_explanation"],
        )
