from abc import ABC, abstractmethod

from app.ai_engine.schemas import AIExplanationRequest, AIExplanationResponse


class AIExplanationProvider(ABC):
    """Base class for AI explanation providers."""

    @abstractmethod
    def generate_explanation(
        self,
        request: AIExplanationRequest,
    ) -> AIExplanationResponse:
        """Generate a beginner-friendly explanation."""
        raise NotImplementedError