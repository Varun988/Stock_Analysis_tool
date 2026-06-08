from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, desc, select

from app.db import SessionLocal, engine
from app.explanation_engine.schemas import RecommendationExplanationResponse

metadata = MetaData()

explanations_table = Table(
    "recommendation_explanations",
    metadata,
    Column("explanation_id", String, primary_key=True),
    Column("recommendation_id", String, nullable=False, index=True),
    Column("provider", String, nullable=True),
    Column("explanation", Text, nullable=False),
    Column("beginner_summary", Text, nullable=False),
    Column("risk_explanation", Text, nullable=False),
    Column("disclaimer", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    extend_existing=True,
)

# metadata.create_all(bind=engine)


def save_explanation(
    explanation: RecommendationExplanationResponse,
) -> RecommendationExplanationResponse:
    db = SessionLocal()

    created_at = explanation.created_at or datetime.now(timezone.utc)

    try:
        db.execute(
            explanations_table.insert().values(
                explanation_id=explanation.explanation_id,
                recommendation_id=explanation.recommendation_id,
                provider=explanation.provider,
                explanation=explanation.explanation,
                beginner_summary=explanation.beginner_summary,
                risk_explanation=explanation.risk_explanation,
                disclaimer=explanation.disclaimer,
                created_at=created_at,
            )
        )
        db.commit()

        return RecommendationExplanationResponse(
            explanation_id=explanation.explanation_id,
            recommendation_id=explanation.recommendation_id,
            provider=explanation.provider,
            explanation=explanation.explanation,
            beginner_summary=explanation.beginner_summary,
            risk_explanation=explanation.risk_explanation,
            disclaimer=explanation.disclaimer,
            created_at=created_at,
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _row_to_explanation(row) -> RecommendationExplanationResponse:
    return RecommendationExplanationResponse(
        explanation_id=row.explanation_id,
        recommendation_id=row.recommendation_id,
        provider=row.provider,
        explanation=row.explanation,
        beginner_summary=row.beginner_summary,
        risk_explanation=row.risk_explanation,
        disclaimer=row.disclaimer,
        created_at=row.created_at,
    )


def get_latest_explanation_from_db() -> RecommendationExplanationResponse | None:
    db = SessionLocal()

    try:
        statement = (
            select(explanations_table)
            .order_by(desc(explanations_table.c.created_at))
            .limit(1)
        )
        row = db.execute(statement).mappings().first()

        if row is None:
            return None

        return _row_to_explanation(row)
    finally:
        db.close()

def list_explanations_from_db(
    limit: int = 20,
) -> list[RecommendationExplanationResponse]:
    db = SessionLocal()

    try:
        statement = (
            select(explanations_table)
            .order_by(desc(explanations_table.c.created_at))
            .limit(limit)
        )

        rows = db.execute(statement).mappings().all()

        return [
            _row_to_explanation(row)
            for row in rows
        ]
    finally:
        db.close()