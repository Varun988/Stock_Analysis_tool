from datetime import datetime

from sqlalchemy import Column, DateTime, Float, JSON, MetaData, String, Table, Text, desc, select

from app.db import SessionLocal, engine
from app.recommendation_engine.enums import (
    RecommendationAction,
    RecommendationReasonCode,
)
from app.recommendation_engine.schemas import (
    AllocationPlanItem,
    RecommendationResponse,
    RecommendationScoreBreakdown,
)
from app.research.schemas import ResearchContextResponse

metadata = MetaData()

recommendations_table = Table(
    "recommendations",
    metadata,
    Column("recommendation_id", String, primary_key=True),
    Column("recommendation_date", DateTime(timezone=True), nullable=False),
    Column("suggested_action", String, nullable=False),
    Column("suggested_amount", Float, nullable=True),
    Column("summary", Text, nullable=False),
    Column("reason_codes", JSON, nullable=False),
    Column("risk_note", Text, nullable=False),
    Column("disclaimer", Text, nullable=False),
    Column("allocation_plan", JSON, nullable=False),
    Column("score_breakdown", JSON, nullable=True),
    Column("research_context", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    extend_existing=True,
)

# metadata.create_all(bind=engine)


def save_recommendation(recommendation: RecommendationResponse) -> RecommendationResponse:
    db = SessionLocal()

    try:
        db.execute(
            recommendations_table.insert().values(
                recommendation_id=recommendation.recommendation_id,
                recommendation_date=recommendation.recommendation_date,
                suggested_action=recommendation.suggested_action.value,
                suggested_amount=recommendation.suggested_amount,
                summary=recommendation.summary,
                reason_codes=[
                    reason_code.value
                    for reason_code in recommendation.reason_codes
                ],
                risk_note=recommendation.risk_note,
                disclaimer=recommendation.disclaimer,
                allocation_plan=[
                    item.model_dump(mode="json")
                    for item in recommendation.allocation_plan
                ],
                score_breakdown=(
                    recommendation.score_breakdown.model_dump(mode="json")
                    if recommendation.score_breakdown is not None
                    else None
                ),
                research_context=(
                    recommendation.research_context.model_dump(mode="json")
                    if recommendation.research_context is not None
                    else None
                ),
                created_at=datetime.now(recommendation.recommendation_date.tzinfo),
            )
        )
        db.commit()
        return recommendation
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _row_to_recommendation(row) -> RecommendationResponse:
    score_breakdown = None

    if row.score_breakdown is not None:
        score_breakdown = RecommendationScoreBreakdown(**row.score_breakdown)

    research_context = None

    if row.get("research_context") is not None:
        research_context = ResearchContextResponse(**row.research_context)

    return RecommendationResponse(
        recommendation_id=row.recommendation_id,
        recommendation_date=row.recommendation_date,
        suggested_action=RecommendationAction(row.suggested_action),
        suggested_amount=row.suggested_amount,
        summary=row.summary,
        reason_codes=[
            RecommendationReasonCode(reason_code)
            for reason_code in row.reason_codes
        ],
        risk_note=row.risk_note,
        disclaimer=row.disclaimer,
        allocation_plan=[
            AllocationPlanItem(**item)
            for item in row.allocation_plan
        ],
        score_breakdown=score_breakdown,
        research_context=research_context,
    )


def get_latest_recommendation_from_db() -> RecommendationResponse | None:
    db = SessionLocal()

    try:
        statement = (
            select(recommendations_table)
            .order_by(desc(recommendations_table.c.recommendation_date))
            .limit(1)
        )
        row = db.execute(statement).mappings().first()

        if row is None:
            return None

        return _row_to_recommendation(row)
    finally:
        db.close()


def list_recommendations_from_db(
    limit: int = 20,
) -> list[RecommendationResponse]:
    db = SessionLocal()

    try:
        statement = (
            select(recommendations_table)
            .order_by(desc(recommendations_table.c.recommendation_date))
            .limit(limit)
        )

        rows = db.execute(statement).mappings().all()

        return [
            _row_to_recommendation(row)
            for row in rows
        ]
    finally:
        db.close()
