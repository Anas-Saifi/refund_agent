from fastapi import APIRouter

from refund_agent.agent import ALLOWED_REASONS, REFUND_WINDOW_DAYS
from refund_agent.api.schemas import PolicyResponse

router = APIRouter(prefix="/api/policy", tags=["Policy"])


@router.get(
    "",
    response_model=PolicyResponse,
    summary="Get refund policy rules",
    description="Returns the business rules and criteria enforced by the refund agent.",
)
def get_policy() -> PolicyResponse:
    return PolicyResponse(
        refund_window_days=REFUND_WINDOW_DAYS,
        allowed_reasons=ALLOWED_REASONS,
    )
