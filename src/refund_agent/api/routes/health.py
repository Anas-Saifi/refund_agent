import os
from datetime import datetime, timezone
from fastapi import APIRouter

from refund_agent.api.config import settings
from refund_agent.api.schemas import HealthResponse
from refund_agent.api.services.agent_service import agent_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the server health and agent readiness.",
)
@router.get(
    "/api/health",
    response_model=HealthResponse,
    include_in_schema=False,
)
def health_check() -> HealthResponse:
    has_key = bool(os.environ.get("HIVE_API_KEY", "").strip())
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        hive_api_key_configured=has_key,
        active_threads=agent_service.active_thread_count(),
        server_time=datetime.now(timezone.utc).isoformat(),
    )
