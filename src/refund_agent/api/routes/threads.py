from fastapi import APIRouter, status

from refund_agent.api.schemas import (
    AgentStateSummary,
    DeleteThreadResponse,
    ThreadHistoryResponse,
    ThreadListResponse,
)
from refund_agent.api.services.agent_service import agent_service

router = APIRouter(prefix="/api/threads", tags=["Threads"])


@router.get(
    "",
    response_model=ThreadListResponse,
    summary="List all active conversation threads",
    description="Returns summaries for all currently tracked conversation threads, sorted by recent activity.",
)
def list_threads() -> ThreadListResponse:
    threads = agent_service.list_threads()
    return ThreadListResponse(total=len(threads), threads=threads)


@router.get(
    "/{thread_id}/state",
    response_model=AgentStateSummary,
    summary="Get current state of a thread",
    description="Inspect the latest state of a refund thread including username, order details, and refund status.",
)
def get_thread_state(thread_id: str) -> AgentStateSummary:
    return agent_service.get_thread_state(thread_id)


@router.get(
    "/{thread_id}/history",
    response_model=ThreadHistoryResponse,
    summary="Get conversation history of a thread",
    description="Retrieve the chronological message log and state snapshot for a specific thread.",
)
def get_thread_history(thread_id: str) -> ThreadHistoryResponse:
    return agent_service.get_thread_history(thread_id)


@router.delete(
    "/{thread_id}",
    response_model=DeleteThreadResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete/reset a conversation thread",
    description="Deletes checkpoints and session data for the specified thread ID.",
)
def delete_thread(thread_id: str) -> DeleteThreadResponse:
    deleted = agent_service.delete_thread(thread_id)
    return DeleteThreadResponse(
        thread_id=thread_id,
        deleted=deleted,
        message=f"Thread '{thread_id}' successfully cleared.",
    )
