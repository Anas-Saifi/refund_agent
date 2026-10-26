from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from refund_agent.api.schemas import ChatRequest, ChatResponse
from refund_agent.api.services.agent_service import agent_service

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the Refund Agent",
    description="Processes a user message through the LangGraph refund agent workflow, preserving multi-turn state.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty.",
        )
    try:
        response = await agent_service.process_chat(
            message=request.message.strip(),
            thread_id=request.thread_id,
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(exc)}",
        )


@router.post(
    "/stream",
    summary="Stream messages and agent node transitions (Server-Sent Events)",
    description="Streams graph node execution events and the final assistant response as SSE (text/event-stream).",
)
async def chat_stream(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty.",
        )

    return StreamingResponse(
        agent_service.stream_chat(
            message=request.message.strip(),
            thread_id=request.thread_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
