from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="The user's message to the refund agent.",
        examples=["My username is alextaylor1001 and I want a refund for my eggs"],
    )
    thread_id: str | None = Field(
        default=None,
        description="Optional conversation thread ID. If not provided, a new thread ID is automatically generated.",
        examples=["b3c58850-9c2b-4d43-a20d-7ce61b24479f"],
    )


class AgentStateSummary(BaseModel):
    username: str | None = Field(default=None, description="Extracted username")
    user_exists: bool | None = Field(default=None, description="Whether the user exists in database")
    date_of_purchase: str | None = Field(default=None, description="Date of purchase (ISO format)")
    items: list[dict[str, Any]] | None = Field(default=None, description="Original purchased order items")
    total_amount: float | str | None = Field(default=None, description="Total amount paid for original order")
    refund_applicable: bool | None = Field(default=None, description="Whether purchase is within 7-day refund window")
    items_for_refund: dict[str, Any] | None = Field(default=None, description="Validated items requested for refund")
    to_refund: float | None = Field(default=None, description="Calculated total refund amount")
    refund_reason: str | None = Field(default=None, description="Category of the refund reason")
    refund_approved: bool | None = Field(default=None, description="Final decision: True if approved, False if denied")
    is_final: bool = Field(default=False, description="True if final decision has already been reached")
    stage: str = Field(
        default="initial",
        description="Current conversational stage: need_username, order_lookup, item_selection, reason_validation, completed",
    )


class ChatResponse(BaseModel):
    thread_id: str = Field(..., description="The conversation thread ID")
    reply: str = Field(..., description="Agent response message")
    state: AgentStateSummary = Field(..., description="Current agent state summary")
    is_final: bool = Field(default=False, description="True if the refund request is fully approved or denied")


class MessageRecord(BaseModel):
    role: Literal["user", "assistant", "system", "tool"] = Field(..., description="Message role")
    content: str = Field(..., description="Message text content")
    timestamp: str | None = Field(default=None, description="ISO timestamp")


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    message_count: int
    messages: list[MessageRecord]
    state: AgentStateSummary


class ThreadSummary(BaseModel):
    thread_id: str
    created_at: str
    last_active_at: str
    turn_count: int
    last_reply: str | None = None
    username: str | None = None
    refund_approved: bool | None = None
    is_final: bool = False


class ThreadListResponse(BaseModel):
    total: int
    threads: list[ThreadSummary]


class DeleteThreadResponse(BaseModel):
    thread_id: str
    deleted: bool
    message: str


class CustomerItem(BaseModel):
    item: str
    quantity: int
    price: float


class CustomerDetail(BaseModel):
    name: str
    order_id: int
    date_of_purchase: str
    items: list[CustomerItem]
    total_amount: float
    username: str
    is_eligible_for_refund: bool


class CustomerSummary(BaseModel):
    name: str
    username: str
    order_id: int
    date_of_purchase: str
    total_amount: float
    item_count: int
    is_eligible_for_refund: bool


class PolicyResponse(BaseModel):
    refund_window_days: int = 7
    allowed_reasons: dict[str, str] = Field(
        default_factory=lambda: {
            "expired": "the product was expired / past its expiry date",
            "damaged": "the product was damaged, broken, spoiled or leaking",
            "wrong_product": "the wrong product was delivered",
        }
    )
    ineligible_reasons: list[str] = Field(
        default_factory=lambda: [
            "change of mind / no longer needed",
            "found cheaper elsewhere",
            "ordered by mistake",
            "did not like it",
        ]
    )
    policy_summary: str = (
        "Refunds are applicable only for orders placed within the last 7 days. "
        "Only items with reasons of expiration, damage/spoilage, or wrong delivery are eligible for refund. "
        "Any other reasons are strictly denied. Once a decision is finalized, it cannot be appealed."
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    hive_api_key_configured: bool
    active_threads: int
    server_time: str
