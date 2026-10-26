from typing import Literal
from pydantic import BaseModel, Field


class InputResponse(BaseModel):
    username: str | None = Field(
        default=None,
        description="Username explicitly provided by the user, else null",
    )


class ValidateDate(BaseModel):
    refund_applicable: bool = Field(
        description="True if the purchase date is within the refund window"
    )


class RequestedItem(BaseModel):
    item: str = Field(description="Item name as the user said it")
    quantity: int | None = Field(
        default=None,
        description="Quantity only if the user explicitly stated one, else null",
    )


class RefundRequest(BaseModel):
    items: list[RequestedItem] = Field(default_factory=list)


class RefundReason(BaseModel):
    category: Literal["expired", "damaged", "wrong_product", "other", "not_stated"] = Field(
        description="Category of the reason the user gave for the refund"
    )
    reason: str | None = Field(
        default=None, description="Short paraphrase of the user's stated reason"
    )