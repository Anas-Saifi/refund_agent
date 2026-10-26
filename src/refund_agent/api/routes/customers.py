from fastapi import APIRouter, HTTPException, status

from refund_agent.api.schemas import CustomerDetail, CustomerSummary
from refund_agent.api.services.customer_service import customer_service

router = APIRouter(prefix="/api/customers", tags=["Customers"])


@router.get(
    "",
    response_model=list[CustomerSummary],
    summary="List customers and orders",
    description="Retrieve all customers and their purchase summary from the grocery store database.",
)
def list_customers() -> list[CustomerSummary]:
    return customer_service.list_customers()


@router.get(
    "/{username}",
    response_model=CustomerDetail,
    summary="Get customer details by username",
    description="Retrieve order history and item details for a specific customer username.",
)
def get_customer(username: str) -> CustomerDetail:
    customer = customer_service.get_customer(username)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with username '{username}' was not found.",
        )
    return customer
