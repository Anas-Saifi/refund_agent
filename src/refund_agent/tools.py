import json
from pathlib import Path
from langchain_core.tools import tool
from datetime import datetime

DATA_FILE = Path(__file__).parent / "data.json"


@tool
def get_customer_data(username: str) -> dict:
    """Find a customer by username and return their profile and order history."""

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    for customer in data["customers"]:
        if customer["username"].lower() == username.strip().lower():
            return customer

    return {"error": "Customer not found."}

@tool
def get_current_time():
    """Find out the current date and time"""

    current_date_time = datetime.now()
    return current_date_time


@tool
def calculate_refund(*items):
    """Calculates the total refund amount for one or more items. Each item must include the following fields:

    - **name** (`str`): The name of the item.
    - **price** (`float`): The total price paid for the item's original quantity, not the unit price.
    - **quantity** (`int`): The original quantity purchased. Must be greater than zero.
    - **refund_quantity** (`int`, optional): The number of units for which a refund is requested. If provided, it must be greater than zero and cannot exceed the original quantity. If omitted, the entire original quantity is considered for the refund.

    For each item, the tool calculates the unit price by dividing `price` by `quantity`, then multiplies it by `refund_quantity` if specified, or by the original `quantity` otherwise. It returns the sum of the refund amounts for all items. If no items are provided, it returns `0`.

    Use this tool to calculate the total refund amount before processing a refund."""
    total_refund = 0

    for item in items:
        price = item["price"]
        quantity = item["quantity"]
        refund_quantity = item.get("refund_quantity")

        # Calculate the price of one item
        price_per_item = price / quantity

        # Use the requested refund quantity if provided;
        # otherwise, use the total quantity
        quantity_to_refund = (
            refund_quantity
            if refund_quantity is not None
            else quantity
        )

        total_refund += price_per_item * quantity_to_refund

    return total_refund