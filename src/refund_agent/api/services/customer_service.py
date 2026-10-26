import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from refund_agent.agent import REFUND_WINDOW_DAYS
from refund_agent.api.schemas import CustomerDetail, CustomerSummary

DATA_FILE = Path(__file__).resolve().parents[2] / "data.json"


class CustomerService:
    def __init__(self, data_path: Path = DATA_FILE):
        self.data_path = data_path
        self._cache: dict[str, Any] | None = None

    def _load_data(self) -> dict[str, Any]:
        if self._cache is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Customer data file not found at {self.data_path}")
            with open(self.data_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        return self._cache

    def is_eligible_date(self, date_str: str) -> bool:
        """Check if purchase date is within the 7-day refund window."""
        try:
            purchase_dt = datetime.fromisoformat(date_str.strip())
            now = datetime.now()
            return (now - purchase_dt) <= timedelta(days=REFUND_WINDOW_DAYS)
        except Exception:
            return False

    def list_customers(self) -> list[CustomerSummary]:
        data = self._load_data()
        customers = []
        for cust in data.get("customers", []):
            eligible = self.is_eligible_date(cust["date_of_purchase"])
            customers.append(
                CustomerSummary(
                    name=cust["name"],
                    username=cust["username"],
                    order_id=cust["order_id"],
                    date_of_purchase=cust["date_of_purchase"],
                    total_amount=float(cust["total_amount"]),
                    item_count=len(cust.get("items", [])),
                    is_eligible_for_refund=eligible,
                )
            )
        return customers

    def get_customer(self, username: str) -> CustomerDetail | None:
        data = self._load_data()
        normalized = username.strip().lower()
        for cust in data.get("customers", []):
            if cust["username"].strip().lower() == normalized:
                eligible = self.is_eligible_date(cust["date_of_purchase"])
                return CustomerDetail(
                    name=cust["name"],
                    username=cust["username"],
                    order_id=cust["order_id"],
                    date_of_purchase=cust["date_of_purchase"],
                    items=cust.get("items", []),
                    total_amount=float(cust["total_amount"]),
                    is_eligible_for_refund=eligible,
                )
        return None


customer_service = CustomerService()
