import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from refund_agent.api.main import app
from refund_agent.api.services.agent_service import agent_service
from langchain_core.messages import AIMessage

client = TestClient(app)


def test_root_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert "Grocery Refund Assistant" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_api_info():
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Refund Agent API"
    assert "endpoints" in data
    assert data["endpoints"]["chat"] == "POST /api/chat"


def test_eval_report_pdf():
    response = client.get("/api/eval-report")
    assert response.status_code == 200
    assert "application/pdf" in response.headers.get("content-type", "")
    assert len(response.content) > 1000



def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "active_threads" in data


def test_policy():
    response = client.get("/api/policy")
    assert response.status_code == 200
    data = response.json()
    assert data["refund_window_days"] == 7
    assert "expired" in data["allowed_reasons"]
    assert "damaged" in data["allowed_reasons"]
    assert "wrong_product" in data["allowed_reasons"]


def test_customers_list():
    response = client.get("/api/customers")
    assert response.status_code == 200
    customers = response.json()
    assert isinstance(customers, list)
    assert len(customers) > 0
    first = customers[0]
    assert "username" in first
    assert "order_id" in first
    assert "is_eligible_for_refund" in first


def test_customer_detail_found():
    response = client.get("/api/customers/alextaylor1001")
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "alextaylor1001"
    assert "items" in data
    assert len(data["items"]) > 0


def test_customer_detail_not_found():
    response = client.get("/api/customers/non_existent_user_999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_chat_validation_empty_message():
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_and_threads_flow():
    thread_id = "test-thread-fastapi-1"

    # We mock graph.invoke to keep unit tests fast and independent of Hive API limits
    mock_reply = AIMessage(content="Could you please provide your username?")
    mock_response = {"messages": [mock_reply]}

    with patch("refund_agent.api.services.agent_service.graph.invoke", return_value=mock_response):
        chat_resp = client.post("/api/chat", json={"message": "hello", "thread_id": thread_id})
        assert chat_resp.status_code == 200
        data = chat_resp.json()
        assert data["thread_id"] == thread_id
        assert data["reply"] == "Could you please provide your username?"
        assert data["state"]["is_final"] is False

    # Check thread state endpoint
    state_resp = client.get(f"/api/threads/{thread_id}/state")
    assert state_resp.status_code == 200
    assert "stage" in state_resp.json()

    # Check thread history endpoint
    hist_resp = client.get(f"/api/threads/{thread_id}/history")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["thread_id"] == thread_id
    assert "messages" in hist_data

    # Check list threads
    threads_resp = client.get("/api/threads")
    assert threads_resp.status_code == 200
    threads_data = threads_resp.json()
    assert threads_data["total"] >= 1

    # Delete thread
    del_resp = client.delete(f"/api/threads/{thread_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True
