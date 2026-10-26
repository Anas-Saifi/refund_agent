# Refund Agent - FastAPI Backend

A LangGraph-powered conversational AI agent and FastAPI backend for managing grocery store refund requests.

---

## Architecture & Workflow

The agent uses a stateful multi-turn LangGraph state machine:

1. **Username Identification**:
   - Asks for and extracts username using structured LLM output (`InputResponse`).
2. **Order Lookup**:
   - Calls the `get_customer_data` tool to fetch customer order history from `data.json`.
3. **Refund Window Eligibility**:
   - Validates that the purchase date is within **7 days** of today's date. Orders older than 7 days are denied.
4. **Item Extraction & Validation**:
   - Extracts requested items and optional quantities (`RefundRequest`).
   - Uses fuzzy string matching against order history (`difflib`, cutoff 0.8).
   - Validates and caps quantities to the purchased amount. Negative or zero quantities and non-matching items are discarded.
5. **Refund Calculation & Reason Validation**:
   - Calculates the exact refund amount (`calculate_refund`).
   - Classifies customer reason (`RefundReason`):
     - **Approved reasons**: `expired`, `damaged` (broken, spoiled, leaking), `wrong_product`.
     - **Denied reasons**: `other` (changed mind, found cheaper, mistake, etc.) -> zeroes refund amount and denies.
6. **Decision Finality**:
   - Once approved or denied, subsequent queries are handled idempotently (`REFUND_DONE`).

---

## API Endpoints

### 1. Chat
- **`POST /api/chat`**: Standard JSON chat endpoint.
  ```json
  {
    "message": "My username is alextaylor1001",
    "thread_id": "optional-uuid"
  }
  ```
  Returns:
  ```json
  {
    "thread_id": "optional-uuid",
    "reply": "Your details are...",
    "state": {
      "username": "alextaylor1001",
      "stage": "selecting_items",
      "is_final": false
    },
    "is_final": false
  }
  ```
- **`POST /api/chat/stream`**: Server-Sent Events (SSE) streaming node transitions (`node_update`), assistant replies (`message`), and state snapshots (`state`).

### 2. Threads Management
- **`GET /api/threads`**: List active sessions with timestamps and stages.
- **`GET /api/threads/{thread_id}/state`**: Get snapshot of thread state (items, refund amount, decision).
- **`GET /api/threads/{thread_id}/history`**: Get message transcript.
- **`DELETE /api/threads/{thread_id}`**: Reset/clear a conversation thread.

### 3. Customer & Policy Data
- **`GET /api/customers`**: List customers in database with order summary and refund eligibility.
- **`GET /api/customers/{username}`**: Detailed customer record and order items.
- **`GET /api/policy`**: Returns business rules (7-day window, eligible reasons).
- **`GET /health`**: Health check and API readiness status.

---

## Running the Server

### Development Server
```bash
uv run uvicorn refund_agent.api.main:app --reload --port 8000
```
Or using the registered CLI command:
```bash
uv run refund-server
```

### Interactive Documentation
Once started, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Running Tests

Run the test suite:
```bash
uv run pytest tests/test_api.py -v
```
