import os
import json
import difflib
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from refund_agent.tools import get_customer_data
from refund_agent.schemas import InputResponse, ValidateDate, RefundRequest, RefundReason
import re
load_dotenv()
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\.main",
)

REFUND_WINDOW_DAYS = 7


# 1. Graph state
class State(MessagesState):
    username: str | None
    user_exists: bool
    date_of_purchase: str
    items: list
    total_amount: str
    refund_applicable: bool
    items_for_refund: dict
    to_refund: float
    refund_reason: str
    refund_approved: bool


# 2. Model
deepseek_model = ChatOpenAI(
    model="deepseek-ai/deepseek-v4.1-flash-1017",
    base_url="https://api-cdn.thehive.ai/api/v3",
    temperature=0,
    api_key=os.environ.get("HIVE_API_KEY", ""),
    streaming=False,
    max_retries=3,
    max_tokens=4096,
)


def _last_human_text(state: State) -> str:
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )


# ---------------------------------------------------------------------------
# Entry router: resume from the right stage on every new user turn
# ---------------------------------------------------------------------------
def route_entry(state: State):
    if state.get("refund_approved") is not None:
        return "REFUND_DONE"          # final decision already made
    if state.get("to_refund") is not None:
        return "VALIDATE_REASON"      # amount calculated, waiting for the reason
    if state.get("refund_applicable"):
        return "VALIDATE_ITEMS"
    return "VALIDATION_NODE"


# ---------------------------------------------------------------------------
# Username extraction
# ---------------------------------------------------------------------------
input_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You extract usernames for an online grocery store.

        Your ONLY task is to extract the user's username.

        Rules:
        - Extract a username only when the user explicitly provides one.
        - If the user says something unrelated, return null.
        - Never invent or guess a username.
        - If the user says "my username is anas123",
          extract "anas123".
        - Return only the structured output.
        """
    ),
    ("human", "{user_input}"),
])

validation_chain = input_prompt | deepseek_model.with_structured_output(InputResponse)


def validation_node(state: State):
    if state.get("username"):
        return {}

    result = validation_chain.invoke({"user_input": _last_human_text(state)})

    if result.username:
        return {"username": result.username.strip()}
    return {}


def ask_username_node(state: State):
    return {"messages": [AIMessage(content="Could you please provide your username?")]}


def check_username(state: State):
    if state.get("username"):
        return "INFORMATION_NODE"
    return "ASK_USERNAME"


# ---------------------------------------------------------------------------
# Fetch order history
# ---------------------------------------------------------------------------
information_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant working for an online grocery store\n"
            "you are given a username, your task is to find the username in the database and use extract their order history\n"
            "if username is found, return it with the order history, else return false\n"
            "you have the necessary tool to search for the user and their order history\n"
        ),
        ("human", "username: {username}"),
    ]
)

information_chain = information_prompt | deepseek_model.bind_tools([get_customer_data])


def information_node(state: State):
    res = information_chain.invoke({"username": state["username"]})
    return {"messages": res}


tool_node = ToolNode([get_customer_data])


def save_info(state: State):
    last_message = json.loads(state["messages"][-1].content)
    if "error" in last_message:
        return {"user_exists": False}

    items = last_message["items"]
    if not isinstance(items, list):
        items = [items]

    return {
        "user_exists": True,
        "date_of_purchase": last_message["date_of_purchase"],
        "items": items,
        "total_amount": last_message["total_amount"],
    }


def if_user_exists(state: State):
    if state.get("user_exists"):
        return "VALIDATE_DATE"
    return "USER_DOESNT_EXIST"


def user_doesnt_exist(state: State):
    return {
        "username": None,  # let the user retry with another username
        "user_exists": False,
        "messages": [
            AIMessage(
                content="User not found, refund cannot be processed. Please provide a valid username."
            )
        ],
    }


# ---------------------------------------------------------------------------
# Refund window check
# ---------------------------------------------------------------------------
date_validation_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that works for an online grocery store.\n"
            f"If the date of purchase is within the last {REFUND_WINDOW_DAYS} days, return refund_applicable as true else false\n"
        ),
        (
            "human",
            "date of purchase: {date}\n"
            "today's date: {current_date}"
        ),
    ]
)

date_validation_chain = date_validation_prompt | deepseek_model.with_structured_output(ValidateDate)


def validate_date(state: State):
    now = datetime.now()
    raw = state["date_of_purchase"]

    # deterministic check first
    try:
        purchased = datetime.fromisoformat(str(raw).strip())
        if purchased.tzinfo is not None:
            purchased = purchased.replace(tzinfo=None)
        return {"refund_applicable": (now - purchased) <= timedelta(days=REFUND_WINDOW_DAYS)}
    except ValueError:
        pass

    # fallback to LLM only if the date format isn't ISO
    res = date_validation_chain.invoke({"date": raw, "current_date": now.isoformat()})
    return {"refund_applicable": res.refund_applicable}


def should_validate_tool_call(state: State):
    if state["refund_applicable"]:
        return "SHOW_INFO"
    return "DATE_VALIDATION_FAILED"


def date_validation_failed(state: State):
    return {
        "messages": [
            AIMessage(
                content="Sorry! Your purchase is older than the refund period, so a refund cannot be processed."
            )
        ]
    }


def show_info(state: State):
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Your details are:\n"
                    f" date of purchase: {state['date_of_purchase']}\n"
                    f" items: {state['items']}\n"
                    f" total amount: {state['total_amount']}\n"
                    f"For what items do you wish to get a refund? "
                    f"(quantity is optional)"
                )
            )
        ]
    }


# ---------------------------------------------------------------------------
# Extract requested items (LLM only extracts, Python validates + calculates)
# ---------------------------------------------------------------------------
extract_items_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract the grocery items the user wants a refund for.\n"
        "- Return each item's name and quantity.\n"
        "- quantity is null unless the user explicitly states a number.\n"
        "- Never guess or add items the user did not mention.\n"
        "- If the message names no items, return an empty list."
    ),
    ("human", "{user_input}"),
])

extract_items_chain = extract_items_prompt | deepseek_model.with_structured_output(RefundRequest)

def validate_items_node(state: State):
    text = _last_human_text(state)

    # reject negative quantities such as "-2 eggs" before the LLM can drop the sign
    if re.search(r"(?<![\w])-\s*\d", text):
        return {
            "items_for_refund": {},
            "messages": [
                AIMessage(
                    content="I couldn't find any of those items in your order. Which items would you like refunded?"
                )
            ],
        }

    request = extract_items_chain.invoke({"user_input": text})

    # order history keyed by normalized name
    catalog = {item["item"].strip().lower(): item for item in state["items"]}

    requested: dict[str, int] = {}
    for req in request.items:
        match = difflib.get_close_matches(
            req.item.strip().lower(), list(catalog), n=1, cutoff=0.8
        )
        if not match:
            continue  # not in order history -> ignore
        key = match[0]
        purchased = catalog[key]["quantity"]
        qty = purchased if req.quantity is None else req.quantity  # quantity optional
        if qty <= 0:
            continue
        requested[key] = min(requested.get(key, 0) + qty, purchased)  # never above purchased

    items_for_refund = {
        catalog[k]["item"]: {"quantity": q, "price": catalog[k]["price"]}
        for k, q in requested.items()
    }

    if not items_for_refund:
        return {
            "items_for_refund": {},
            "messages": [
                AIMessage(
                    content="I couldn't find any of those items in your order. Which items would you like refunded?"
                )
            ],
        }
    return {"items_for_refund": items_for_refund}


def after_validate_items(state: State):
    return "CALCULATE_REFUND" if state.get("items_for_refund") else END


# ---------------------------------------------------------------------------
# Last node: calculate and save the total refund amount in state
# ---------------------------------------------------------------------------
def calculate_refund_node(state: State):
    lines, total = [], 0.0
    for name, d in state["items_for_refund"].items():
        amount = d["quantity"] * float(d["price"])  # price = per-unit price
        total += amount
        lines.append(f"- {name} x{d['quantity']}: {amount:.2f}")
    total = round(total, 2)

    return {
        "to_refund": total,
        "messages": [
            AIMessage(
                content=(
                    "Refund summary:\n" + "\n".join(lines)
                    + f"\nTotal refund: {total:.2f}\n"
                    + "What is the reason for this refund?"
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Reason check: only expired / damaged / wrong product are allowed
# ---------------------------------------------------------------------------
ALLOWED_REASONS = {
    "expired": "the product was expired",
    "damaged": "the product was damaged",
    "wrong_product": "the wrong product was delivered",
}

reason_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Classify the user's reason for requesting a refund at an online grocery store.\n"
        "Categories:\n"
        "- expired: the product was expired / past its expiry date\n"
        "- damaged: the product was damaged, broken, spoiled or leaking\n"
        "- wrong_product: a different product than the one ordered was delivered\n"
        "- other: any other reason (no longer needed, change of mind, found cheaper, "
        "didn't like it, ordered by mistake, etc.)\n"
        "- not_stated: the message gives no reason at all\n"
        "Rules:\n"
        "- Only use expired, damaged or wrong_product if the user clearly states it.\n"
        "- Never guess a reason the user did not give.\n"
        "- Put a short paraphrase of the user's reason in `reason`."
    ),
    ("human", "{user_input}"),
])
reason_chain = reason_prompt | deepseek_model.with_structured_output(RefundReason)


def validate_reason_node(state: State):
    res = reason_chain.invoke({"user_input": _last_human_text(state)})

    if res.category == "not_stated":
        return {
            "messages": [
                AIMessage(content="Please tell me the reason for the refund so I can proceed.")
            ]
        }

    if res.category in ALLOWED_REASONS:
        return {
            "refund_reason": res.category,
            "refund_approved": True,
            "messages": [
                AIMessage(
                    content=(
                        f"Thanks, since {ALLOWED_REASONS[res.category]}, your refund of "
                        f"{state['to_refund']:.2f} has been approved."
                    )
                )
            ],
        }

    return {
        "refund_reason": res.category,
        "refund_approved": False,
        "to_refund": 0.0,  # nothing to pay out
        "messages": [
            AIMessage(
                content=(
                    "Sorry, refunds are only available if the product was expired, "
                    "damaged, or the wrong product was delivered. "
                    "Your refund request cannot be approved."
                )
            )
        ],
    }


def refund_done(state: State):
    if state.get("refund_approved"):
        text = f"Your refund of {state['to_refund']:.2f} has already been approved."
    else:
        text = "Your refund request was already reviewed and could not be approved."
    return {"messages": [AIMessage(content=text)]}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
builder = StateGraph(State)

builder.add_node("VALIDATION_NODE", validation_node)
builder.add_node("ASK_USERNAME", ask_username_node)
builder.add_node("INFORMATION_NODE", information_node)
builder.add_node("tool_node", tool_node)
builder.add_node("SAVE_INFO", save_info)
builder.add_node("USER_DOESNT_EXIST", user_doesnt_exist)
builder.add_node("VALIDATE_DATE", validate_date)
builder.add_node("DATE_VALIDATION_FAILED", date_validation_failed)
builder.add_node("SHOW_INFO", show_info)
builder.add_node("VALIDATE_ITEMS", validate_items_node)
builder.add_node("CALCULATE_REFUND", calculate_refund_node)
builder.add_node("VALIDATE_REASON", validate_reason_node)
builder.add_node("REFUND_DONE", refund_done)

builder.add_conditional_edges(
    START,
    route_entry,
    {
        "VALIDATION_NODE": "VALIDATION_NODE",
        "VALIDATE_ITEMS": "VALIDATE_ITEMS",
        "VALIDATE_REASON": "VALIDATE_REASON",
        "REFUND_DONE": "REFUND_DONE",
    },
)

builder.add_conditional_edges(
    "VALIDATION_NODE",
    check_username,
    {"INFORMATION_NODE": "INFORMATION_NODE", "ASK_USERNAME": "ASK_USERNAME"},
)
builder.add_edge("ASK_USERNAME", END)

builder.add_edge("INFORMATION_NODE", "tool_node")
builder.add_edge("tool_node", "SAVE_INFO")

builder.add_conditional_edges(
    "SAVE_INFO",
    if_user_exists,
    {"VALIDATE_DATE": "VALIDATE_DATE", "USER_DOESNT_EXIST": "USER_DOESNT_EXIST"},
)
builder.add_edge("USER_DOESNT_EXIST", END)

builder.add_conditional_edges(
    "VALIDATE_DATE",
    should_validate_tool_call,
    {"SHOW_INFO": "SHOW_INFO", "DATE_VALIDATION_FAILED": "DATE_VALIDATION_FAILED"},
)
builder.add_edge("DATE_VALIDATION_FAILED", END)
builder.add_edge("SHOW_INFO", END)  # turn ends here, waits for the user's items

builder.add_conditional_edges(
    "VALIDATE_ITEMS",
    after_validate_items,
    {"CALCULATE_REFUND": "CALCULATE_REFUND", END: END},
)
builder.add_edge("CALCULATE_REFUND", END)   # turn ends, waits for the reason
builder.add_edge("VALIDATE_REASON", END)
builder.add_edge("REFUND_DONE", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

try:
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
except Exception:
    pass  # needs network; don't crash startup


config = {"configurable": {"thread_id": "client_1"}}


if __name__ == "__main__":
    print("Refund Agent started. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print("Agent:", result["messages"][-1].content)