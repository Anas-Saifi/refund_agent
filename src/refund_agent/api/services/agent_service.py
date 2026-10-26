import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

from refund_agent.agent import checkpointer, graph
from refund_agent.api.schemas import (
    AgentStateSummary,
    ChatResponse,
    MessageRecord,
    ThreadHistoryResponse,
    ThreadSummary,
)


class AgentService:
    def __init__(self):
        # In-memory registry for thread metadata
        self._threads: dict[str, dict] = {}

    def _get_or_create_thread(self, thread_id: str | None) -> str:
        tid = thread_id.strip() if thread_id and thread_id.strip() else str(uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        if tid not in self._threads:
            self._threads[tid] = {
                "created_at": now_iso,
                "last_active_at": now_iso,
                "turn_count": 0,
                "last_reply": None,
            }
        else:
            self._threads[tid]["last_active_at"] = now_iso
        return tid

    def _determine_stage(self, state: dict) -> str:
        if state.get("refund_approved") is not None:
            return "completed"
        if state.get("to_refund") is not None:
            return "awaiting_reason"
        if state.get("refund_applicable"):
            return "selecting_items"
        if state.get("user_exists"):
            return "validating_order"
        if state.get("username"):
            return "order_lookup"
        return "need_username"

    def _build_state_summary(self, state_dict: dict) -> AgentStateSummary:
        is_final = state_dict.get("refund_approved") is not None
        stage = self._determine_stage(state_dict)

        return AgentStateSummary(
            username=state_dict.get("username"),
            user_exists=state_dict.get("user_exists"),
            date_of_purchase=state_dict.get("date_of_purchase"),
            items=state_dict.get("items"),
            total_amount=state_dict.get("total_amount"),
            refund_applicable=state_dict.get("refund_applicable"),
            items_for_refund=state_dict.get("items_for_refund"),
            to_refund=state_dict.get("to_refund"),
            refund_reason=state_dict.get("refund_reason"),
            refund_approved=state_dict.get("refund_approved"),
            is_final=is_final,
            stage=stage,
        )

    def _extract_reply(self, response: dict) -> str:
        messages = response.get("messages", [])
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                return m.content
            if isinstance(m, dict) and m.get("role") in {"assistant", "ai"}:
                return m.get("content", "")
        if messages:
            last = messages[-1]
            return getattr(last, "content", str(last))
        return "No response generated."

    def _map_message_to_record(self, msg) -> MessageRecord:
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            role = "user"
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        elif isinstance(msg, AIMessage) or (isinstance(msg, dict) and msg.get("role") in {"assistant", "ai"}):
            role = "assistant"
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        elif isinstance(msg, ToolMessage) or (isinstance(msg, dict) and msg.get("role") == "tool"):
            role = "tool"
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        else:
            role = "system"
            content = msg.content if hasattr(msg, "content") else str(msg)
        return MessageRecord(role=role, content=content)

    async def process_chat(self, message: str, thread_id: str | None = None) -> ChatResponse:
        """Process a user message synchronously through the LangGraph agent."""
        tid = self._get_or_create_thread(thread_id)
        cfg = {"configurable": {"thread_id": tid}}

        # Run invocation in thread pool to avoid blocking async event loop
        response = await asyncio.to_thread(
            graph.invoke,
            {"messages": [HumanMessage(content=message)]},
            config=cfg,
        )

        reply = self._extract_reply(response)

        # Update thread registry metadata
        self._threads[tid]["turn_count"] += 1
        self._threads[tid]["last_reply"] = reply
        self._threads[tid]["last_active_at"] = datetime.now(timezone.utc).isoformat()

        # Fetch current state snapshot
        snap = graph.get_state(cfg)
        summary = self._build_state_summary(snap.values)

        return ChatResponse(
            thread_id=tid,
            reply=reply,
            state=summary,
            is_final=summary.is_final,
        )

    async def stream_chat(
        self, message: str, thread_id: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream node transitions and the final reply via Server-Sent Events (SSE)."""
        tid = self._get_or_create_thread(thread_id)
        cfg = {"configurable": {"thread_id": tid}}

        yield f"event: start\ndata: {json.dumps({'thread_id': tid})}\n\n"

        # Stream node execution updates
        final_reply = ""
        try:
            # We run stream in thread loop or use graph.stream
            def _run_stream():
                return list(
                    graph.stream(
                        {"messages": [HumanMessage(content=message)]},
                        config=cfg,
                        stream_mode="updates",
                    )
                )

            updates = await asyncio.to_thread(_run_stream)

            for step in updates:
                node_name = list(step.keys())[0] if step else "unknown"
                node_output = step[node_name]
                node_data = {"node": node_name}

                # If this node produced messages, check for reply content
                if isinstance(node_output, dict) and "messages" in node_output:
                    msgs = node_output["messages"]
                    if msgs:
                        last = msgs[-1] if isinstance(msgs, list) else msgs
                        if hasattr(last, "content"):
                            final_reply = last.content
                            node_data["message"] = last.content

                yield f"event: node_update\ndata: {json.dumps(node_data)}\n\n"

            # Get current state
            snap = graph.get_state(cfg)
            summary = self._build_state_summary(snap.values)

            # Fallback if final_reply was not caught in updates
            if not final_reply:
                final_reply = self._extract_reply(snap.values)

            self._threads[tid]["turn_count"] += 1
            self._threads[tid]["last_reply"] = final_reply
            self._threads[tid]["last_active_at"] = datetime.now(timezone.utc).isoformat()

            yield f"event: message\ndata: {json.dumps({'reply': final_reply})}\n\n"
            yield f"event: state\ndata: {json.dumps(summary.model_dump())}\n\n"
            yield f"event: done\ndata: {json.dumps({'thread_id': tid, 'status': 'completed'})}\n\n"

        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    def get_thread_state(self, thread_id: str) -> AgentStateSummary:
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = graph.get_state(cfg)
        return self._build_state_summary(snap.values)

    def get_thread_history(self, thread_id: str) -> ThreadHistoryResponse:
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = graph.get_state(cfg)
        messages = snap.values.get("messages", [])
        records = [self._map_message_to_record(m) for m in messages]
        summary = self._build_state_summary(snap.values)

        return ThreadHistoryResponse(
            thread_id=thread_id,
            message_count=len(records),
            messages=records,
            state=summary,
        )

    def delete_thread(self, thread_id: str) -> bool:
        try:
            checkpointer.delete_thread(thread_id)
        except Exception:
            pass
        if thread_id in self._threads:
            del self._threads[thread_id]
            return True
        return True

    def list_threads(self) -> list[ThreadSummary]:
        results = []
        for tid, meta in self._threads.items():
            state = self.get_thread_state(tid)
            results.append(
                ThreadSummary(
                    thread_id=tid,
                    created_at=meta["created_at"],
                    last_active_at=meta["last_active_at"],
                    turn_count=meta["turn_count"],
                    last_reply=meta.get("last_reply"),
                    username=state.username,
                    refund_approved=state.refund_approved,
                    is_final=state.is_final,
                )
            )
        # Sort by most recently active
        results.sort(key=lambda t: t.last_active_at, reverse=True)
        return results

    def active_thread_count(self) -> int:
        return len(self._threads)


agent_service = AgentService()
