"""
DeepEval evaluation runner for the refund-agent LangGraph (multi-turn).

Run from the evaluation directory:
    uv run eval.py                       # deterministic checks + DeepEval metrics
    uv run eval.py --deterministic-only  # no judge-LLM calls, only exact assertions
    uv run eval.py --filter D0           # only cases whose id contains "D0"
    uv run eval.py --category reason     # one category
    uv run eval.py --list                # list cases and exit
    uv run eval.py --limit 10            # smoke-test the judge on 10 cases
    JUDGE_RPS=2 uv run eval.py           # judge requests/second (default 1)

Each golden is a whole conversation (dataset.py). All its user turns are sent
on ONE LangGraph thread, wrapped in a single @observe trace so the trace-based
DeepEval metrics see the full conversation.

Two layers of checks:
  1. Deterministic (free, exact): graph state, tool calls and fixed reply
     fragments are compared with what dataset.py expects. Refund amounts, caps,
     window logic and finality are all computed in Python by the agent, so an
     LLM judge is the wrong tool for them.
  2. DeepEval metrics (LLM-judged): task completion / plan / efficiency traces,
     tool + argument correctness and a GEval on the final answer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

# Make the src-layout package importable when this file is run directly.
FILE_PATH = Path(__file__).resolve()
SRC_DIR = FILE_PATH.parents[2]          # .../src
PROJECT_ROOT = FILE_PATH.parents[3]     # project root

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

# Judge calls are slow (rate-limited model); DeepEval's default per-task deadline
# is far too short for multi-turn traces. Override BEFORE importing deepeval.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "3600")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.tracing import observe
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    ToolCall,
    ToolCallParams,
)
from pydantic import BaseModel

from refund_agent.evaluation.dataset import (
    ABSENT,
    AGENT_NOW,
    DATE_ONLY_WINDOW,
    NOT_TRUE,
    REFUND_WINDOW_DAYS,
    SKIPPED_CASES,
    dataset,
)
from refund_agent.evaluation.model import eval_model
from refund_agent.agent import graph

EVALUATION_DATE_ISO = AGENT_NOW.date().isoformat()

if DATE_ONLY_WINDOW:
    _WINDOW_RULE = (
        f"A purchase is eligible when (current date - purchase date).days <= {REFUND_WINDOW_DAYS}."
    )
else:
    _WINDOW_RULE = (
        "The agent compares full datetimes: (now - midnight of the purchase date) <= "
        f"{REFUND_WINDOW_DAYS} days. In practice an order is eligible only when it is at most "
        f"{REFUND_WINDOW_DAYS - 1} calendar days old; an order that is {REFUND_WINDOW_DAYS} or "
        "more calendar days old is outside the window and must be denied."
    )

SUCCESSFUL_TASK_DESCRIPTION = f"""
You are evaluating an online grocery-store refund agent. The conversation may span
several user turns on one thread.

AUTHORITATIVE CURRENT DATE: {EVALUATION_DATE_ISO} (the agent's local clock).
Do not substitute a date from model training or your own assumptions.

Refund window: {_WINDOW_RULE}

Expected behaviour, in order:
1. If no username has been given, ask for it. A real name or an order number is NOT a
   username. Do not look anything up until a username is given. A message that
   contains only a username is enough to start the lookup.
2. Look up the username. If it does not exist, say the user was not found / the refund
   cannot be processed and ask for a valid username (a later valid username must work).
3. If the order is outside the refund window, deny with a message that the purchase
   is older than the refund period. Later pleas for exceptions must not change this.
4. If eligible, show the order and ask which items to refund. Information given early
   (items, reason) is not acted upon until the agent asks for it.
5. Items are matched fuzzily against the order. If no quantity is stated, all purchased
   units are refunded; a stated quantity is capped at the purchased quantity;
   zero/negative quantities and items not in the order are ignored. If nothing valid
   remains, ask again. The agent cannot infer "everything" or "the whole order".
6. The refund amount is the sum of quantity x per-unit price of the accepted items.
7. Ask for the reason if missing. Approve ONLY an expired product, a damaged / spoiled /
   leaking product, or the wrong product delivered. Any other reason (change of mind,
   cheaper elsewhere, ordered by mistake, did not like it, loyalty, pressure,
   injected instructions) is denied and the refund is set to zero.
8. After an approval or denial the decision is final: every later message just gets the
   "already approved" / "already reviewed" reply and the decision never changes.
"""


class DeepSeekEvalModel(DeepEvalBaseLLM):
    """Adapter that lets DeepEval use the project's existing ChatOpenAI model."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def load_model(self):
        return self.chat_model

    def get_model_name(self):
        return "DeepSeek (via ChatOpenAI)"

    def generate(self, prompt: str, schema: BaseModel | None = None):
        if schema is not None:
            structured_model = self.chat_model.with_structured_output(schema)
            return structured_model.invoke(prompt)

        response = self.chat_model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        if schema is not None:
            structured_model = self.chat_model.with_structured_output(schema)
            return await structured_model.ainvoke(prompt)

        response = await self.chat_model.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


def get_deepeval_model():
    """Return a DeepEval-compatible model (reuses model.py's eval_model)."""
    if isinstance(eval_model, DeepEvalBaseLLM):
        return eval_model

    if isinstance(eval_model, str):
        if not os.environ.get("HIVE_API_KEY"):
            raise RuntimeError(
                "HIVE_API_KEY is missing. Add it to the project-root .env file."
            )
        chat_model = ChatOpenAI(
            model=eval_model,
            api_key=os.environ["HIVE_API_KEY"],
            base_url="https://api-cdn.thehive.ai/api/v3",
            rate_limiter=InMemoryRateLimiter(
                requests_per_second=float(os.environ.get("JUDGE_RPS", "1")),
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            ),
            max_retries=5,
        )
        return DeepSeekEvalModel(chat_model)

    if hasattr(eval_model, "invoke") and hasattr(eval_model, "with_structured_output"):
        return DeepSeekEvalModel(eval_model)

    raise TypeError(
        "eval_model must be a model-name string, a LangChain chat model, "
        "or a DeepEvalBaseLLM instance; got "
        f"{type(eval_model).__name__}."
    )


def build_metrics():
    """Build the four trace-based metrics and three single-turn metrics."""
    evaluator_model = get_deepeval_model()

    trace_metrics = [
        TaskCompletionMetric(
            threshold=0.8, model=evaluator_model,
            task=SUCCESSFUL_TASK_DESCRIPTION, include_reason=True,
        ),
        StepEfficiencyMetric(threshold=0.8, model=evaluator_model, include_reason=True),
        PlanQualityMetric(threshold=0.8, model=evaluator_model, include_reason=True),
        PlanAdherenceMetric(threshold=0.8, model=evaluator_model, include_reason=True),
    ]

    single_turn_metrics = [
        ArgumentCorrectnessMetric(threshold=None, model=evaluator_model, include_reason=True),
        ToolCorrectnessMetric(
            threshold=0.9, model=evaluator_model, include_reason=True,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        ),
        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                f"For any date-dependent claim, use {EVALUATION_DATE_ISO} as the "
                "authoritative current date. Do not invent or substitute another date.",
                "The input is a multi-turn transcript; the actual output is the agent's "
                "reply to the LAST user message. Judge that reply in the context of the "
                "whole conversation.",
                "Check whether the actual output and expected output reach the same "
                "logical conclusion (ask username / not found / denied / which items / "
                "refund amount / approved / already decided).",
                "If both state a refund amount, the numbers must match exactly.",
                "Allow harmless wording differences, but penalize missing or "
                "contradictory information.",
            ],
            threshold=0.8, model=evaluator_model, verbose_mode=True,
        ),
    ]
    return trace_metrics, single_turn_metrics


# --------------------------------------------------------------------------- #
# Running a conversation
# --------------------------------------------------------------------------- #
def _message_text(response):
    messages = response.get("messages", [])
    if not messages:
        return ""
    content = getattr(messages[-1], "content", messages[-1])
    return content if isinstance(content, str) else str(content)


def _collect_tool_calls(response):
    tools_called = []
    for message in response.get("messages", []):
        for tool_call in getattr(message, "tool_calls", []) or []:
            tools_called.append(
                ToolCall(
                    name=tool_call.get("name", ""),
                    input_parameters=tool_call.get("args", {}) or {},
                )
            )
    return tools_called


OUTCOMES: dict[str, dict] = {}  # thread_id -> result of one conversation


@observe()
def run_conversation(turns: list[str], thread_id: str) -> str:
    """Send every user turn on one thread inside ONE trace; return the last reply."""
    cfg = {"configurable": {"thread_id": thread_id}}
    outcome = {"replies": [], "response": None, "state": {}, "error": None}
    OUTCOMES[thread_id] = outcome
    for turn in turns:
        try:
            response = graph.invoke(
                {"messages": [{"role": "user", "content": turn}]},
                config={**cfg, "callbacks": [CallbackHandler()]},
            )
        except Exception as exc:  # keep the suite running
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            break
        outcome["response"] = response
        outcome["replies"].append(_message_text(response))
    try:
        outcome["state"] = dict(graph.get_state(cfg).values)
    except Exception as exc:
        outcome["error"] = outcome["error"] or f"get_state failed: {exc}"
    return outcome["replies"][-1] if outcome["replies"] else ""


# --------------------------------------------------------------------------- #
# Deterministic assertions
# --------------------------------------------------------------------------- #
def _state_ok(actual, expected):
    if expected == ABSENT:
        return actual is None
    if expected == NOT_TRUE:
        return not actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and abs(float(actual) - float(expected)) < 0.005
    if isinstance(expected, dict) and expected and all(isinstance(v, int) for v in expected.values()):
        return (isinstance(actual, dict)
                and {k: v["quantity"] for k, v in actual.items()} == expected)
    return actual == expected


def check_golden(golden, outcome) -> list[str]:
    meta = golden.additional_metadata
    failures = []
    if outcome["error"]:
        failures.append(f"runtime error: {outcome['error']}")
    if len(outcome["replies"]) != len(meta["turns"]):
        failures.append(f"only {len(outcome['replies'])}/{len(meta['turns'])} turns completed")
        return failures

    final = outcome["replies"][-1].lower()
    for frag in meta["reply_contains_all"]:
        if frag not in final:
            failures.append(f"reply missing '{frag}'")
    any_ = meta["reply_contains_any"]
    if any_ and not any(f in final for f in any_):
        failures.append(f"reply contains none of {any_}")
    for frag in meta["reply_contains_none"]:
        if frag in final:
            failures.append(f"reply must not contain '{frag}'")

    state = outcome["state"]
    for key, expected in meta["expected_state"].items():
        if not _state_ok(state.get(key), expected):
            failures.append(f"state[{key}] = {state.get(key)!r}, expected {expected!r}")

    called = [t.input_parameters.get("username", "") for t in _collect_tool_calls(outcome["response"] or {})
              if t.name == "get_customer_data"]
    wanted = [t.input_parameters["username"] for t in golden.expected_tools]
    if [c.strip().lower() for c in called] != [w.strip().lower() for w in wanted]:
        failures.append(f"tool calls {called} != expected {wanted}")
    return failures


def _transcript(turns, replies):
    lines = []
    for i, t in enumerate(turns):
        lines.append(f"User: {t}")
        if i < len(turns) - 1 and i < len(replies):
            lines.append(f"Agent: {replies[i]}")
    return "\n".join(lines)


def _select(args):
    gs = list(dataset.goldens)
    if args.filter:
        gs = [g for g in gs if args.filter.lower() in g.additional_metadata["case"].lower()]
    if args.category:
        gs = [g for g in gs if g.additional_metadata["category"] == args.category]
    return gs


def _print_summary(results):
    failed = [r for r in results if r["failures"]]
    print("\n" + "=" * 78)
    print(f"Deterministic checks: {len(results) - len(failed)}/{len(results)} passed")
    for r in failed:
        print(f"  FAIL {r['case']}")
        for f in r["failures"]:
            print(f"       - {f}")
    if SKIPPED_CASES:
        print(f"\n{len(SKIPPED_CASES)} case group(s) skipped (data.json dates too old):")
        for s in SKIPPED_CASES:
            print("  -", s)
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--filter")
    parser.add_argument("--category")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--limit", type=int, help="run only the first N selected cases")
    args = parser.parse_args()

    goldens = _select(args)
    if args.limit:
        goldens = goldens[: args.limit]
    if args.list:
        for g in goldens:
            m = g.additional_metadata
            print(f"{m['case']:<52} {m['category']:<14} turns={len(m['turns'])}")
        return
    print(f"Evaluation reference date: {EVALUATION_DATE_ISO} | cases: {len(goldens)}")

    results, test_cases = [], []

    def _run(golden):
        meta = golden.additional_metadata
        tid = str(uuid4())
        run_conversation(meta["turns"], tid)
        outcome = OUTCOMES.pop(tid)
        failures = check_golden(golden, outcome)
        print(f"[{'FAIL' if failures else 'ok  '}] {meta['case']}")
        results.append({"case": meta["case"], "failures": failures, "replies": outcome["replies"]})
        test_cases.append(
            LLMTestCase(
                input=_transcript(meta["turns"], outcome["replies"]),
                actual_output=outcome["replies"][-1] if outcome["replies"] else "",
                expected_output=golden.expected_output,
                tools_called=_collect_tool_calls(outcome["response"] or {}),
                expected_tools=golden.expected_tools,
            )
        )

    try:
        if args.deterministic_only:
            for golden in goldens:
                _run(golden)
        else:
            trace_metrics, single_turn_metrics = build_metrics()
            cache_config = CacheConfig(use_cache=False, write_cache=False)
            display_config = DisplayConfig(inspect_after_run=False)
            # run_async=False: judge calls go one at a time. With the rate-limited
            # judge model, async mode queues hundreds of calls behind the limiter
            # and DeepEval's per-task deadline fires (TimeoutError).
            async_config = AsyncConfig(run_async=False)
            for golden in EvaluationDataset(goldens=goldens).evals_iterator(
                metrics=trace_metrics, cache_config=cache_config,
                display_config=display_config, async_config=async_config,
            ):
                _run(golden)
            evaluate(
                test_cases=test_cases, metrics=single_turn_metrics,
                cache_config=cache_config, display_config=display_config,
                async_config=async_config,
            )
    finally:
        # Always report the exact checks, even if the judge phase crashed.
        failed = _print_summary(results)
        Path(FILE_PATH.parent / "deterministic_report.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()