"""Chunk 4: Human-in-the-loop approval as a PAUSABLE LangGraph workflow.

Two approval architectures live in this project (they COMPLEMENT each other):

  Pattern A (Chunks 1-3): DB rows. The agent creates a pending ``Approval`` row
      and finishes its turn; an admin flips it later. The ``approvals`` table
      is the durable record and the admin dashboard's source of truth.

  Pattern B (this file): the agent graph itself PAUSES mid-run at the approval
      node via ``interrupt()``. Its full state is saved by a CHECKPOINTER, and
      it RESUMES only when a human passes the decision back in with
      ``Command(resume=...)``.

Graph shape:

    START -> request -> approval -> resolve -> END
               \-> END   (nothing to approve: error or duplicate request)

Run the demo:
    python -m agents.approval_workflow
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.tool_calling_agent import (
    approve_approval,
    cancel_order,
    decline_approval,
    process_refund,
)

# Dev: SQLite save-file (same idea as customer_support.db).
# Prod: swap for PostgresSaver — one-line change.
CHECKPOINT_DB = "approval_checkpoints.db"


class ApprovalState(TypedDict):
    """Everything the workflow knows. Saved WHOLE by the checkpointer at the pause."""

    # inputs (set when the workflow starts)
    order_id: str
    kind: str            # "refund" | "cancel"
    reason: str
    requested_by: str
    # written by the nodes
    approval_id: int     # -1 = nothing to approve (error or duplicate request)
    decision: str        # "approved" | "declined" — only known AFTER resume
    resolved_by: str     # which admin decided (arrives with the resume)
    result: dict         # output of the last tool call


# ============================================================
# Nodes — each does ONE thing (same rule as langgraph_agent.py)
# ============================================================

def request_node(state: ApprovalState) -> dict:
    """Create the PENDING request (Pattern A). The agent prepares, never executes."""
    print(f"\n  [NODE: request] creating pending {state['kind']} for order {state['order_id']}...")

    if state["kind"] == "refund":
        result = process_refund(state["order_id"], state["reason"], requested_by=state["requested_by"])
    elif state["kind"] == "cancel":
        result = cancel_order(state["order_id"], requested_by=state["requested_by"])
    else:
        result = {"error": f"Unknown approval kind: {state['kind']}"}

    if "error" in result or result.get("already_requested"):
        # Nothing new to approve -> the router sends this run straight to END.
        return {"result": result, "approval_id": -1}

    return {"result": result, "approval_id": result["approval_id"]}


def approval_node(state: ApprovalState) -> dict:
    """THE PAUSE.

    ``interrupt(payload)`` behaves differently on each pass:
      - FIRST run : saves the whole state via the checkpointer and STOPS the
                    graph. This line never finishes executing.
      - ON RESUME : returns whatever the human passed in ``Command(resume=...)``
                    and the node continues as if it had waited here all along.
    """
    print("  [NODE: approval] PAUSING — waiting for a human decision...")
    answer = interrupt({
        "question": f"Approve {state['kind']} for order {state['order_id']}?",
        "approval_id": state["approval_id"],
        "reason": state["reason"],
        "amount": state["result"].get("refund_amount", 0.0),
    })
    print(f"  [NODE: approval] resumed: decision={answer['decision']!r} by={answer['by']!r}")
    return {"decision": answer["decision"], "resolved_by": answer["by"]}


def resolve_node(state: ApprovalState) -> dict:
    """Apply the human's decision exactly once (Chunk 3's tools own the guards)."""
    print(f"  [NODE: resolve] applying decision={state['decision']}...")
    if state["decision"] == "approved":
        result = approve_approval(state["approval_id"], approved_by=state["resolved_by"])
    else:
        result = decline_approval(state["approval_id"], declined_by=state["resolved_by"])
    return {"result": result}


def route_after_request(state: ApprovalState) -> str:
    """Same conditional-edge idea as route_message() in langgraph_agent.py."""
    if state.get("approval_id", -1) == -1:
        print("  [ROUTE] -> END (nothing to approve)")
        return END
    print("  [ROUTE] -> approval (human gate)")
    return "approval"


# ============================================================
# Graph builder — with a CHECKPOINTER so it can pause/resume
# ============================================================

_checkpointer: SqliteSaver | None = None
_app = None


def get_checkpointer() -> SqliteSaver:
    """One long-lived saver. ``check_same_thread=False`` mirrors db/base.py
    because FastAPI runs handlers in worker threads."""
    global _checkpointer
    if _checkpointer is None:
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    return _checkpointer


def build_approval_graph():
    graph = StateGraph(ApprovalState)

    graph.add_node("request", request_node)
    graph.add_node("approval", approval_node)
    graph.add_node("resolve", resolve_node)

    graph.add_edge(START, "request")
    graph.add_conditional_edges("request", route_after_request, {"approval": "approval", END: END})
    graph.add_edge("approval", "resolve")
    graph.add_edge("resolve", END)

    # THE one line that makes pausing possible: compile WITH a checkpointer.
    return graph.compile(checkpointer=get_checkpointer())


def get_approval_app():
    global _app
    if _app is None:
        _app = build_approval_graph()
    return _app


# ============================================================
# Public API — these two functions are what the API layer (Chunk 6) calls
# ============================================================

def start_approval(order_id: str, kind: str, reason: str, requested_by: str = "system") -> dict:
    """Run the graph until it PAUSES at the human gate.

    Returns the ``thread_id`` — the name of the save-file an admin will
    resume later. In production this is what /chat returns to the customer.
    """
    app = get_approval_app()
    thread_id = f"{kind}-{order_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial: ApprovalState = {
        "order_id": order_id,
        "kind": kind,
        "reason": reason,
        "requested_by": requested_by,
        "approval_id": 0,
        "decision": "",
        "resolved_by": "",
        "result": {},
    }
    final = app.invoke(initial, config)

    interrupts = final.get("__interrupt__") or []
    return {
        "thread_id": thread_id,
        "paused": bool(interrupts),
        "question": interrupts[0].value if interrupts else None,
        "immediate_result": final.get("result"),
    }


def resume_approval(thread_id: str, decision: str, by: str) -> dict:
    """Continue a paused graph with the human's answer.

    This is what the admin endpoint (Chunk 6) calls after authentication.
    """
    app = get_approval_app()
    config = {"configurable": {"thread_id": thread_id}}
    final = app.invoke(Command(resume={"decision": decision, "by": by}), config)
    return {
        "thread_id": thread_id,
        "decision": final.get("decision"),
        "result": final.get("result"),
    }


# ============================================================
# Demo — pause and resume inside one process (in real life these are
# two separate API calls, hours apart)
# ============================================================

def run_demo() -> None:
    print("=" * 60)
    print("CHUNK 4 DEMO — pausable approval workflow")
    print("=" * 60)

    print("\n--- 1) Refund for order 3310: agent STARTS, then PAUSES ---")
    started = start_approval("3310", "refund", "demo: speaker arrived broken", requested_by="demo-user")
    print(f"thread_id : {started['thread_id']}")
    print(f"paused    : {started['paused']}")
    print(f"question  : {started['question']}")

    print("\n--- 2) (hours later) an admin approves: the graph RESUMES ---")
    finished = resume_approval(started["thread_id"], "approved", by="demo-admin")
    print(f"decision  : {finished['decision']}")
    print(f"result    : {finished['result']}")

    print("\n--- 3) Cancel order 1234: pauses, admin DECLINES ---")
    cancel = start_approval("1234", "cancel", "demo: changed my mind", requested_by="demo-user")
    print(f"thread_id : {cancel['thread_id']}  paused={cancel['paused']}")
    declined = resume_approval(cancel["thread_id"], "declined", by="demo-admin")
    print(f"result    : {declined['result']}")

    print("\n--- 4) DUPLICATE refund request: no pause at all (idempotent) ---")
    dup = start_approval("3310", "refund", "demo: duplicate attempt", requested_by="demo-user")
    print(f"paused           : {dup['paused']}")
    print(f"immediate_result : {dup['immediate_result']}")

    print("\nNote: this demo approved one real refund and declined one real cancel.")
    print("Reset demo data anytime: delete customer_support.db, then: python -m db.seed")


if __name__ == "__main__":
    run_demo()
