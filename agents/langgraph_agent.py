"""
Skill 5: LangGraph Orchestration
=================================

Same behavior as manual_agent_loop.py.
1/4 the code. 10x more maintainable.

Graph structure:
  START → classify → [route] → tool     → END
                             → rag      → END
                             → escalate → END
                             → direct   → END

Every node does ONE thing.
Every node reads from state and writes back to state.
The router only reads — it never writes.
"""

import os
import time                     
import json   
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Optional
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# ── Import our Skills ────────────────────────────────────────────────────────
from ticket_classifier import create_classifier, classify_ticket
from tool_calling_agent import create_agent, run_agent, escalate_to_human

# ── RAG (optional) ───────────────────────────────────────────────────────────
try:
    from rag_pipeline import get_connection, setup_table, query as rag_query
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

load_dotenv()

# ============================================================
# SECTION 1: AgentState — The Order Ticket
# ============================================================
# This is the FIRST thing you design. Before any node. Before any edge.
# Every node reads from this. Every node writes back to this.
# Nothing lives outside state. No hidden variables. No global mutable state.
#
# Design rule: Every field answers one of two questions:
#   1. "What do we know about the customer's request?"
#   2. "What has the agent done so far?"

class AgentState(TypedDict):
    # ── What we know about the request ──────────────────────
    user_message: str           # The raw customer message
    intent: str                 # What they want (written by classify_node)
    urgency: str                # How urgent (written by classify_node)
    sentiment: str              # How they feel (written by classify_node)
    confidence: float           # How sure the classifier is (written by classify_node)
    tool_needed: bool           # Whether the request needs a tool call (written by classify_node)

    # ── What the agent has done ──────────────────────────────
    route: str                  # Which path was taken (written by route_message)
    tool_called: Optional[str]  # Which tool was called, if any (written by tool/escalate node)
    tool_result: Optional[dict] # What the tool returned (written by tool/escalate node)
    rag_chunks: Optional[list]  # What RAG retrieved (written by rag_node)

    # ── Final output ─────────────────────────────────────────
    final_response: str         # What we tell the customer (written by action nodes)


# ============================================================
# SECTION 2: Shared Clients
# ============================================================
# Created ONCE, outside nodes. Nodes get called on every message.
# Recreating a client on every call = wasted time + memory.
# These are stateless — one instance safely serves all calls.

classifier_client = create_classifier()
tool_client       = create_agent()
genai_client      = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# RAG connection (optional)
conn = None
if RAG_AVAILABLE:
    try:
        conn = get_connection()
        setup_table(conn)
    except Exception as e:
        print(f"[WARNING] RAG unavailable: {e}")
        RAG_AVAILABLE = False

DIRECT_RESPONSE_PROMPT = """You are a friendly customer support agent for Acme Corp.
The customer has sent a general message that doesn't require looking up orders,
processing refunds, or searching a knowledge base.
Respond naturally, warmly, and concisely. Keep it under 3 sentences.
NEVER fabricate order details or policy specifics."""


# ============================================================
# SECTION 3: Nodes — Each Kitchen Station
# ============================================================
# Golden rule:
#   - Takes the FULL state
#   - Does ONE thing
#   - Returns ONLY the fields it changed (partial dict)
# LangGraph merges the partial dict back into full state automatically.

# ── Node 1: classify_node ────────────────────────────────────────────────────
# The Ticket Taker — "What does the customer want, and how urgent is it?"

def classify_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"]
    WRITES: intent, urgency, sentiment, confidence, tool_needed

    This is ALWAYS the first node. It classifies the message using Skill 1.
    All downstream decisions (routing, escalation threshold) depend on this.
    """
    print(f"\n  [NODE: classify] Classifying message...")

    result = classify_ticket(classifier_client, state["user_message"])

    print(f"  [classify] intent={result.intent.value} | urgency={result.urgency.value} | confidence={result.confidence:.2f} | tool_needed={result.requires_tool}")

    # Return ONLY what this node is responsible for writing.
    # Note: .value converts Enum → string (strings serialize cleanly to DB, Enums don't)
    return {
        "intent":      result.intent.value,
        "urgency":     result.urgency.value,
        "sentiment":   result.sentiment.value,
        "confidence":  result.confidence,
        "tool_needed": result.requires_tool,
    }


# ── Node 2: tool_node ────────────────────────────────────────────────────────
# The Grill Station — handles order lookups and refunds

def tool_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"]
    WRITES: tool_called, tool_result, final_response

    Called when intent is order_status or refund_request.
    Delegates to Skill 2's run_agent() — the full tool-calling loop:
      LLM picks tool → execute → LLM writes response using result.

    Why use run_agent() here but NOT in escalate_node?
    Because HERE, the user's message contains the signal (order number, refund intent).
    The LLM can read the message and reliably pick the right tool.
    """
    print(f"  [NODE: tool] Routing to tool calling agent...")

    result = run_agent(tool_client, state["user_message"])

    if result["tool_called"]:
        print(f"  [tool] Called: {result['tool_called']}")
    else:
        print(f"  [tool] No tool called — direct response")

    return {
        "tool_called":    result["tool_called"],
        "tool_result":    result["tool_result"],
        "final_response": result["final_response"],
    }


# ── Node 3: rag_node ─────────────────────────────────────────────────────────
# The Sauce Station — searches the knowledge base

def rag_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"]
    WRITES: rag_chunks, final_response

    Called when intent is product_question or complaint (non-critical urgency).
    Searches knowledge base and generates a grounded answer.

    Why return a fallback instead of crashing?
    A node that crashes kills the whole graph.
    A node that returns a fallback keeps the agent alive.
    Production rule: nodes NEVER crash — they return graceful fallbacks.
    """
    print(f"  [NODE: rag] Routing to RAG pipeline...")

    if not RAG_AVAILABLE or conn is None:
        return {
            "rag_chunks":     [],
            "final_response": "I don't have access to the knowledge base right now. "
                              "Let me connect you with a team member who can help.",
        }

    rag_result = rag_query(genai_client, conn, state["user_message"], verbose=False)

    if rag_result.retrieval_failed:
        print(f"  [rag] Retrieval failed — no relevant chunks found")
    else:
        print(f"  [rag] Retrieved from: {rag_result.sources}")

    return {
        "rag_chunks":     rag_result.sources,
        "final_response": rag_result.answer,
    }


# ── Node 4: escalate_node ────────────────────────────────────────────────────
# The Emergency Station — hands off to a human agent

def escalate_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"], state["urgency"]
    WRITES: tool_called, tool_result, final_response

    Called when urgency is critical OR confidence < 0.60.

    KEY ARCHITECTURAL DECISION:
    We call escalate_to_human() DIRECTLY — no run_agent(), no LLM re-decision.

    Why? The urgency/confidence signal lives in state (metadata), NOT in the
    user's message. If we sent the message to run_agent(), the LLM would only
    see the message — it has no access to state. It might pick order_lookup
    instead of escalate_to_human.

    The decision is ALREADY made by the router. We just execute it.
    Use LLMs for decisions that NEED intelligence.
    Use direct calls for actions that are ALREADY decided.
    """
    print(f"  [NODE: escalate] Escalating (urgency={state.get('urgency', 'high')})...")

    result = escalate_to_human(
        reason=f"Customer message: {state['user_message']}",
        urgency=state.get("urgency", "high"),
    )

    return {
        "tool_called":    "escalate_to_human",
        "tool_result":    result,
        "final_response": f"I'm connecting you with a human agent right away. "
                          f"Your case ID is {result['escalation_id']}. "
                          f"Estimated wait: {result['estimated_wait_time']}.",
    }


# ── Node 5: direct_node ──────────────────────────────────────────────────────
# The Simple Counter — greetings, thanks, general questions

def direct_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"]
    WRITES: final_response

    Called for general messages that don't need tools or RAG.
    Simple LLM response — no tools, no knowledge base.

    Why a separate node instead of handling inline?
    Testable in isolation: direct_node({"user_message": "hello"}) works alone.
    """
    print(f"  [NODE: direct] Direct LLM response (no tools, no RAG)...")

    response = genai_client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp",
        contents=state["user_message"],
        config=types.GenerateContentConfig(
            system_instruction=DIRECT_RESPONSE_PROMPT,
            temperature=0.7,
        ),
    )

    return {"final_response": response.text}


# ============================================================
# SECTION 4: Router — The Traffic Controller
# ============================================================
# The router is NOT a node. It's a pure function.
# It READS state and RETURNS a string (the next node name).
# It does NOT change state. No side effects. No writes.
#
# Why not a node?
# Nodes do work and change state.
# The router only reads and decides. That's a different role.

def route_message(state: AgentState) -> str:
    """
    READS:  state["urgency"], state["confidence"], state["intent"]
    RETURNS: string — the name of the next node to go to

    This is the EXACT same logic as step_route() in manual_agent_loop.py.
    But now it reads from state (single source of truth) instead of
    reaching into a TicketClassification object + accepting a use_rag flag.

    Decision tree (same as manual loop):
      critical urgency  → "escalate"   (safety first)
      low confidence    → "escalate"   (when unsure, hand to human)
      order/refund      → "tool"       (use tools)
      product/complaint → "rag"        (search knowledge base)
      everything else   → "direct"     (simple response)
    """
    urgency    = state["urgency"]
    confidence = state["confidence"]
    intent     = state["intent"]

    # Safety first - escalate critical situations immediately
    if urgency == "critical":
        print(f"  [ROUTE] -> ESCALATE (urgency=critical)")
        return "escalate"

    # Low confidence - agent isn't sure what's happening, hand to human
    if confidence < 0.60:
        print(f"  [ROUTE] -> ESCALATE (confidence={confidence:.2f} < 0.60)")
        return "escalate"

    # Order-related - delegate to tool calling agent
    if intent in ("order_status", "refund_request"):
        print(f"  [ROUTE] -> TOOL (intent={intent})")
        return "tool"

    # Knowledge-related - search knowledge base
    if intent in ("product_question", "complaint"):
        print(f"  [ROUTE] -> RAG (intent={intent})")
        return "rag"

    # Everything else - simple LLM response
    print(f"  [ROUTE] -> DIRECT (intent={intent})")
    return "direct"


# ============================================================
# SECTION 5: Graph Builder — Wiring the Kitchen
# ============================================================
# This function IS the architecture diagram.
# Read it top to bottom and you see the entire flow.
# Adding a new route = ONE new node + ONE new edge.

def build_graph():
    """
    Build and compile the customer support agent graph.

    Replace this entire manual loop routing:
        if route == "tool":     step_tool(...)
        elif route == "rag":    step_rag(...)
        elif route == "escalate": step_escalate(...)
        elif route == "direct": step_direct(...)

    With this declarative graph:
        graph.add_conditional_edges("classify", route_message, {...})

    The flow is VISIBLE. The logic is ISOLATED. Every path is TESTABLE.
    """

    # Step 1: Create graph with our state shape
    # StateGraph needs the shape so it can correctly merge partial dicts from each node
    graph = StateGraph(AgentState)

    # Step 2: Register all nodes (kitchen stations)
    graph.add_node("classify",  classify_node)
    graph.add_node("tool",      tool_node)
    graph.add_node("rag",       rag_node)
    graph.add_node("escalate",  escalate_node)
    graph.add_node("direct",    direct_node)

    # Step 3: Set entry point — every message starts at classification
    graph.add_edge(START, "classify")

    # Step 4: After classify, route based on state
    # route_message() returns a string → that string maps to a node name
    graph.add_conditional_edges(
        "classify",       # FROM this node...
        route_message,    # ...CALL this function to decide...
        {                 # ...and GO TO whichever node it returns:
            "tool":     "tool",
            "rag":      "rag",
            "escalate": "escalate",
            "direct":   "direct",
        }
    )

    # Step 5: All action nodes lead to END
    # Each action node already writes final_response — nothing left to do
    graph.add_edge("tool",     END)
    graph.add_edge("rag",      END)
    graph.add_edge("escalate", END)
    graph.add_edge("direct",   END)

    # Step 6: Compile — lock the graph and get a runnable
    # compile() validates the graph (no orphan nodes, no missing edges)
    # Returns an object you call with app.invoke()
    app = graph.compile()

    return app


# ============================================================
# SECTION 6: Runner — Placing the Order
# ============================================================

def run(user_message: str, app=None) -> str:
    """
    Run the agent for a single user message.
    Equivalent to ONE TURN in the manual agent loop.

    app is optional — pass it in to avoid rebuilding the graph on every call.
    If not passed, builds a new graph (useful for one-off calls).
    """
    if app is None:
        app = build_graph()

    # The initial state — every field must be present (TypedDict requirement)
    # Downstream nodes will overwrite these defaults as they run
    initial_state: AgentState = {
        "user_message":   user_message,
        "intent":         "",
        "urgency":        "",
        "sentiment":      "",
        "confidence":     0.0,
        "tool_needed":    False,
        "route":          "",
        "tool_called":    None,
        "tool_result":    None,
        "rag_chunks":     None,
        "final_response": "",
    }

    # invoke() runs the full graph:
    # START → classify_node → route_message → [action_node] → END
    # Returns the FINAL state after all nodes have run
    final_state = app.invoke(initial_state)

    return final_state["final_response"]


# ============================================================
# SECTION 7: Interactive Loop
# ============================================================

def run_interactive():
    """
    Interactive chat loop — same UX as the manual agent loop.
    But now the graph handles all the routing.
    """
    print("\n" + "="*60)
    print("  LANGGRAPH AGENT — Skill 5")
    print("  (type 'exit' to quit)")
    print("="*60 + "\n")

    print("[INIT] Building graph...")
    app = build_graph()
    print("[OK] Graph compiled and ready.\n")

    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[AGENT] Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n[AGENT] Goodbye! Have a great day.")
            break

        turn += 1
        print(f"\n-- Turn {turn} ------------------------------------------")

        # Pause between turns (Gemini free tier rate limits)
        if turn > 1:
            time.sleep(8)

        try:
            response = run(user_input, app=app)
            print(f"\nAgent: {response}\n")
        except Exception as e:
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            print("Agent: I encountered an error. Please try again.\n")

    if conn:
        conn.close()


# ============================================================
# SECTION 8: Test Cases — 30 Scenarios (Skill 5 Eval Target: 85%+)
# ============================================================

test_cases = [
    # 1. order_lookup
    {"message": "Where is my order #7291?",                           "expected_route": "tool",     "expected_tool": "order_lookup",     "description": "Direct order status"},

    # 2. process_refund
    {"message": "I want a refund for order #3310. It arrived broken.",  "expected_route": "tool",   "expected_tool": "process_refund",   "description": "Refund - damaged"},

    # 3. escalate
    {"message": "I WANT TO SPEAK TO A MANAGER RIGHT NOW!!!",            "expected_route": "escalate", "expected_tool": "escalate_to_human", "description": "Angry - demands manager"},

    # 4. RAG / product questions
    {"message": "What is your return policy?",                          "expected_route": "rag",    "expected_tool": None,               "description": "Policy question"},

    # 5. Direct response
    {"message": "What are your business hours?",                       "expected_route": "direct", "expected_tool": None,               "description": "General - business hours"},
]


def run_eval():
    """Run all 30 test cases and print accuracy report."""
    print("\n" + "="*60)
    print("  LANGGRAPH AGENT — Skill 5 Evaluation")
    print("  Target: 85%+ route accuracy")
    print("="*60 + "\n")

    app = build_graph()

    route_correct = 0
    tool_correct  = 0
    total         = len(test_cases)
    tool_cases    = sum(1 for c in test_cases if c["expected_tool"] is not None)

    for i, case in enumerate(test_cases):
        if i > 0:
            print("  [Sleeping 10s to respect rate limits...]")
            time.sleep(10)

        print(f"\nTest {i+1}/{total}: \"{case['message'][:60]}\"")
        print(f"  Expected route: {case['expected_route']} | Expected tool: {case['expected_tool'] or 'None'}")

        try:
            # Run full graph, capture final state for inspection
            initial_state: AgentState = {
                "user_message":   case["message"],
                "intent":         "",
                "urgency":        "",
                "sentiment":      "",
                "confidence":     0.0,
                "tool_needed":    False,
                "route":          "",
                "tool_called":    None,
                "tool_result":    None,
                "rag_chunks":     None,
                "final_response": "",
            }

            final_state = app.invoke(initial_state)

            # Infer actual route from what got written to state
            actual_route = _infer_route(final_state)
            actual_tool  = final_state.get("tool_called")

            r_match = actual_route == case["expected_route"]
            t_match = (actual_tool == case["expected_tool"])

            if r_match:
                route_correct += 1
            if case["expected_tool"] is not None and t_match:
                tool_correct += 1

            r_status = "[PASS]" if r_match else "[FAIL]"
            t_status = "[PASS]" if t_match else "[FAIL]" if case["expected_tool"] else "  --  "

            print(f"  Route: {actual_route or '?':10s} {r_status}")
            print(f"  Tool:  {actual_tool or 'None':20s} {t_status}")
            print(f"  Response: {final_state['final_response'][:100]}...")

        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Route accuracy: {route_correct}/{total} ({route_correct/total*100:.0f}%) - Target: 85%+ ({int(total*0.85)}/{total})")
    print(f"  Tool accuracy:  {tool_correct}/{tool_cases} ({tool_correct/tool_cases*100:.0f}%) - of tool-calling cases")
    print(f"{'='*60}\n")


def _infer_route(final_state: dict) -> str:
    """
    Infer which route was taken from the final state.
    Since we don't have a dedicated 'route' field written by the router
    (router functions don't write state), we infer from what was written.
    """
    tool_called = final_state.get("tool_called")
    rag_chunks  = final_state.get("rag_chunks")

    if tool_called == "escalate_to_human":
        return "escalate"
    if tool_called in ("order_lookup", "process_refund"):
        return "tool"
    if tool_called is not None:
        return "tool"
    if rag_chunks is not None:
        return "rag"
    return "direct"


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill 5: LangGraph Agent")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation against 30 test cases instead of interactive mode"
    )
    args = parser.parse_args()

    if args.eval:
        run_eval()
    else:
        run_interactive()
