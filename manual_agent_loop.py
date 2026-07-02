"""
Skill 4: Manual Agent Loop (1 Day Only)
========================================

This is NOT production code. Do NOT refactor this. Do NOT polish this.
The mess IS the lesson.

What you'll feel by the end:
  - State management by hand is painful
  - Routing logic becomes a jungle of if/elif
  - History management is easy to get wrong
  - You'll want LangGraph badly

The agent loop:
  1. Get user message
  2. Classify intent (Skill 1 — ticket_classifier)
  3. Route based on intent:
     - order_status / refund_request → Tool Calling (Skill 2)
     - product_question / complaint   → RAG (Skill 3)
     - complaint (critical urgency)   → Escalate (Skill 2 tool)
     - general / feedback / other     → Direct LLM response
  4. Show response
  5. Repeat until user types "exit"

State is a plain Python dict. It grows. It gets messy. Good.

Usage:
  python manual_agent_loop.py               # Interactive mode (with DB for RAG)
  python manual_agent_loop.py --no-rag      # Interactive mode, skip RAG (no DB needed)
"""

import os
import json
import time
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Import our Skills ───────────────────────────────────────────────────────
# Skill 1: Classifier
from ticket_classifier import create_classifier, classify_ticket, TicketClassification

# Skill 2: Tool calling agent
from tool_calling_agent import create_agent, run_agent

# Skill 3: RAG pipeline (optional — needs PostgreSQL)
try:
    import psycopg2
    from rag_pipeline import get_connection, setup_table, query as rag_query, embed_query, retrieve_chunks, build_rag_prompt, generate_answer, RAG_SYSTEM_PROMPT
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────
DB_URL           = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5432/customer_support")
GENERATION_MODEL = "gemini-2.5-flash"

# ── DIRECT RESPONSE SYSTEM PROMPT ───────────────────────────────────────────
# Used for intents that don't need tools or RAG (greetings, feedback, etc.)
DIRECT_RESPONSE_PROMPT = """You are a friendly customer support agent for Acme Corp.
The customer has sent a general message that doesn't require looking up orders, processing refunds, or searching a knowledge base.

Respond naturally, warmly, and concisely. Keep it under 3 sentences.

NEVER fabricate order details or policy specifics.
NEVER claim to do something you cannot do (like look up an account without tools).
"""


# ============================================================
# THE MANUAL STATE DICT
# This is what gets ugly. It grows with every message.
# In LangGraph, this would be a typed AgentState. Here it's just... a dict.
# ============================================================

def make_initial_state() -> dict:
    """Create the starting state for a new conversation."""
    return {
        "conversation_id": f"conv_{int(time.time())}",
        "turn_count": 0,
        "history": [],          # list of {"role": "user"|"agent", "message": str}
        "last_intent": None,
        "last_urgency": None,
        "last_sentiment": None,
        "last_tool_called": None,
        "last_tool_result": None,
        "last_rag_chunks": None,
        "last_confidence": None,
        "errors": [],           # accumulates errors — in production you'd alert on this
    }


# ============================================================
# STEP 1: CLASSIFY
# ============================================================

def step_classify(client, state: dict, user_message: str) -> TicketClassification:
    """Call Skill 1 — classify the user message."""
    print(f"\n  [CLASSIFY] Classifying message...")
    result = classify_ticket(client, user_message)
    
    # Store in state (this is how state gets messy — just dump it all in)
    state["last_intent"] = result.intent.value
    state["last_urgency"] = result.urgency.value
    state["last_sentiment"] = result.sentiment.value
    state["last_confidence"] = result.confidence
    
    print(f"  [CLASSIFY] intent={result.intent.value} | urgency={result.urgency.value} | confidence={result.confidence:.2f}")
    return result


# ============================================================
# STEP 2: ROUTE — the ugly if/elif tree
# ============================================================

def step_route(classification: TicketClassification, use_rag: bool) -> str:
    """
    Decide what to do with this message.
    
    Returns one of:
      "tool"      → use Skill 2 (order lookup / refund / escalation)
      "rag"       → use Skill 3 (search knowledge base)
      "escalate"  → call escalate_to_human tool directly (critical)
      "direct"    → respond without tools or RAG
    """
    intent   = classification.intent.value
    urgency  = classification.urgency.value
    confidence = classification.confidence

    # Critical urgency on any complaint → escalate immediately
    if urgency == "critical":
        return "escalate"

    # Low confidence → escalate (agent isn't sure what's happening)
    if confidence < 0.60:
        return "escalate"

    # Order / refund → use tools
    if intent in ("order_status", "refund_request"):
        return "tool"

    # Product questions and complaints → RAG (if available)
    if intent in ("product_question", "complaint"):
        if use_rag:
            return "rag"
        else:
            return "direct"   # fallback if no DB

    # Everything else → direct response
    return "direct"


# ============================================================
# STEP 3A: ACT — Tool Calling (Skill 2)
# ============================================================

def step_tool(tool_agent_client, state: dict, user_message: str) -> str:
    """Call Skill 2 — the tool calling agent."""
    print(f"  [ACTION] Routing to TOOL CALLING agent...")
    
    result = run_agent(tool_agent_client, user_message)
    
    # Dump into state
    state["last_tool_called"] = result["tool_called"]
    state["last_tool_result"] = result["tool_result"]
    
    if result["tool_called"]:
        print(f"  [TOOL] Called: {result['tool_called']}")
    else:
        print(f"  [TOOL] No tool called — direct response")
    
    return result["final_response"]


# ============================================================
# STEP 3B: ACT — RAG (Skill 3)
# ============================================================

def step_rag(genai_client, conn, state: dict, user_message: str) -> str:
    """Call Skill 3 — the RAG pipeline."""
    print(f"  [ACTION] Routing to RAG pipeline...")
    
    rag_result = rag_query(genai_client, conn, user_message, verbose=True)
    
    # Dump into state
    state["last_rag_chunks"] = rag_result.sources
    
    if rag_result.retrieval_failed:
        print(f"  [RAG] Retrieval failed — no relevant chunks found")
    else:
        print(f"  [RAG] Retrieved from: {rag_result.sources}")
    
    return rag_result.answer


# ============================================================
# STEP 3C: ACT — Escalate
# ============================================================

def step_escalate(tool_agent_client, state: dict, user_message: str, urgency: str) -> str:
    """Force-call the escalation tool via the Skill 2 agent."""
    print(f"  [ACTION] ESCALATING (urgency={urgency})...")
    
    # Override the message to force escalation behavior
    # This is a hack. In LangGraph, you'd have a dedicated escalation node.
    escalation_message = f"I need to escalate this conversation to a human agent immediately. The customer said: '{user_message}'. Urgency: {urgency}."
    
    result = run_agent(tool_agent_client, escalation_message)
    state["last_tool_called"] = "escalate_to_human"
    state["last_tool_result"] = result["tool_result"]
    
    return result["final_response"]


# ============================================================
# STEP 3D: ACT — Direct LLM Response
# ============================================================

def step_direct(genai_client, state: dict, user_message: str) -> str:
    """Respond directly without tools or RAG. For greetings, thanks, etc."""
    print(f"  [ACTION] Direct LLM response (no tools, no RAG)...")
    
    response = genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=DIRECT_RESPONSE_PROMPT,
            temperature=0.7,
        ),
    )
    return response.text


# ============================================================
# THE MAIN AGENT LOOP
# ============================================================

def run_agent_loop(use_rag: bool = True):
    """
    The manual agent loop. This is the whole point of Skill 4.
    Feel how messy this is. That pain is the lesson.
    """
    print("\n" + "="*60)
    print("  MANUAL AGENT LOOP — Skill 4")
    print("  (type 'exit' to quit, 'state' to inspect current state)")
    print("="*60 + "\n")

    # ── Set up all the clients ───────────────────────────────
    print("[INIT] Setting up clients...")
    
    # Skill 1 client
    classifier_client = create_classifier()
    print("  [OK] Ticket classifier ready (Skill 1)")
    
    # Skill 2 client
    tool_client = create_agent()
    print("  [OK] Tool calling agent ready (Skill 2)")
    
    # Skill 3 setup (optional)
    conn = None
    genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    if use_rag and RAG_AVAILABLE:
        try:
            conn = get_connection()
            setup_table(conn)
            print("  [OK] RAG pipeline connected (Skill 3)")
        except Exception as e:
            print(f"  [X] RAG pipeline failed: {e}")
            print("    Falling back to direct responses for product questions.")
            use_rag = False
            conn = None
    else:
        if not RAG_AVAILABLE:
            print("  [X] RAG not available (psycopg2 or pgvector not installed)")
        else:
            print("  ─ RAG disabled (--no-rag flag)")
        use_rag = False

    # ── Initialize state ─────────────────────────────────────
    state = make_initial_state()
    print(f"\n[STATE] Conversation started: {state['conversation_id']}")
    print("[READY] Agent is ready. Start chatting!\n")

    # ── THE LOOP ─────────────────────────────────────────────
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[AGENT] Goodbye!")
            break

        # Special commands
        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n[AGENT] Goodbye! Have a great day.")
            break

        if user_input.lower() == "state":
            print("\n[DEBUG STATE]")
            print(json.dumps(state, indent=2, default=str))
            print()
            continue

        if user_input.lower() == "history":
            print("\n[CONVERSATION HISTORY]")
            for entry in state["history"]:
                role = "You  " if entry["role"] == "user" else "Agent"
                print(f"  {role}: {entry['message'][:100]}...")
            print()
            continue

        # Update turn count
        state["turn_count"] += 1
        turn = state["turn_count"]
        print(f"\n── Turn {turn} ──────────────────────────────────────────")

        # Store user message in history
        state["history"].append({"role": "user", "message": user_input})

        # ─────────────────────────────────────────────────────
        # THE CORE LOOP: classify → route → act → respond
        # This is what LangGraph will replace with a clean graph.
        # Feel the pain.
        # ─────────────────────────────────────────────────────

        agent_response = None

        try:
            # STEP 1: Classify
            # NOTE: We're sleeping between classify + action to avoid rate limits.
            # In LangGraph you'd handle this at the edge level. Here: manual sleep.
            classification = step_classify(classifier_client, state, user_input)
            
            # Brief pause between Skill 1 (classify) and Skill 2/3 (action)
            # to avoid hitting Gemini free tier rate limits
            time.sleep(8)

            # STEP 2: Route
            route = step_route(classification, use_rag)
            print(f"  [ROUTE] → {route.upper()}")

            # STEP 3: Act
            if route == "tool":
                agent_response = step_tool(tool_client, state, user_input)

            elif route == "rag":
                agent_response = step_rag(genai_client, conn, state, user_input)

            elif route == "escalate":
                agent_response = step_escalate(
                    tool_client, state, user_input,
                    urgency=state["last_urgency"] or "high"
                )

            elif route == "direct":
                agent_response = step_direct(genai_client, state, user_input)

            else:
                # Should never happen, but manual code has no type safety
                agent_response = "I'm sorry, I couldn't process your request. Please try again."
                state["errors"].append(f"Turn {turn}: Unknown route '{route}'")

        except Exception as e:
            # In production, this logs and falls back gracefully.
            # Here, we just print it and let the user know.
            agent_response = f"I encountered an error processing your request. Please try again."
            state["errors"].append(f"Turn {turn}: {type(e).__name__}: {e}")
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        # STEP 4: Respond
        print(f"\nAgent: {agent_response}")

        # Store agent response in history
        state["history"].append({"role": "agent", "message": agent_response})

        # Show state summary after each turn (the mess accumulates)
        print(f"\n  [STATE SNAPSHOT] turn={state['turn_count']} | intent={state['last_intent']} | route_confidence={state['last_confidence']} | errors={len(state['errors'])}")
        print()

    # ── Cleanup ──────────────────────────────────────────────
    if conn:
        conn.close()
    
    # Final state dump — look at how much junk accumulated
    print("\n" + "="*60)
    print("FINAL STATE (this is why we need LangGraph)")
    print("="*60)
    print(f"  Turns completed:  {state['turn_count']}")
    print(f"  Errors logged:    {len(state['errors'])}")
    print(f"  History entries:  {len(state['history'])}")
    print(f"  Last intent:      {state['last_intent']}")
    print(f"  Last tool called: {state['last_tool_called']}")
    if state["errors"]:
        print(f"\n  Error log:")
        for err in state["errors"]:
            print(f"    - {err}")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill 4: Manual Agent Loop")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (no PostgreSQL needed). Product questions get direct LLM responses."
    )
    args = parser.parse_args()

    run_agent_loop(use_rag=not args.no_rag)
