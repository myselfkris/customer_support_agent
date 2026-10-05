"""
Skill 2: Tool Calling Agent
Build this yourself. Fill in every section marked with TODO.

The goal: Given a customer message, the LLM should decide:
  1. WHICH tool to call (or no tool at all)
  2. WITH WHAT arguments
  3. Then use the tool's result to give a final response

You have 3 tools:
  - order_lookup(order_id) → returns order status info
  - process_refund(order_id, reason) → processes a refund
  - escalate_to_human(reason, urgency) → flags for human agent

For general questions, the LLM should call NO tool and respond directly.
"""

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time
import json
from datetime import datetime, timezone


load_dotenv()  # Load GEMINI_API_KEY from .env file

from db.base import SessionLocal, init_db
from db.models import Approval, Customer, Order, Refund, Ticket


# ============================================================
# SECTION 1: TOOL FUNCTIONS — The actual Python functions
# ============================================================
# These simulate real backend operations.
# In production, these would hit a database or API.

def order_lookup(order_id: str) -> dict:
    """Look up a customer order from the database.

    Args:
        order_id: The order ID to look up (e.g., "7291")

    Returns:
        Dict with order status information
    """
    init_db()
    session = SessionLocal()
    try:
        order = session.get(Order, order_id)
        if order is None:
            return {"error": f"Order #{order_id} not found in our system."}
        return {
            "order_id": order.id,
            "customer": order.customer.name,
            "status": order.status,
            "tracking_number": order.tracking_number,
            "items": [item.strip() for item in order.items.split(",") if item.strip()],
        }
    finally:
        session.close()


def process_refund(order_id: str, reason: str, requested_by: str = "system") -> dict:
    """Request a refund against an order (recorded as PENDING, never executed).

    Phase 3: money only moves after an authenticated human approves the request.
    This function is IDEMPOTENT — calling it twice for the same order returns the
    SAME pending request instead of creating duplicate refunds.

    Args:
        order_id: The order ID to refund
        reason: The reason for the refund
        requested_by: Who is asking (supplied from the API key in Phase 3)

    Returns:
        Dict with refund + approval details
    """
    init_db()
    session = SessionLocal()
    try:
        order = session.get(Order, order_id)
        if order is None:
            return {"error": f"Cannot process refund: Order #{order_id} not found."}

        # Idempotency guard: one open refund per order. If a pending or already
        # approved refund exists, return it instead of creating a duplicate.
        existing = (
            session.query(Refund)
            .filter(Refund.order_id == order.id, Refund.status.in_(["pending", "approved"]))
            .first()
        )
        if existing is not None:
            return {
                "refund_id": f"REF-{existing.id}",
                "order_id": order.id,
                "status": existing.status,
                "refund_amount": existing.amount,
                "reason": existing.reason,
                "already_requested": True,
            }

        refund = Refund(order_id=order.id, amount=order.amount, reason=reason, status="pending")
        session.add(refund)
        session.flush()  # assign refund.id so the approval can reference it

        approval = Approval(
            kind="refund",
            order_id=order.id,
            refund_id=refund.id,
            amount=order.amount,
            reason=reason,
            status="pending",
            requested_by=requested_by,
        )
        session.add(approval)
        session.commit()
        session.refresh(refund)
        return {
            "refund_id": f"REF-{refund.id}",
            "approval_id": approval.id,
            "order_id": order.id,
            "status": refund.status,
            "refund_amount": refund.amount,
            "reason": refund.reason,
            "already_requested": False,
        }
    finally:
        session.close()


def get_customer_details(customer_id: int) -> dict:
    """Look up a customer by their numeric ID from the database."""
    init_db()
    session = SessionLocal()
    try:
        customer = session.get(Customer, customer_id)
        if customer is None:
            return {"error": f"Customer #{customer_id} not found."}
        return {
            "customer_id": customer.id,
            "name": customer.name,
            "email": customer.email,
        }
    finally:
        session.close()


def create_support_ticket(customer_id: int, subject: str) -> dict:
    """Create a support ticket for a customer (recorded in the database)."""
    init_db()
    session = SessionLocal()
    try:
        customer = session.get(Customer, customer_id)
        if customer is None:
            return {"error": f"Customer #{customer_id} not found."}
        ticket = Ticket(customer_id=customer.id, subject=subject, status="open")
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return {
            "ticket_id": ticket.id,
            "customer_id": ticket.customer_id,
            "subject": ticket.subject,
            "status": ticket.status,
        }
    finally:
        session.close()


def cancel_order(order_id: str, requested_by: str = "system") -> dict:
    """Request an order cancellation (recorded as PENDING, never executed).

    Phase 3: the order moves to ``pending_cancel``; it only becomes ``cancelled``
    after a human approves the request. Idempotent — repeat calls return the
    current pending state instead of re-requesting.
    """
    init_db()
    session = SessionLocal()
    try:
        order = session.get(Order, order_id)
        if order is None:
            return {"error": f"Cannot cancel: Order #{order_id} not found."}

        # Idempotency guard: a cancellation already requested or done — just report it.
        if order.status in ("pending_cancel", "cancelled"):
            return {"order_id": order.id, "status": order.status, "already_requested": True}

        previous = order.status
        order.status = "pending_cancel"
        approval = Approval(
            kind="cancel",
            order_id=order.id,
            previous_status=previous,
            amount=0.0,
            reason="Customer requested order cancellation.",
            status="pending",
            requested_by=requested_by,
        )
        session.add(approval)
        session.commit()
        return {
            "order_id": order.id,
            "status": order.status,
            "approval_id": approval.id,
            "already_requested": False,
        }
    finally:
        session.close()


def escalate_to_human(reason: str, urgency: str) -> dict:
    """Escalate a conversation to a human agent.
    
    Args:
        reason: Why the conversation needs human attention
        urgency: How urgent — "low", "medium", "high", or "critical"
    
    Returns:
        Dict with escalation confirmation
    """
    return {
        "escalation_id": "ESC-20260607-001",
        "status": "queued",
        "urgency": urgency,
        "reason": reason,
        "estimated_wait_time": "3 minutes" if urgency in ["high", "critical"] else "10 minutes",
    }


def approve_approval(approval_id: int, approved_by: str) -> dict:
    """Approve a pending approval and apply its effect EXACTLY ONCE.

    Admin-only (Phase 3). This function is deliberately NOT registered in
    ``TOOL_FUNCTIONS`` or the Gemini ``tools`` list, so the customer-facing LLM
    can NEVER call it — the agent can request money, but it cannot approve it.
    """
    init_db()
    session = SessionLocal()
    try:
        approval = session.get(Approval, approval_id)
        if approval is None:
            return {"error": f"Approval #{approval_id} not found."}

        # Transition guard: only a pending approval can be decided.
        if approval.status != "pending":
            return {"error": f"Approval #{approval_id} already {approval.status}.", "status": approval.status}

        now = datetime.now(timezone.utc)
        if approval.kind == "refund":
            refund = session.get(Refund, approval.refund_id)
            if refund is None:
                return {"error": f"Refund for approval #{approval_id} not found."}
            refund.status = "approved"
            refund.approved_by = approved_by
            refund.approved_at = now
        elif approval.kind == "cancel":
            order = session.get(Order, approval.order_id)
            if order is not None:
                order.status = "cancelled"
        else:
            return {"error": f"Unknown approval kind: {approval.kind}"}

        approval.status = "approved"
        approval.approved_by = approved_by
        approval.resolved_at = now
        session.commit()
        return {
            "approval_id": approval.id,
            "kind": approval.kind,
            "status": approval.status,
            "order_id": approval.order_id,
        }
    finally:
        session.close()


def decline_approval(approval_id: int, declined_by: str) -> dict:
    """Decline a pending approval and revert/close the pending request."""
    init_db()
    session = SessionLocal()
    try:
        approval = session.get(Approval, approval_id)
        if approval is None:
            return {"error": f"Approval #{approval_id} not found."}
        if approval.status != "pending":
            return {"error": f"Approval #{approval_id} already {approval.status}.", "status": approval.status}

        now = datetime.now(timezone.utc)
        if approval.kind == "refund":
            refund = session.get(Refund, approval.refund_id)
            if refund is not None:
                refund.status = "declined"
        elif approval.kind == "cancel":
            order = session.get(Order, approval.order_id)
            if order is not None:
                # Restore the order to the status it had before the cancel request.
                order.status = approval.previous_status or "processing"
        else:
            return {"error": f"Unknown approval kind: {approval.kind}"}

        approval.status = "declined"
        approval.approved_by = declined_by
        approval.resolved_at = now
        session.commit()
        return {
            "approval_id": approval.id,
            "kind": approval.kind,
            "status": approval.status,
            "order_id": approval.order_id,
        }
    finally:
        session.close()


def list_pending_approvals() -> list[dict]:
    """List every approval still waiting for a human decision."""
    init_db()
    session = SessionLocal()
    try:
        approvals = session.query(Approval).filter(Approval.status == "pending").all()
        return [
            {
                "approval_id": a.id,
                "kind": a.kind,
                "order_id": a.order_id,
                "amount": a.amount,
                "reason": a.reason,
                "status": a.status,
                "requested_by": a.requested_by,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in approvals
        ]
    finally:
        session.close()


# ============================================================
# SECTION 2: TOOL DECLARATIONS — Tell the LLM what tools exist
# ============================================================
# 
# TODO: Define the tool declarations for the Gemini API.
#
# Each tool needs:
#   - A name (must match the Python function name)
#   - A description (the LLM reads this to decide WHEN to use it)
#   - Parameters with types and descriptions
#
# HINT: Use types.FunctionDeclaration and types.Tool
# 
# Think about:
#   - What makes a good tool description? (be specific about WHEN to use it)
#   - What parameters does each tool need?
#   - Which parameters are required vs optional?
#
# Reference: https://ai.google.dev/gemini-api/docs/function-calling

# Tool declarations — YOUR work (cleaned up)
order_lookup_declaration = types.FunctionDeclaration(
    name="order_lookup",
    description="Look up the status of a customer order. Use when the customer asks about order status, shipping, tracking, or delivery. Statuses can be: delivered, processing, partially_shipped, or shipped.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "order_id": types.Schema(
                type=types.Type.STRING,
                description="The numeric order ID provided by the customer, e.g. '7291'",
            ),
        },
        required=["order_id"],
    ),
)

process_refund_declaration = types.FunctionDeclaration(
    name="process_refund",
    description="Process a refund for a customer order. Use when the customer explicitly requests a refund, money back, or return AND provides an order number.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "order_id": types.Schema(
                type=types.Type.STRING,
                description="The order ID to refund",
            ),
            "reason": types.Schema(
                type=types.Type.STRING,
                description="The reason for the refund, extracted from the customer's message",
            ),
        },
        required=["order_id", "reason"],
    ),
)

escalate_to_human_declaration = types.FunctionDeclaration(
    name="escalate_to_human",
    description="Escalate the conversation to a human agent. Use when the customer is extremely upset, makes legal threats, explicitly asks for a human/manager, or has a complex issue that tools cannot handle.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "reason": types.Schema(
                type=types.Type.STRING,
                description="Why this conversation needs human attention",
            ),
            "urgency": types.Schema(
                type=types.Type.STRING,
                description="How urgent: 'low', 'medium', 'high', or 'critical'",
            ),
        },
        required=["reason", "urgency"],
    ),
)


# Bundle all tool declarations into a Tool object
get_customer_details_declaration = types.FunctionDeclaration(
    name="get_customer_details",
    description="Look up a customer by their numeric customer ID. Use when the customer asks about their account or personal details.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "customer_id": types.Schema(type=types.Type.INTEGER, description="The numeric customer ID"),
        },
        required=["customer_id"],
    ),
)

create_support_ticket_declaration = types.FunctionDeclaration(
    name="create_support_ticket",
    description="Create a support ticket for a customer. Use when an issue needs to be tracked or followed up.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "customer_id": types.Schema(type=types.Type.INTEGER, description="The numeric customer ID"),
            "subject": types.Schema(type=types.Type.STRING, description="Short description of the issue"),
        },
        required=["customer_id", "subject"],
    ),
)

cancel_order_declaration = types.FunctionDeclaration(
    name="cancel_order",
    description="Cancel an order. Use when the customer explicitly asks to cancel an order they placed.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "order_id": types.Schema(type=types.Type.STRING, description="The order ID to cancel"),
        },
        required=["order_id"],
    ),
)

tools = types.Tool(
    function_declarations=[
        order_lookup_declaration,
        process_refund_declaration,
        escalate_to_human_declaration,
        get_customer_details_declaration,
        create_support_ticket_declaration,
        cancel_order_declaration,
    ]
)


# ============================================================
# SECTION 3: SYSTEM PROMPT — YOUR work (kept as you wrote it)
# ============================================================

SYSTEM_PROMPT = """
You are a customer support agent. You can look up orders, process refunds,
look up customers, create support tickets, cancel orders, and escalate issues to human agents.

TOOL SELECTION RULES — When to use each tool:
- Use order_lookup when customer asks about order status AND provides an order number.
- Use process_refund when customer explicitly requests a refund AND provides an order number.
- Use get_customer_details when the customer asks about their account or personal details.
- Use create_support_ticket when an issue needs to be tracked or followed up.
- Use cancel_order when the customer explicitly asks to cancel an order.
- Use escalate_to_human when customer is extremely upset, makes legal threats, or asks for a human.
- Use escalate_to_human when customer has a mixture of queries which cannot be handled by tools.

NO-TOOL RULES — When NOT to use any tool:
- For general questions about policies, products, or business info, respond directly without calling any tool.
- For greetings, thanks, or social messages, respond naturally without calling any tool.

RESPONSE RULES:
- After receiving tool results, use the data to write a helpful, friendly response.
- If a tool returns an error, apologize and offer alternatives.
- Keep responses concise and professional.

NEGATIVE CONSTRAINTS:
- NEVER call order_lookup without a specific order number from the customer.
- NEVER call process_refund unless the customer explicitly asks for a refund or money back.
- NEVER call escalate_to_human just because a customer is slightly unhappy.
- NEVER fabricate order information or tracking numbers.
"""


# ============================================================
# SECTION 4: THE TOOL CALLING LOOP — The core of Skill 2
# ============================================================
#
# The flow:
#   1. Send user message to LLM (with tool declarations)
#   2. LLM responds with either:
#      a. A text response (no tool needed) → done
#      b. A function_call (tool needed) → go to step 3
#   3. Execute the function locally with the LLM's arguments
#   4. Send the function result back to the LLM
#   5. LLM generates a final text response using the result → done

def create_agent():
    """Create and return the configured Gemini client."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return client


# Map of tool names to actual Python functions
TOOL_FUNCTIONS = {
    "order_lookup": order_lookup,
    "process_refund": process_refund,
    "escalate_to_human": escalate_to_human,
    "get_customer_details": get_customer_details,
    "create_support_ticket": create_support_ticket,
    "cancel_order": cancel_order,
}


def generate_content_with_retry(client, model: str, contents, config, max_retries: int = 5, initial_backoff: float = 15.0):
    """Wrapper around client.models.generate_content to handle rate limits and transient errors."""
    import time
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str or "demand" in err_str
            
            if (is_rate_limit or is_unavailable) and attempt < max_retries - 1:
                # 429 rate limit errors benefit from a longer wait to clear the minute quota window
                wait_time = 45.0 if is_rate_limit else backoff
                print(f"\n  [WARNING] API call failed ({'429 Rate Limit' if is_rate_limit else '503 High Demand/Unavailable'}). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                if not is_rate_limit:
                    backoff *= 2.0
            else:
                raise e


def run_agent(client, user_message: str) -> dict:
    """
    Run the tool-calling agent for a single user message.
    
    Returns a dict with:
        - tool_called: str or None (name of tool called)
        - tool_args: dict or None (arguments passed to tool)
        - tool_result: dict or None (result from tool execution)
        - final_response: str (the agent's final text response)
    """
    from agents.mock import is_mock_mode, mock_run_agent
    if is_mock_mode():
        return mock_run_agent(user_message)

    MODEL = "gemini-2.5-flash"

    # Step 1: Send the user message to the LLM with tool declarations
    response = generate_content_with_retry(
        client=client,
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tools],
            temperature=0.2,
        ),
    )

    # Step 2: Check if the LLM wants to call a tool
    # Look through the response parts for a function_call
    function_call = None
    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_call = part.function_call
            break

    if function_call is None:
        # No tool called — LLM responded directly
        return {
            "tool_called": None,
            "tool_args": None,
            "tool_result": None,
            "final_response": response.text,
        }

    # Step 3: Execute the tool locally
    tool_name = function_call.name
    tool_args = dict(function_call.args) if function_call.args else {}

    print(f"  [TOOL CALL]: {tool_name}({tool_args})")

    # Look up and execute the actual Python function
    if tool_name in TOOL_FUNCTIONS:
        tool_result = TOOL_FUNCTIONS[tool_name](**tool_args)
    else:
        tool_result = {"error": f"Unknown tool: {tool_name}"}

    print(f"  [TOOL RESULT]: {json.dumps(tool_result, indent=2)}")

    # Step 4: Send the tool result back to the LLM
    # IMPORTANT: Use the ACTUAL model response content (not a manual reconstruction)
    # because thinking models include a thought_signature that must be preserved.

    # Create the function response part
    function_response_part = types.Part.from_function_response(
        name=tool_name,
        response=tool_result,
    )

    # Build the conversation history using the ACTUAL response from the model
    conversation = [
        # Turn 1: The user's original message
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        ),
        # Turn 2: The model's ACTUAL response (includes thought_signature)
        response.candidates[0].content,
        # Turn 3: The function result we're sending back
        types.Content(
            role="user",
            parts=[function_response_part],
        ),
    ]

    # Send the full conversation back so the LLM can write a final response
    final_response = generate_content_with_retry(
        client=client,
        model=MODEL,
        contents=conversation,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tools],
            temperature=0.2,
        ),
    )

    # Step 5: Return the results
    return {
        "tool_called": tool_name,
        "tool_args": tool_args,
        "tool_result": tool_result,
        "final_response": final_response.text,
    }


# ============================================================
# SECTION 5: TEST CASES — 15 cases across all scenarios
# ============================================================

test_cases = [
    # --- order_lookup cases (5) ---
    {
        "message": "Where is my order #7291? It should have arrived by now.",
        "expected_tool": "order_lookup",
        "description": "Direct order status inquiry with order ID",
    },
    {
        "message": "Can you check the status of order 5592?",
        "expected_tool": "order_lookup",
        "description": "Simple order status check",
    },
    {
        "message": "I ordered two items but only got one. Order #5592.",
        "expected_tool": "order_lookup",
        "description": "Partial delivery complaint with order ID",
    },
    {
        "message": "Has order #1234 shipped yet?",
        "expected_tool": "order_lookup",
        "description": "Shipping status check",
    },
    {
        "message": "I need tracking info for my order 3310.",
        "expected_tool": "order_lookup",
        "description": "Tracking information request",
    },

    # --- process_refund cases (4) ---
    {
        "message": "I want a refund for order #3310. The speaker arrived broken.",
        "expected_tool": "process_refund",
        "description": "Refund request with reason (damaged item)",
    },
    {
        "message": "Please refund order 7291, I changed my mind.",
        "expected_tool": "process_refund",
        "description": "Refund request — buyer's remorse",
    },
    {
        "message": "Order #1234 hasn't shipped and I don't want it anymore. Give me my money back.",
        "expected_tool": "process_refund",
        "description": "Refund for unshipped order",
    },
    {
        "message": "I received the wrong item in order 5592. I want a full refund immediately.",
        "expected_tool": "process_refund",
        "description": "Refund for wrong item — urgent tone",
    },

    # --- escalate_to_human cases (3) ---
    {
        "message": "I WANT TO SPEAK TO A MANAGER RIGHT NOW. THIS IS THE WORST SERVICE EVER.",
        "expected_tool": "escalate_to_human",
        "description": "Angry customer demanding manager",
    },
    {
        "message": "I've called 5 times about this issue and nobody has helped me. I need a real person.",
        "expected_tool": "escalate_to_human",
        "description": "Frustrated customer — repeated contact",
    },
    {
        "message": "Your AI is useless. Connect me to a human agent now.",
        "expected_tool": "escalate_to_human",
        "description": "Explicit request for human agent",
    },

    # --- no tool cases (3) ---
    {
        "message": "What are your business hours?",
        "expected_tool": None,
        "description": "General question — no tool needed",
    },
    {
        "message": "Do you ship to Canada?",
        "expected_tool": None,
        "description": "General question — no tool needed",
    },
    {
        "message": "Thanks for your help! Everything is great.",
        "expected_tool": None,
        "description": "Positive feedback — no tool needed",
    },
]


# ============================================================
# SECTION 6: TEST RUNNER
# ============================================================

if __name__ == "__main__":
    client = create_agent()

    correct = 0
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        if i > 0:
            print("  Sleeping 15 seconds (tool calling uses 2 API calls per test)...")
            time.sleep(15)

        print(f"\nTest {i+1}/{total}: \"{case['message'][:60]}...\"")
        print(f"  Expected tool: {case['expected_tool'] or 'None (direct response)'}")

        try:
            result = run_agent(client, case["message"])

            tool_match = result["tool_called"] == case["expected_tool"]
            if tool_match:
                correct += 1

            status = "[PASS]" if tool_match else "[FAIL]"
            print(f"  Tool called:   {result['tool_called'] or 'None'} {status}")
            if result["tool_args"]:
                print(f"  Tool args:     {result['tool_args']}")
            print(f"  Response:      {result['final_response'][:120]}...")

        except Exception as e:
            print(f"  [ERROR]: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Tool Selection Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"Target: 85%+ ({int(total * 0.85)}/{total})")
