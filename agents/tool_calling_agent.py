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


load_dotenv()  # Load GEMINI_API_KEY from .env file


# ============================================================
# SECTION 1: TOOL FUNCTIONS — The actual Python functions
# ============================================================
# These simulate real backend operations.
# In production, these would hit a database or API.

def order_lookup(order_id: str) -> dict:
    """Look up the status of a customer order.
    
    Args:
        order_id: The order ID to look up (e.g., "7291")
    
    Returns:
        Dict with order status information
    """
    # Simulated database of orders
    orders = {
        "7291": {
            "order_id": "7291",
            "status": "shipped",
            "tracking_number": "TRK-998877",
            "estimated_delivery": "2026-06-09",
            "items": ["Wireless Mouse", "USB-C Cable"],
        },
        "3310": {
            "order_id": "3310",
            "status": "delivered",
            "delivered_date": "2026-06-05",
            "items": ["Bluetooth Speaker"],
        },
        "5592": {
            "order_id": "5592",
            "status": "partially_shipped",
            "shipped_items": ["Laptop Stand"],
            "pending_items": ["Monitor Arm"],
            "estimated_delivery": "2026-06-10",
        },
        "1234": {
            "order_id": "1234",
            "status": "processing",
            "items": ["Mechanical Keyboard"],
            "estimated_ship_date": "2026-06-08",
        },
    }

    if order_id in orders:
        return orders[order_id]
    else:
        return {"error": f"Order #{order_id} not found in our system."}


def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund for a customer order.
    
    Args:
        order_id: The order ID to refund
        reason: The reason for the refund
    
    Returns:
        Dict with refund confirmation details
    """
    # Simulated refund processing
    known_orders = ["7291", "3310", "5592", "1234"]

    if order_id in known_orders:
        return {
            "refund_id": f"REF-{order_id}-001",
            "order_id": order_id,
            "status": "approved",
            "refund_amount": "$49.99",
            "estimated_refund_date": "2026-06-12",
            "reason": reason,
        }
    else:
        return {"error": f"Cannot process refund: Order #{order_id} not found."}


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
tools = types.Tool(
    function_declarations=[
        order_lookup_declaration,
        process_refund_declaration,
        escalate_to_human_declaration,
    ]
)


# ============================================================
# SECTION 3: SYSTEM PROMPT — YOUR work (kept as you wrote it)
# ============================================================

SYSTEM_PROMPT = """
You are a customer support agent. You can look up orders, process refunds, 
and escalate issues to human agents.

TOOL SELECTION RULES — When to use each tool:
- Use order_lookup when customer asks about order status AND provides an order number.
- Use process_refund when customer explicitly requests a refund AND provides an order number.
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
