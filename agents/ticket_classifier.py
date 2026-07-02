"""
Skill 1: Customer Support Ticket Classifier
Build this yourself. Fill in every section marked with TODO.
"""

from google import genai
from google.genai import types
from pydantic import BaseModel
from enum import Enum
from dotenv import load_dotenv
import json
import os
import time


load_dotenv()  # Load GEMINI_API_KEY from .env file


# ============================================================
# SECTION 1: ENUMS — Define the allowed values
# ============================================================

# TODO: Create Intent enum with 5 values
#       Think: what are the 5 types of customer messages?
class Intent(str, Enum):
    order_status = "order_status"
    refund_request = "refund_request"
    product_question = "product_question"
    complaint = "complaint"
    general = "general"
    feedback = "feedback"
    price_match = "price_match"
    other = "other"

    


# TODO: Create Urgency enum with 4 values
#       Think: low → critical, what levels make sense?
class Urgency(str, Enum):
    no_urgency = "no_urgency"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"
    


# TODO: Create Sentiment enum with 4 values
#       Think: what emotions can a customer have?
class Sentiment(str, Enum):
    highly_positive="highly_positive"
    positive="positive"
    neutral="neutral"
    negative="negative"
    highly_negative="highly_negative"
    


    

    # Replace this with your enum values


# ============================================================
# SECTION 2: PYDANTIC MODEL — Define the output shape
# ============================================================

# TODO: Add 6 fields to this model
#       - intent (which type?)
#       - urgency (which type?)
#       - sentiment (which type?)
#       - requires_tool (what type for yes/no?)
#       - confidence (what type for a number like 0.85?)
#       - reasoning (what type for text?)
class TicketClassification(BaseModel):
    intent: Intent
    urgency: Urgency
    sentiment: Sentiment
    requires_tool: bool
    confidence: float 
    reasoning: str
    



# ============================================================
# SECTION 3: SYSTEM PROMPT — This is where YOUR thinking goes
# ============================================================

# TODO: Write the system prompt. Include:
#   1. Role — what is this model? (one sentence)
#   2. Rules — how should it classify each field?
#   3. Few-shot examples — at least 4 examples (input → output)
#   4. Negative constraints — at least 3 "NEVER do X" rules
#
# This is the hardest part. Take your time.

SYSTEM_PROMPT = """

Role: You are a highly trained expert to classify the customer support tickets.

Rules:
1. Always the data should be in json format.
2. Don't fabricate anything which you are not aware of.
3. Don't give any extra information 
5. when the confidence is less than 0.60 then call out the human agent
6. the confidence should be between 0 and 1
7. the reasoning should be in the text format based on the rules and few-shot examples



Few-shot examples:

---
input: "My order #7291 was supposed to arrive yesterday. Where is it?"
output: {"intent": "order_status", "urgency": "high", "sentiment": "negative", "requires_tool": true, "confidence": 0.85, "reasoning": "Customer is asking about a late order. Needs order lookup tool."}

---
input: "Lol what kind of worst service is this i will buy somewhere else"
output: {"intent": "complaint", "urgency": "high", "sentiment": "highly_negative", "requires_tool": false, "confidence": 0.60, "reasoning": "Customer is complaining and threatening to leave. No tool needed — needs human response."}

---
input: "what the fuck you had done I asked to cancel one service you cancelled another i will move to court."
output: {"intent": "complaint", "urgency": "critical", "sentiment": "highly_negative", "requires_tool": false, "confidence": 0.92, "reasoning": "Extreme complaint with legal threat. Critical urgency. No tool needed — requires immediate human escalation."}

---
input: "thank you so much for your help"
output: {"intent": "general", "urgency": "no_urgency", "sentiment": "highly_positive", "requires_tool": false, "confidence": 0.95, "reasoning": "Customer is expressing gratitude. No action needed."}

---
input: "can you do a price match with another website?"
output: {"intent": "price_match", "urgency": "medium", "sentiment": "neutral", "requires_tool": true, "confidence": 0.88, "reasoning": "Customer is asking for a price match — requires checking competitor pricing."}

---
input: "asdfghjkl"
output: {"intent": "general", "urgency": "no_urgency", "sentiment": "neutral", "requires_tool": false, "confidence": 0.15, "reasoning": "Gibberish input. Cannot determine intent. Very low confidence."}
"""

# ============================================================
# SECTION 4: MODEL SETUP — (boilerplate, already done for you)
# ============================================================

def create_classifier():
    """Create and return the configured Gemini client."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return client


def classify_ticket(client, message: str) -> TicketClassification:
    """Classify a customer support message."""
    import time
    backoff = 15.0
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=TicketClassification,
                    temperature=0.2,
                ),
            )
            print(f"  [DEBUG Raw JSON]: {response.text.strip()}")
            data = json.loads(response.text)
            return TicketClassification(**data)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str or "demand" in err_str
            if (is_rate_limit or is_unavailable) and attempt < 4:
                wait_time = 45.0 if is_rate_limit else backoff
                print(f"\n  [WARNING] Classifier API call failed. Retrying in {wait_time}s... (Attempt {attempt + 1}/5)")
                time.sleep(wait_time)
                if not is_rate_limit:
                    backoff *= 2.0
            else:
                raise e


# ============================================================
# SECTION 5: TEST CASES — (already done for you)
# ============================================================

test_cases = [
    {
        "message": "My order #7291 was supposed to arrive yesterday. Where is it?",
        "expected_intent": "order_status",
        "expected_requires_tool": True,
    },
    {
        "message": "I'd like a full refund for order #3310. The item arrived broken.",
        "expected_intent": "refund_request",
        "expected_requires_tool": True,
    },
    {
        "message": "Does the wireless mouse come in black?",
        "expected_intent": "product_question",
        "expected_requires_tool": False,
    },
    {
        "message": "Your website is so slow it's unusable. Fix it.",
        "expected_intent": "complaint",
        "expected_requires_tool": False,
    },
    {
        "message": "Thanks! Got my package today, love it!",
        "expected_intent": "general",
        "expected_requires_tool": False,
    },
    {
        "message": "I WANT TO SPEAK TO YOUR MANAGER RIGHT NOW. THIS IS UNACCEPTABLE.",
        "expected_intent": "complaint",
        "expected_requires_tool": False,
    },
    {
        "message": "Can I exchange my medium shirt for a large?",
        "expected_intent": "refund_request",
        "expected_requires_tool": True,
    },
    {
        "message": "What's your return policy?",
        "expected_intent": "product_question",
        "expected_requires_tool": False,
    },
    {
        "message": "asdfghjkl",
        "expected_intent": "general",
        "expected_requires_tool": False,
    },
    {
        "message": "I ordered two items but only one arrived. Order #5592. I need the other one ASAP.",
        "expected_intent": "order_status",
        "expected_requires_tool": True,
    },
]


# ============================================================
# SECTION 6: TEST RUNNER — (already done for you)
# ============================================================

if __name__ == "__main__":
    client = create_classifier()

    intent_correct = 0
    tool_correct = 0
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        if i > 0:
            print("  Sleeping 12 seconds to prevent Gemini free tier rate limit (5 RPM)...")
            time.sleep(12)
        print(f"\nTest {i+1}/{total}: \"{case['message'][:60]}...\"")

        try:
            result = classify_ticket(client, case["message"])

            intent_match = result.intent.value == case["expected_intent"]
            tool_match = result.requires_tool == case["expected_requires_tool"]

            if intent_match:
                intent_correct += 1
            if tool_match:
                tool_correct += 1

            print(f"  Intent:      {result.intent.value} {'[PASS]' if intent_match else '[FAIL]'} (expected: {case['expected_intent']})")
            print(f"  Tool:        {result.requires_tool} {'[PASS]' if tool_match else '[FAIL]'} (expected: {case['expected_requires_tool']})")
            print(f"  Urgency:     {result.urgency.value}")
            print(f"  Sentiment:   {result.sentiment.value}")
            print(f"  Confidence:  {result.confidence}")
            print(f"  Reasoning:   {result.reasoning}")

        except Exception as e:
            print(f"  [ERROR]: {e}")

    print(f"\n{'='*40}")
    print(f"RESULTS")
    print(f"{'='*40}")
    print(f"Intent Accuracy:  {intent_correct}/{total} ({intent_correct/total*100:.0f}%)")
    print(f"Tool Accuracy:    {tool_correct}/{total} ({tool_correct/total*100:.0f}%)")
    print(f"Overall:          {intent_correct+tool_correct}/{total*2} ({(intent_correct+tool_correct)/(total*2)*100:.0f}%)")
