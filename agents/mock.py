"""Offline mock mode for development and testing.

Enable with ``MOCK_MODE=true`` in ``.env`` (or the environment). When enabled,
every LLM boundary is replaced by fast, deterministic keyword rules, so the whole
system runs without the Gemini API — no cost, no rate limits, no network.

Entry points that respect ``MOCK_MODE``:
  - ``ReliableAgent.chat()``                 (main agent; used by the future API)
  - ``ticket_classifier.classify_ticket()``  (Skill 1)
  - ``tool_calling_agent.run_agent()``       (Skill 2)
  - ``rag.rag_pipeline.query()``             (Skill 3)
"""

from __future__ import annotations

import os
import re

# ── Keyword groups (mirror the real classifier's intent space) ───────────────
_REFUND_WORDS = ("refund", "money back", "return", "exchange", "reimburse")
_ORDER_WORDS = ("order", "tracking", "where is", "shipped", "delivery", "package", "arrive")
_ESCALATE_WORDS = ("manager", "human", "speak to", "lawsuit", "court", "worst", "unacceptable", "never again")
_POLICY_WORDS = ("policy", "shipping", "business hours", "hours", "canada", "cost", "price")


def is_mock_mode() -> bool:
    """True when the MOCK_MODE environment variable is enabled.

    Loads ``.env`` so the flag works whether the process was started with an
    environment variable or only from the ``.env`` file. Idempotent.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.getenv("MOCK_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _find_order_id(message: str):
    """Extract the first 3+ digit order id from a message, if present."""
    match = re.search(r'#?\s*(\d{3,})', message or "")
    return match.group(1) if match else None


def mock_classify_ticket(message: str):
    """Return a TicketClassification using simple keyword rules (no LLM call)."""
    from agents.ticket_classifier import Intent, Sentiment, TicketClassification, Urgency

    text = (message or "").lower()

    if any(w in text for w in _ESCALATE_WORDS):
        return TicketClassification(
            intent=Intent.complaint,
            urgency=Urgency.critical,
            sentiment=Sentiment.highly_negative,
            requires_tool=False,
            confidence=0.9,
            reasoning="Mock: escalation keywords detected.",
        )
    if any(w in text for w in _POLICY_WORDS):
        return TicketClassification(
            intent=Intent.product_question,
            urgency=Urgency.low,
            sentiment=Sentiment.neutral,
            requires_tool=False,
            confidence=0.9,
            reasoning="Mock: policy keywords detected.",
        )
    if any(w in text for w in _REFUND_WORDS):
        return TicketClassification(
            intent=Intent.refund_request,
            urgency=Urgency.high,
            sentiment=Sentiment.negative,
            requires_tool=True,
            confidence=0.9,
            reasoning="Mock: refund keywords detected.",
        )
    if any(w in text for w in _ORDER_WORDS):
        return TicketClassification(
            intent=Intent.order_status,
            urgency=Urgency.medium,
            sentiment=Sentiment.neutral,
            requires_tool=True,
            confidence=0.9,
            reasoning="Mock: order keywords detected.",
        )
    return TicketClassification(
        intent=Intent.general,
        urgency=Urgency.no_urgency,
        sentiment=Sentiment.neutral,
        requires_tool=False,
        confidence=0.9,
        reasoning="Mock: no keywords matched — general message.",
    )


def mock_run_agent(message: str) -> dict:
    """Return a run_agent()-shaped result using keyword rules (no LLM call)."""
    text = (message or "").lower()
    order_id = _find_order_id(message) or "7291"

    if any(w in text for w in _ESCALATE_WORDS):
        return {
            "tool_called": "escalate_to_human",
            "tool_args": {"reason": message, "urgency": "critical"},
            "tool_result": {
                "escalation_id": "ESC-MOCK-001",
                "status": "queued",
                "urgency": "critical",
                "reason": message,
                "estimated_wait_time": "3 minutes",
            },
            "final_response": "I'm connecting you with a human agent right away. Your case ID is ESC-MOCK-001.",
        }
    if any(w in text for w in _POLICY_WORDS):
        return {
            "tool_called": None,
            "tool_args": None,
            "tool_result": None,
            "final_response": "That's a policy question — I'd answer it directly without calling a tool (mock mode).",
        }
    if any(w in text for w in _REFUND_WORDS):
        return {
            "tool_called": "process_refund",
            "tool_args": {"order_id": order_id, "reason": message},
            "tool_result": {
                "refund_id": f"REF-{order_id}-MOCK",
                "order_id": order_id,
                "status": "approved",
                "refund_amount": "$49.99",
            },
            "final_response": f"Your refund for order #{order_id} has been approved (mock mode).",
        }
    if any(w in text for w in _ORDER_WORDS):
        return {
            "tool_called": "order_lookup",
            "tool_args": {"order_id": order_id},
            "tool_result": {
                "order_id": order_id,
                "status": "shipped",
                "tracking_number": "TRK-MOCK-001",
            },
            "final_response": f"Your order #{order_id} has shipped (mock mode).",
        }
    return {
        "tool_called": None,
        "tool_args": None,
        "tool_result": None,
        "final_response": "Hi! I'm running in mock mode. I can help with orders, refunds, and policies.",
    }


def mock_rag_query(question: str):
    """Return a RAGResponse with a canned grounded answer (no LLM/DB call)."""
    from rag.rag_pipeline import RAGResponse

    text = (question or "").lower()

    if "shipping" in text or "canada" in text or "ship" in text:
        return RAGResponse(
            answer="Standard shipping takes 5-7 business days. Orders over $50 ship free (mock mode).",
            sources=["shipping_policy.txt — chunk 1"],
            confidence=0.95,
            retrieval_failed=False,
        )
    if "return" in text or "refund" in text or "policy" in text:
        return RAGResponse(
            answer="Items can be returned within 30 days of purchase in original condition (mock mode).",
            sources=["return_policy.txt — chunk 1"],
            confidence=0.95,
            retrieval_failed=False,
        )
    return RAGResponse(
        answer="I don't have that information in my knowledge base.",
        sources=[],
        confidence=0.0,
        retrieval_failed=True,
    )


class MockApp:
    """Offline stand-in for the compiled LangGraph app (same ``invoke()`` contract).

    Mirrors the real graph's routing (critical/low-confidence → escalate,
    order/refund → tool, product/complaint → RAG, else → direct) using
    keyword rules instead of LLM calls.
    """

    def invoke(self, state: dict) -> dict:
        message = state.get("user_message", "")
        classification = mock_classify_ticket(message)
        intent = classification.intent.value
        urgency = classification.urgency.value
        confidence = classification.confidence

        if urgency == "critical" or confidence < 0.60:
            route = "escalate"
            result = mock_run_agent(message)
        elif intent in ("order_status", "refund_request"):
            route = "tool"
            result = mock_run_agent(message)
        elif intent in ("product_question", "complaint"):
            route = "rag"
            rag = mock_rag_query(message)
            result = {
                "tool_called": None,
                "tool_args": None,
                "tool_result": None,
                "final_response": rag.answer,
            }
        else:
            route = "direct"
            result = mock_run_agent(message)

        return {
            "user_message": message,
            "intent": intent,
            "urgency": urgency,
            "sentiment": classification.sentiment.value,
            "confidence": confidence,
            "tool_needed": classification.requires_tool,
            "route": route,
            "tool_called": result.get("tool_called"),
            "tool_result": result.get("tool_result"),
            "rag_chunks": [],
            "final_response": result["final_response"],
        }
