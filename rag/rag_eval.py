"""
Skill 3 Eval — 20 Q&A test cases
=================================
Tracks two things separately (as specified in implementation_plan.md):
  (1) Retrieval accuracy  — did we find relevant chunks? (retrieval_failed = False)
  (2) Answer accuracy     — did the LLM answer correctly? (keyword check)

Run:
  python rag_eval.py

Prereq: Documents must already be indexed.
  python rag_pipeline.py --index
"""

import os
import time
import json
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from google import genai

from rag_pipeline import query, get_connection, setup_table

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ============================================================
# 20 EVAL TEST CASES
# Split across:
#   - return policy questions (8)
#   - shipping policy questions (7)
#   - out-of-scope questions (5) — retrieval should FAIL gracefully
# ============================================================

TEST_CASES = [
    # --- RETURN POLICY (8 cases) ---
    {
        "id": 1,
        "question": "How many days do I have to return an item?",
        "expected_keywords": ["30 days", "30"],
        "retrieval_should_succeed": True,
        "category": "return_policy",
    },
    {
        "id": 2,
        "question": "What is the refund policy for Black Friday sale items?",
        "expected_keywords": ["store credit", "cash refund"],
        "retrieval_should_succeed": True,
        "category": "return_policy",
    },
    {
        "id": 3,
        "question": "Can I get a cash refund for a Black Friday purchase?",
        "expected_keywords": ["store credit", "not available", "cash refund"],
        "retrieval_should_succeed": True,
        "category": "return_policy",
    },
    {
        "id": 4,
        "question": "What happens if I receive a damaged item?",
        "expected_keywords": ["full refund", "replacement", "48 hours", "photo"],
        "retrieval_should_succeed": True,
        "category": "return_policy",
    },
    {
        "id": 5,
        "question": "Can I return earbuds I already opened?",
        "expected_keywords": ["cannot be returned", "non-returnable", "hygiene"],
        "retrieval_should_succeed": True,
        "category": "return_policy",
    },
    {
        "id": 6,
        "question": "How long does it take to receive my refund after returning an item?",
        "expected_keywords": ["5", "7", "business days"],
        "retrieval_should_succeed": True,
        "category": "return_policy",
    },
    {
        "id": 7,
        "question": "What is the return policy for items purchased with a gift card?",
        "expected_keywords": ["store credit", "gift card"],
        "retrieval_should_succeed": True,
        "category": "return_policy",
    },
    {
        "id": 8,
        "question": "I live in Germany. How long do I have to initiate a return?",
        "expected_keywords": ["14 days", "international"],
        "retrieval_should_succeed": True,
        "category": "return_policy",
    },

    # --- SHIPPING POLICY (7 cases) ---
    {
        "id": 9,
        "question": "What are your shipping options and their costs?",
        "expected_keywords": ["standard", "expedited", "overnight", "free"],
        "retrieval_should_succeed": True,
        "category": "shipping_policy",
    },
    {
        "id": 10,
        "question": "How long does standard shipping take?",
        "expected_keywords": ["5", "7", "business days"],
        "retrieval_should_succeed": True,
        "category": "shipping_policy",
    },
    {
        "id": 11,
        "question": "Do you ship to Canada? How much does it cost?",
        "expected_keywords": ["Canada", "$14.99", "14.99"],
        "retrieval_should_succeed": True,
        "category": "shipping_policy",
    },
    {
        "id": 12,
        "question": "My package tracking hasn't updated in a week. What should I do?",
        "expected_keywords": ["lost", "investigation", "support", "5"],
        "retrieval_should_succeed": True,
        "category": "shipping_policy",
    },
    {
        "id": 13,
        "question": "What are your customer support business hours?",
        "expected_keywords": ["Monday", "Friday", "9", "EST", "Saturday"],
        "retrieval_should_succeed": True,
        "category": "shipping_policy",
    },
    {
        "id": 14,
        "question": "Can I choose which carrier delivers my package?",
        "expected_keywords": ["cannot", "reserves the right", "choose"],
        "retrieval_should_succeed": True,
        "category": "shipping_policy",
    },
    {
        "id": 15,
        "question": "What happens if my package was delivered but I can't find it?",
        "expected_keywords": ["delivered", "stolen", "insurance", "courtesy"],
        "retrieval_should_succeed": True,
        "category": "shipping_policy",
    },

    # --- OUT OF SCOPE (5 cases — retrieval should FAIL gracefully) ---
    {
        "id": 16,
        "question": "What is your company's revenue for 2025?",
        "expected_keywords": ["don't have", "knowledge base", "information"],
        "retrieval_should_succeed": False,
        "category": "out_of_scope",
    },
    {
        "id": 17,
        "question": "What programming language is your backend written in?",
        "expected_keywords": ["don't have", "knowledge base", "information"],
        "retrieval_should_succeed": False,
        "category": "out_of_scope",
    },
    {
        "id": 18,
        "question": "Who is the CEO of Acme Corp?",
        "expected_keywords": ["don't have", "knowledge base", "information"],
        "retrieval_should_succeed": False,
        "category": "out_of_scope",
    },
    {
        "id": 19,
        "question": "What is the weather like in New York today?",
        "expected_keywords": ["don't have", "knowledge base", "information"],
        "retrieval_should_succeed": False,
        "category": "out_of_scope",
    },
    {
        "id": 20,
        "question": "Do you offer loyalty reward points?",
        "expected_keywords": ["don't have", "knowledge base", "information"],
        "retrieval_should_succeed": False,
        "category": "out_of_scope",
    },
]


# ============================================================
# ANSWER ACCURACY CHECK
# Keyword-based: answer must contain at least 1 expected keyword
# This is a pragmatic check — not perfect, but fast and auditable
# ============================================================

def check_answer_accuracy(answer: str, expected_keywords: list[str]) -> bool:
    """Return True if the answer contains at least one expected keyword (case-insensitive)."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


# ============================================================
# EVAL RUNNER
# ============================================================

def run_eval():
    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        conn = get_connection()
    except Exception as e:
        print(f"[ERROR] Cannot connect to DB: {e}")
        return

    setup_table(conn)

    total = len(TEST_CASES)
    retrieval_correct = 0
    answer_correct = 0
    results = []

    print(f"\n{'='*60}")
    print(f"RAG EVAL — {total} test cases")
    print(f"{'='*60}")

    for i, case in enumerate(TEST_CASES):
        # Rate limit — embedding + generation = 2 API calls per test
        if i > 0:
            time.sleep(3)

        print(f"\nTest {case['id']:02d}/{total} [{case['category']}]")
        print(f"  Q: {case['question']}")

        try:
            rag_result = query(client, conn, case["question"], verbose=False)

            # --- Retrieval accuracy ---
            retrieval_succeeded = not rag_result.retrieval_failed
            retrieval_correct_for_case = (retrieval_succeeded == case["retrieval_should_succeed"])
            if retrieval_correct_for_case:
                retrieval_correct += 1
            retrieval_status = "[PASS]" if retrieval_correct_for_case else "[FAIL]"

            # --- Answer accuracy ---
            answer_ok = check_answer_accuracy(rag_result.answer, case["expected_keywords"])
            if answer_ok:
                answer_correct += 1
            answer_status = "[PASS]" if answer_ok else "[FAIL]"

            print(f"  Retrieval: {retrieval_status}  (expected_to_succeed={case['retrieval_should_succeed']}, actually_succeeded={retrieval_succeeded})")
            print(f"  Answer:    {answer_status}  (keywords: {case['expected_keywords']})")
            print(f"  Response:  {rag_result.answer[:150]}{'...' if len(rag_result.answer) > 150 else ''}")

            results.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "retrieval_pass": retrieval_correct_for_case,
                "answer_pass": answer_ok,
                "answer": rag_result.answer,
                "sources": rag_result.sources,
                "confidence": rag_result.confidence,
                "retrieval_failed": rag_result.retrieval_failed,
            })

        except Exception as e:
            print(f"  [ERROR]: {e}")
            results.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "retrieval_pass": False,
                "answer_pass": False,
                "error": str(e),
            })

    # ============================================================
    # FINAL REPORT
    # ============================================================

    print(f"\n{'='*60}")
    print(f"EVAL RESULTS")
    print(f"{'='*60}")
    print(f"Retrieval Accuracy: {retrieval_correct}/{total} ({retrieval_correct/total*100:.0f}%)  — Target: 80%+")
    print(f"Answer Accuracy:    {answer_correct}/{total} ({answer_correct/total*100:.0f}%)  — Target: 80%+")

    # Break down by category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"retrieval": 0, "answer": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.get("retrieval_pass"):
            categories[cat]["retrieval"] += 1
        if r.get("answer_pass"):
            categories[cat]["answer"] += 1

    print(f"\nBreakdown by category:")
    for cat, counts in categories.items():
        t = counts["total"]
        print(f"  {cat}: retrieval={counts['retrieval']}/{t}, answer={counts['answer']}/{t}")

    # Failures worth investigating
    failures = [r for r in results if not r.get("retrieval_pass") or not r.get("answer_pass")]
    if failures:
        print(f"\nFailed cases ({len(failures)}):")
        for f in failures:
            r_status = "OK" if f.get("retrieval_pass") else "RETRIEVAL_FAIL"
            a_status = "OK" if f.get("answer_pass") else "ANSWER_FAIL"
            print(f"  [{f['id']:02d}] {r_status} | {a_status} — {f['question'][:70]}")

    # Save full results to JSON for analysis
    results_path = "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {results_path}")

    conn.close()


if __name__ == "__main__":
    run_eval()
