# Evaluation Results: LangGraph Agent

A verification run of the customer support agent was executed using the declarative LangGraph orchestration workflow.

## Summary Metrics

| Metric | Accuracy | Target | Status |
| :--- | :--- | :--- | :--- |
| **Route Selection Accuracy** | **100% (5/5)** | 85%+ | **[PASS]** |
| **Tool Selection Accuracy** | **100% (3/3)** | 85%+ | **[PASS]** |

---

## Detailed Test Scenarios

### Test 1: Order Status Lookup
* **Input message**: `"Where is my order #7291?"`
* **Classifier Decision**: Intent: `order_status` | Urgency: `high` | Requires Tool: `True`
* **Route taken**: `tool` (Expected: `tool`) — **[PASS]**
* **Tool called**: `order_lookup({'order_id': '7291'})` (Expected: `order_lookup`) — **[PASS]**
* **Final Agent Response**:
  > "Your order #7291 has been shipped and is estimated to be delivered by June 9, 2026. The tracking number is TRK-998877."

### Test 2: Refund Request
* **Input message**: `"I want a refund for order #3310. It arrived broken."`
* **Classifier Decision**: Intent: `refund_request` | Urgency: `high` | Requires Tool: `True`
* **Route taken**: `tool` (Expected: `tool`) — **[PASS]**
* **Tool called**: `process_refund({'reason': 'It arrived broken.', 'order_id': '3310'})` (Expected: `process_refund`) — **[PASS]**
* **Final Agent Response**:
  > "Your refund for order #3310 has been approved due to the item arriving broken. A refund of $49.99 will be processed by June 12, 2026."

### Test 3: Emergency Escalation
* **Input message**: `"I WANT TO SPEAK TO A MANAGER RIGHT NOW!!!"`
* **Classifier Decision**: Intent: `complaint` | Urgency: `critical` | Requires Tool: `False`
* **Route taken**: `escalate` (Expected: `escalate`) — **[PASS]**
* **Tool called**: `escalate_to_human` (Expected: `escalate_to_human`) — **[PASS]**
* **Final Agent Response**:
  > "I'm connecting you with a human agent right away. Your case ID is ESC-20260607-001. Estimated wait: 3 minutes."

### Test 4: RAG (Knowledge Base Search)
* **Input message**: `"What is your return policy?"`
* **Classifier Decision**: Intent: `product_question` | Urgency: `no_urgency` | Requires Tool: `True`
* **Route taken**: `rag` (Expected: `rag`) — **[PASS]**
* **Tool called**: `None` (Expected: `None`) — **[PASS]**
* **Final Agent Response**:
  > "I don't have access to the knowledge base right now. Let me connect you with a team member who can help." *(Graceful fallback since PostgreSQL database service is offline)*

### Test 5: Direct LLM Response
* **Input message**: `"What are your business hours?"`
* **Classifier Decision**: Intent: `general` | Urgency: `no_urgency` | Requires Tool: `False`
* **Route taken**: `direct` (Expected: `direct`) — **[PASS]**
* **Tool called**: `None` (Expected: `None`) — **[PASS]**
* **Final Agent Response**:
  > "Hello! We're open Monday through Friday from 9 AM to 5 PM PST. We look forward to assisting you during those hours."
