# Production AI Agent — Final Roadmap

> This is the final version. No more planning after this. Next step is building.

---

## The Mindset Before Any Skill

Before reaching for AI, always ask:

> **"Can I solve this without AI?"**

If `"refund" in message.lower()` routes 90% of tickets correctly — use that. It's cheaper, faster, and more reliable than an LLM call.

AI is for the problems that **can't** be solved with rules. Ambiguous language. Nuanced intent. Freeform knowledge retrieval. When you need AI, use it. When you don't, don't.

This mindset separates engineers who build things that work from engineers who build things that are impressive but fragile.

---

## The 7 Skills (Strict Order)

```
Skill 1 → Structured Outputs + Prompt Engineering
Skill 2 → Tool Calling
Skill 3 → RAG
Skill 4 → Manual Agent Loop (1 day only)
Skill 5 → LangGraph Orchestration
Skill 6 → Reliability + Memory
Skill 7 → Production Wrapping
```

---

### Skill 1: Structured Outputs + Prompt Engineering

**The foundation everything else stands on.**

Without this, every downstream skill is a gamble. Tool calling needs structured output to select the right function. RAG needs it to classify whether retrieval is needed. The agent needs it for routing decisions.

**What you learn:**
- System prompts that control LLM behavior
- Pydantic models as output schemas
- Gemini's `response_schema` to force valid JSON
- Few-shot prompting (showing examples in the prompt)
- Negative constraints ("never guess," "never fabricate")

**What "done" looks like:**
A ticket classifier. Give it any customer message → returns a validated Pydantic object with `intent`, `urgency`, `sentiment`, `requires_tool`. Works reliably across 10 test cases.

**Eval:** 10 test messages. Track classification accuracy. Target: 90%+.

---

### Skill 2: Tool Calling (Function Calling)

**Teaching the LLM to DO things, not just SAY things.**

**What you learn:**
- Defining tools as schemas the LLM can see
- The full loop: LLM selects tool → you execute → feed result back → LLM responds using result
- When to call NO tool (and how to prompt for that)
- Handling tool failures (telling the LLM "this failed, what now?")

**What "done" looks like:**
3 tools: `order_lookup`, `process_refund`, `escalate_to_human`. LLM picks the right one based on user message. For general questions, calls no tool. Tested against 15 cases.

**Eval:** 15 test messages across all 3 tools + "no tool needed" cases. Track: correct tool? correct arguments? Target: 85%+.

---

### Skill 3: RAG (Retrieval-Augmented Generation)

**Giving the agent knowledge it wasn't trained on.**

**What you learn:**
- Document chunking (why chunk size matters more than model choice)
- Embeddings (text → numbers for similarity search)
- Vector search with pgvector (cosine similarity)
- Context injection (how to put retrieved chunks into the prompt)
- The "I don't know" response (when retrieval finds nothing relevant)

**What "done" looks like:**
Upload a PDF → ask questions → get accurate answers grounded in the doc. Ask something NOT in the PDF → get "I don't have that information." Tested against 20 Q&A pairs.

**Eval:** 20 test Q&A pairs. Track two things separately: (1) retrieval accuracy — did we find the right chunks? (2) answer accuracy — did the LLM use them correctly? Target: 80%+ on both.

---

### Skill 4: Manual Agent Loop (1 Day Only)

**See what's under the hood before using the framework.**

This is NOT about building a production-quality manual agent. This is about spending one day building a raw version so when LangGraph breaks, you know how to debug the fundamentals: state, prompt, tool schema, LLM response.

**What you learn:**
- The core agent loop: `classify → decide → act → respond`
- How state flows through an agent (a dict that accumulates information)
- Why manual loops get ugly at 3+ branches (firsthand experience)
- What LangGraph abstracts away (and why you want that abstraction)

**What "done" looks like:**
A single Python script. While loop. Handles 3 intents: question (→ RAG), order status (→ tool), escalation. Messy but functional. You feel the pain of managing state manually.

**Then you move on.** Do not polish this. Do not refactor this. The mess IS the lesson.

---

### Skill 5: LangGraph Orchestration

**Wiring Skills 1-3 into a decision graph with proper state management.**

**What you learn:**
- Nodes (actions), edges (transitions), conditional routing
- `AgentState` TypedDict — what the agent knows at each step
- Conditional edges: "if intent is refund AND has order_id → tool node, else → RAG node"
- The classify → route → act → respond pattern as a graph
- Guardrails as routing decisions (low confidence → escalate node)

**What "done" looks like:**
A LangGraph agent that handles full customer support conversations. Classifies intent, routes to the right action, executes, checks confidence, responds or escalates. Compare it to your Skill 4 manual loop — same behavior, 1/4 the code, 10x more maintainable.

**Eval:** 30 test scenarios including edge cases (angry user, off-topic, jailbreak attempt, unknown question). Track: correct route taken? correct final response? graceful escalation when needed? Target: 85%+.

---

### Skill 6: Reliability + Memory

**Making it survive the real world.**

Your agent works on the happy path. Now break it.

**What you learn:**
- Retry logic (API timeouts, rate limits)
- Fallback responses (what to say when everything fails)
- Input validation (50,000-character messages, empty messages, injection attacks)
- Short-term memory (conversation history — agent remembers what was said 5 messages ago)
- Rate limiting (don't let one user burn your API budget)
- Structured logging (every agent decision logged as JSON for debugging)

**What "done" looks like:**
Kill your internet → agent responds gracefully. Send malicious prompts → politely refused. Have a 10-message conversation → context from message 1 is remembered. Send 100 requests/second → rate limited, no crash.

**Eval:** Failure injection tests. Chaos engineering for AI. Track: crash count (target: 0), graceful degradation rate (target: 100%).

---

### Skill 7: Production Wrapping

**Making it usable by real humans.**

Only now do you build the API, widget, and dashboard. Because now the core works, is tested, handles failures, and is reliable.

**What you learn:**
- FastAPI endpoint design (REST + WebSocket)
- Embeddable chat widget (vanilla JS, Shadow DOM)
- Admin dashboard (Jinja2 templates — no React/Next.js)
- Authentication (API keys for widget, JWT for dashboard)
- Deployment (Railway/Render — no Docker needed)

**What "done" looks like:**
A live URL. A chat widget embeddable on any site with one script tag. A dashboard where the business owner uploads docs and sees conversation logs. Deployed and accessible.

---

## Eval Is A Habit, Not A Step

| Skill | Eval You Do | Test Count |
|---|---|---|
| 1. Structured Outputs | Classification accuracy | 10 cases |
| 2. Tool Calling | Correct tool + correct args | 15 cases |
| 3. RAG | Retrieval accuracy + answer accuracy | 20 cases |
| 4. Manual Loop | Functional test | Quick sanity |
| 5. LangGraph Agent | End-to-end scenarios + edge cases | 30 cases |
| 6. Reliability | Failure injection | Break tests |
| 7. Production | Full flow via API | Integration tests |

**How**: A JSON file with test cases. A script that runs them. A printed score. No framework. No dashboard. Just `python run_eval.py` and a number.

---

## Tech Stack (Locked)

| What | Tool | Why |
|---|---|---|
| LLM | Gemini 2.5 Flash | Cheap, fast, strong structured outputs, generous free tier |
| Embeddings | Gemini `text-embedding-004` | Same SDK, 768 dimensions, asymmetric search support |
| Database | PostgreSQL + pgvector | One database for everything — relational data AND vectors |
| Agent Framework | LangGraph | State machines > chains. Supports routing, cycles, conditional logic |
| API | FastAPI | Async, typed, auto-docs, Python-native |
| Frontend | Vanilla JS (widget) + Jinja2 (dashboard) | No React. No Next.js. Keep it in Python's world. |
| Logging | structlog | Structured JSON logs. grep-debuggable. |
| Deploy | Railway or Render | Builds from GitHub. No Docker. |

**What's NOT here** (and why):
- No Redis (PostgreSQL handles sessions fine at your scale)
- No Docker (Railway builds from source)
- No LangChain (LangGraph works independently with `langchain-core`)
- No MCP (3 tools don't need a protocol — native function calling works)
- No multi-LLM (pick one, ship, switch if needed)

---

## What's Next

When you approve this plan, we deep dive into **Skill 1: Structured Outputs + Prompt Engineering**. We build a working ticket classifier. No theory. Just code that works.
