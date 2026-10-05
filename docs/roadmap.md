# Customer Support Agent — Production Roadmap

> Goal: turn the existing Skills 1–6 learning codebase into a **production-grade Agentic AI
> Customer Support Application** that demonstrates genuine AI Application Engineering and
> Agentic AI Engineering ability in startup interviews.

**Principles**
- Keep the existing stack: Gemini, LangGraph, PostgreSQL + pgvector, Pydantic.
- Do not rewrite the working AI core. Add production engineering around it.
- Each phase leaves the project runnable and demoable.

---

## Phase 0 — Repo Hygiene & Packaging  (~0.5 day)  ✅ DONE

**Why:** The repo is not installable (`requirements.txt` is copy-pasted from another project)
and only runs from specific directories. This is the first thing an interviewer sees.

- [x] Fix `requirements.txt` to the real dependency set.
- [x] Add `pyproject.toml` (setuptools packaging, `agents` + `rag` + `app` packages).
- [x] Add `__init__.py` to `agents/`, `rag/`, and `app/`.
- [x] Replace fragile bare imports (`from ticket_classifier import ...`) with package imports
      (`from agents.ticket_classifier import ...`, `from rag.rag_pipeline import ...`).
- [x] Standardize invocation on `python -m agents.<script>` / `python -m rag.rag_pipeline`.
- [x] Update `README.md` run commands accordingly.

**Exit criteria:** `pip install -r requirements.txt` succeeds; `python -m agents.reliable_agent --eval`
runs; all modules import from repo root.

---

## Phase 1 — Application Shell: FastAPI  (1–2 days)  ✅ DONE

**Why:** An "application" must be callable over HTTP, not just a CLI script.

- [x] Create `app/` package with `app/main.py` (FastAPI app factory).
- [x] `POST /chat` — accepts `{user_id, message}`, returns structured `{user_id, response}`.
- [x] `GET /health` — health check (LLM key presence, mock-mode status, model config).
- [x] Pydantic request/response schemas in `app/schemas.py`.
- [x] Wire `ReliableAgent.chat()` as the handler (threaded; lock-guarded).
- [x] Central config via `pydantic-settings` (`app/config.py`).
- [x] Async-safe entry (`uvicorn app.main:app`).

**Exit criteria:** `curl -X POST localhost:8000/chat` returns a structured agent response;
`/docs` renders; no secrets in code.

> Note: the API respects `MOCK_MODE` — with it enabled, `/chat` runs offline with no API key.

---

## Phase 2 — Real Data & Relational Model  (2–3 days)

**Why:** All business tools currently read/write hardcoded Python dicts — that is demo, not product.

- [ ] SQLAlchemy models + Alembic migrations: `customers`, `orders`, `refunds`, `tickets`, `conversations`.
- [ ] Seed data; `order_lookup` / new tools read from Postgres.
- [ ] New tools: `get_customer_details`, `create_support_ticket`, `cancel_order`.
- [ ] Expose `search_knowledge_base` as a first-class callable tool.
- [ ] DB-backed conversation persistence (later folded into checkpointer in Phase 4).

**Exit criteria:** tool calls mutate/query real tables; migrations versioned; no hardcoded order dicts.

---

## Phase 3 — Safety & Human-in-the-Loop  (2–3 days)

**Why:** `process_refund` currently auto-approves money movement — the exact thing Agentic AI
roles are hired to prevent. Human approval is your flagship demo.

- [ ] LangGraph **checkpointer** (Postgres) so state can pause/resume.
- [ ] `interrupt()`-based approval node for `process_refund` / `cancel_order`
      (`PENDING_APPROVAL → approved/declined`), idempotent refund IDs + status lifecycle.
- [ ] `GET/POST /admin/approvals` — list and approve/decline pending actions.
- [ ] Authentication: API key for `/chat`, JWT for `/admin`; `user_id` comes from the token.

**Exit criteria:** a refund cannot execute without an authenticated human approval; no double refunds.

---

## Phase 4 — Agent Depth: Multi-step + Persistent Memory  (2–3 days)

**Why:** The agent is currently single-turn. Multi-step tool chaining and durable memory are the
definition of "agentic."

- [ ] Allow tool chaining (e.g., `order_lookup` → `process_refund`) in one turn.
- [ ] LangGraph cycles + checkpointing for multi-turn conversations.
- [ ] Replace `ReliableAgent._with_memory` text-prepend hack with structured conversation state
      fed only to response nodes (not the classifier).

**Exit criteria:** one turn chains multiple tools; conversation survives process restart; classifier
sees only the current message.

---

## Phase 5 — Observability & Guardrails  (2 days)

**Why:** The target explicitly requires logging, tracing, latency, token/cost tracking, and guardrails.

- [ ] `structlog` with trace IDs end-to-end.
- [ ] Per-LLM-call latency, token counts, and cost tracking.
- [ ] Tool-call tracing in structured logs (tool, args, result).
- [ ] Input/output guardrails: PII redaction, output policy check, hardened injection detection.

**Exit criteria:** every turn logs `trace_id, node, model, tokens, cost, latency, tool, tool_args`.

---

## Phase 6 — Evaluation & Testing  (2–3 days)

**Why:** Keyword-matching "evals" (and 5 cases masquerading as 30) cannot catch hallucination.

- [ ] `pytest` suite replacing print-based "evals" (unit + integration).
- [ ] Golden datasets (classification, tool selection, RAG Q&A, failure injection).
- [ ] Faithfulness / LLM-as-judge eval for RAG (does the answer use only retrieved chunks?).
- [ ] Expand LangGraph eval from 5 → 30+ cases.
- [ ] CI step (Phase 7) runs lint + typecheck + pytest + eval.

**Exit criteria:** `pytest` green; eval reports retrieval / answer / faithfulness separately.

---

## Phase 7 — Deployment  (1–2 days)

**Why:** "It runs on my laptop" is not production.

- [ ] `Dockerfile` (app) + `docker-compose.yml` (app + Postgres with pgvector).
- [ ] CI pipeline (lint, typecheck, pytest, eval).
- [ ] Deploy to Render/Railway with managed Postgres; `.env` via platform secrets.

**Exit criteria:** a live URL runs the full stack; CI is green on every PR.

---

## Phase 8 — Demo Surface (optional)  (2–3 days)

- [ ] Embeddable vanilla-JS chat widget (Shadow DOM).
- [ ] Admin dashboard (Jinja2) — view conversations, pending approvals, logs, costs.

**Exit criteria:** widget embeds with one script tag; admin approves a live refund from a dashboard.

---

## Interview narrative (the 5-minute demo)

> Customer asks → agent classifies → retrieves grounded answer from KB → calls tools →
> refund hits an **approval gate** → human approves in the dashboard → refund executes
> → every step traced, costed, and logged → eval suite proves retrieval/answer/faithfulness.

---

## Appendix — Carried over from the original plan

These two sections come from the earlier `implementation_plan.md`, which was retired because this roadmap replaced it.

### Mindset: ask "can I solve this without AI?" first

Before reaching for AI, always ask:

> **"Can I solve this without AI?"**

If `"refund" in message.lower()` routes 90% of tickets correctly — use that. It's cheaper, faster, and more reliable than an LLM call.

AI is for the problems that **can't** be solved with rules. Ambiguous language. Nuanced intent. Freeform knowledge retrieval. When you need AI, use it. When you don't, don't.

This mindset separates engineers who build things that work from engineers who build things that are impressive but fragile.

### Original tech stack and what was left out

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
