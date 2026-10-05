# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run everything from the repo root as modules (`python -m ...`) — packages are `agents`, `rag`, `app`, `db`.

```bash
pip install -r requirements.txt          # or: pip install -e ".[dev]"
cp .env.example .env                     # GEMINI_API_KEY, MOCK_MODE, JWT_SECRET, + DB URLs (see "Two databases" below)

uvicorn app.main:app --reload            # API (docs at :8000/docs)

python -m agents.ticket_classifier       # Skill 1
python -m agents.tool_calling_agent      # Skill 2
python -m agents.manual_agent_loop       # Skill 4
python -m agents.langgraph_agent [--eval]        # Skill 5
python -m agents.reliable_agent [--eval]         # Skill 6 (--eval is offline failure-injection)
python -m agents.approval_workflow       # pausable human-in-the-loop demo
python -m agents.reliable_agent_e2e_eval

python -m rag.rag_pipeline --index | --reset | --stats | --query "..."
python -m rag.rag_eval                   # 20 RAG test cases
```

There is no pytest suite or linter configured. Verification is via standalone scripts: `python smoke_auth.py` (auth + HTTP doors via `TestClient`), `python smoke_phase3.py`, `python demo_phase3.py`.

**Mock mode:** set `MOCK_MODE=true` to run the classifier, tool loop, RAG query and API fully offline (keyword-rule mocks in `agents/mock.py`, no Gemini/pgvector calls). Use it for development and smoke tests.

## Architecture

Gemini (`google-genai`) + LangGraph + PostgreSQL/pgvector, with SQLAlchemy for the business DB. The project is built incrementally in "skills"/"chunks"; `docs/architecture_guide.md` has a full function map and `docs/roadmap.md` the plan.

Request path (`POST /chat`): `app/main.py` → `ReliableAgent.chat` (`agents/reliable_agent.py`: validation, prompt-injection refusal, per-user rate limit, in-memory short-term memory, retries, fallback, JSONL logs to `logs/`) → `_run_agent`, the single switch between `MockApp` (mock mode) and the LangGraph graph (`agents/langgraph_agent.py::build_graph`). The graph is `classify_node` → `route_message` → one of `tool_node` / `rag_node` / `escalate_node` / `direct_node`. `tool_node` calls `tool_calling_agent.run_agent` (LLM picks tool → execute via `TOOL_FUNCTIONS` → LLM writes reply); `rag_node` calls `rag.rag_pipeline.query`.

Key points that span files:
- **Adding a tool** requires three edits in `agents/tool_calling_agent.py`: the function, its `*_declaration` (added to `tools`), and the `TOOL_FUNCTIONS` name→function map. Tools use the DB models in `db/models.py` via `init_db()` / `SessionLocal()` with commit and `finally: session.close()`.
- **Mock parity:** behavior changes to routing/classification/tools usually need a matching change in `agents/mock.py` so mock mode keeps mirroring the real graph.
- **Two approval patterns coexist:** (A) DB `Approval` rows created by tools (`process_refund` / `cancel_order` create pending rows; admins resolve via `approve_approval` / `decline_approval`, listed by `/admin/approvals`); (B) `agents/approval_workflow.py`, a LangGraph graph that pauses with `interrupt()`, persisted by a SQLite checkpointer (`approval_checkpoints.db`), resumed with `Command(resume=...)`.
- **Auth** (`app/auth.py`): `/chat` identity comes from a customer API key (`cust_live_` prefix, stored hashed) — never from the request body; admin endpoints use password → JWT (`JWT_SECRET` must be set).
- **Concurrency:** `ReliableAgent` keeps per-user memory/rate-limit state in process, so `app/main.py` serializes calls with `_agent_lock` and runs them via `asyncio.to_thread`. Don't assume multi-worker safety.
- **Config:** `app/config.py` (`pydantic-settings`, reads `.env` at repo root) is the settings source for the API; the agent/rag scripts also read env vars directly via python-dotenv.

## Two databases, two env vars — do not mix them up

| Env var | Database | Used by | Default if unset |
|---|---|---|---|
| `RELATIONAL_DATABASE_URL` | **Business DB**: customers, orders, refunds, approvals, tickets, admin users, API keys | SQLAlchemy in `db/base.py` → all tools, `app/auth.py` | `sqlite:///./customer_support.db` |
| `DATABASE_URL` | **RAG vector store only** (pgvector chunks + embeddings) | raw psycopg2 in `rag/rag_pipeline.py`; also `settings.database_url` in `app/config.py` | `postgresql://postgres:password@localhost:5432/customer_support` |

- `DATABASE_URL` does **not** affect the business DB. Changing it never moves orders/refunds/auth data; set `RELATIONAL_DATABASE_URL` for that.
- `RELATIONAL_DATABASE_URL` is not in `.env.example`; without it the business DB is always the local SQLite file (seed with `python -m db.seed`; delete the file to reset).
- `approval_checkpoints.db` is a third, separate SQLite file (LangGraph checkpointer for `agents/approval_workflow.py`), hardcoded, not configurable via either variable.
