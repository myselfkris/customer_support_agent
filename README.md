# 🤖 Customer Support Agent

A modular AI-powered customer support system built with **Google Gemini** and **LangGraph**, progressing from a simple ticket classifier to a full RAG-powered agent pipeline.

---

## 📁 Project Structure

```
customer_support_agent/
├── agents/
│   ├── ticket_classifier.py     # Skill 1: Structured ticket classification
│   ├── tool_calling_agent.py    # Skill 2: Agent with tool use
│   ├── manual_agent_loop.py     # Skill 4: Manual ReAct agent loop
│   └── langgraph_agent.py       # Skill 5: LangGraph multi-step agent
├── rag/
│   ├── rag_pipeline.py          # Skill 3: Full RAG pipeline (pgvector + Gemini)
│   └── rag_eval.py              # RAG evaluation suite (20 test cases)
├── sample_docs/
│   ├── return_policy.txt        # Knowledge base — return policy
│   └── shipping_policy.txt      # Knowledge base — shipping policy
├── docs/
│   ├── rag_deep_dive.md         # Deep dive notes on RAG concepts
│   ├── evaluation_results.md    # RAG evaluation results
│   └── implementation_plan.md   # Project implementation plan
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🧠 Skills Breakdown

### Skill 1 — Ticket Classifier (`agents/ticket_classifier.py`)
Classifies incoming customer support messages into structured output using Gemini's **constrained JSON generation** with Pydantic schemas.

- **Intent**: `order_status`, `refund_request`, `product_question`, `complaint`, `price_match`, `feedback`, `general`, `other`
- **Urgency**: `no_urgency` → `low` → `medium` → `high` → `critical`
- **Sentiment**: `highly_positive` → `positive` → `neutral` → `negative` → `highly_negative`
- **Outputs**: `requires_tool` (bool), `confidence` (float), `reasoning` (str)

```bash
python agents/ticket_classifier.py
```

---

### Skill 2 — Tool Calling Agent (`agents/tool_calling_agent.py`)
An agent that calls real tools (order lookup, refund processing, etc.) based on the classified intent. Demonstrates function calling with Gemini.

```bash
python agents/tool_calling_agent.py
```

---

### Skill 3 — RAG Pipeline (`rag/rag_pipeline.py`)
Full **Retrieval-Augmented Generation** pipeline backed by **PostgreSQL + pgvector**.

**Two Phases:**
1. **Indexing** — Load docs → Clean → Chunk (paragraph-aware) → Embed (`text-embedding-004`) → Store in pgvector with HNSW index
2. **Querying** — Embed query → Cosine similarity search → Inject top-K chunks → Generate structured answer

```bash
# Index all documents in sample_docs/
python rag/rag_pipeline.py --index

# Ask a question
python rag/rag_pipeline.py --query "What is the return window?"

# Reset and re-index from scratch
python rag/rag_pipeline.py --reset

# Show indexing stats
python rag/rag_pipeline.py --stats
```

**Config defaults:**
| Setting | Value |
|---|---|
| Chunk size | 512 chars |
| Chunk overlap | 80 chars |
| Top-K retrieval | 5 |
| Min similarity | 0.50 |
| Embedding model | `text-embedding-004` |
| Generation model | `gemini-2.5-flash` |

---

### Skill 4 — Manual Agent Loop (`agents/manual_agent_loop.py`)
A hand-rolled **ReAct (Reason + Act)** agent loop built without any framework — pure Python. Demonstrates how agents think step by step before acting.

```bash
python agents/manual_agent_loop.py
```

---

### Skill 5 — LangGraph Agent (`agents/langgraph_agent.py`)
A production-grade multi-step agent built with **LangGraph** — handles complex customer queries by routing through a state graph of nodes (classify → retrieve → respond → escalate).

```bash
# Interactive mode
python agents/langgraph_agent.py

# Evaluation mode (runs test suite)
python agents/langgraph_agent.py --eval
```

---

### RAG Evaluation (`rag/rag_eval.py`)
Runs **20 structured test cases** across the RAG pipeline:
- 8 return policy questions
- 7 shipping policy questions
- 5 out-of-scope questions (tests graceful fallback)

```bash
python rag/rag_eval.py
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/myselfkris/customer_support_agent.git
cd customer_support_agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

**Required `.env` variables:**
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/customer_support
```

### 4. Set up PostgreSQL (for RAG pipeline)
Make sure PostgreSQL is running with the `pgvector` extension available:
```sql
CREATE DATABASE customer_support;
-- pgvector is enabled automatically by rag_pipeline.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google `text-embedding-004` |
| Vector DB | PostgreSQL + pgvector (HNSW index) |
| Agent Framework | LangGraph |
| Structured Output | Pydantic + Gemini JSON mode |
| Environment | python-dotenv |

---

## 📋 Requirements

```
google-genai
langgraph
psycopg2-binary
pgvector
pydantic
python-dotenv
```

Install all:
```bash
pip install -r requirements.txt
```
