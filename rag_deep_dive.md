# Skill 3: RAG — Complete Pin-to-Pin Teaching Guide

> **Context**: You've completed Skill 1 (Structured Outputs) and Skill 2 (Tool Calling).
> RAG is what gives your agent *knowledge it wasn't trained on* — your company docs, policy PDFs, FAQs.
> Without RAG, your agent can only answer what Gemini already knows. That's useless for a real business.

---

## What Problem Does RAG Solve?

Imagine a customer asks:

> *"What is your return policy for items bought during the Black Friday sale?"*

Gemini doesn't know your company's Black Friday policy. It was never in its training data. You have two bad options and one good one:

| Option | Problem |
|---|---|
| Fine-tune the model | Expensive, slow, retrains every time policy changes |
| Stuff the entire policy doc into the prompt | Token limits, slow, expensive per-call |
| **RAG** ✅ | Find the *relevant* chunk, inject only that into the prompt |

RAG = **find the right piece of information → give it to the LLM → let the LLM answer using it.**

---

## The RAG Pipeline — 2 Phases

RAG is not one thing. It's a **two-phase system**:

```
PHASE 1: INDEXING (done once, offline)
  Your Docs → Chunking → Embedding → Store in Vector DB

PHASE 2: QUERYING (done at runtime, per user message)
  User Question → Embed Question → Search Vector DB → Retrieve Chunks → Inject into Prompt → LLM Answers
```

Every concept below belongs to one of these two phases. Know which phase you're in at all times.

---

## PHASE 1: INDEXING

### Concept 1: Document Loading

**What it is:**
Reading raw source documents (PDFs, Word docs, plain text, HTML pages) into Python as usable text strings.

**Why it's needed:**
Before you can do anything — chunking, embedding, storing — you need the raw text in memory. PDFs are binary files. Word docs are XML zipped files. You can't embed a `.pdf` file directly. You extract the text first.

**What you need to know:**
- **`PyMuPDF` (fitz)** — best for PDFs. Preserves layout better than PyPDF2.
- **`python-docx`** — for Word documents.
- **Plain `.txt`** — just `open()` and `read()`.
- **Web pages** — `requests` + `BeautifulSoup` to scrape and clean HTML.

**Key gotcha:**
PDFs sometimes have scanned images instead of real text (e.g., scanned policy documents). In that case, the text extraction returns empty strings. You need **OCR** (Optical Character Recognition) — libraries like `pytesseract` — but that's an advanced case. For now, assume text-based PDFs.

**In your project:**
You'll upload customer support docs (return policies, shipping info, FAQs). These will likely be PDFs or plain text files. Load them with PyMuPDF.

---

### Concept 2: Text Cleaning

**What it is:**
After loading, raw extracted text is messy — page numbers, headers, footers, double spaces, weird Unicode characters. Cleaning normalizes it.

**Why it's needed:**
Embeddings are sensitive to garbage. If chunk 1 is `"Return Policy\n\n\n\nPage 3\n\nItems purchased..."`, the embedding will partially represent "Page 3" — which is noise, not signal. Clean text = better embeddings = better retrieval.

**What you clean:**
- Remove page numbers: `re.sub(r'\bPage \d+\b', '', text)`
- Collapse multiple newlines: `re.sub(r'\n{3,}', '\n\n', text)`
- Strip leading/trailing whitespace per line
- Remove headers/footers if they repeat every page
- Normalize Unicode: `text.encode('ascii', 'ignore').decode()`

**Key gotcha:**
Don't over-clean. If you strip all newlines, you lose paragraph structure. Paragraph boundaries matter for chunking.

---

### Concept 3: Chunking

**What it is:**
Splitting a large document into smaller, fixed-size pieces called **chunks**. Each chunk is stored and retrieved independently.

**Why it's needed:**
You cannot embed an entire 50-page document as one unit. Three reasons:
1. **Embedding models have token limits** (Gemini `text-embedding-004` handles up to 2048 tokens per input).
2. **Precision**: If you embed 50 pages together, the embedding represents everything vaguely. If a user asks about returns, you want to retrieve *only* the return policy section — not the entire document diluted by shipping info.
3. **Context window cost**: Injecting 50 pages into your prompt is expensive and hits the LLM's context limit.

**The core parameters:**

| Parameter | What it is | Why it matters |
|---|---|---|
| `chunk_size` | Max tokens/characters per chunk | Too small = no context. Too large = noisy retrieval. |
| `chunk_overlap` | How many tokens the next chunk shares with the previous | Prevents losing context at chunk boundaries |

**The chunk boundary problem:**
Imagine this text:
```
...the customer must submit the return form. The form must be
submitted within 30 days of purchase. Refunds are processed...
```
If you cut right at "The form must be", your chunk ends mid-sentence. The next chunk starts with an orphaned "submitted within 30 days" with no context. **Overlap fixes this** — both chunks share that sentence.

**Chunking strategies (in order of sophistication):**

| Strategy | How it works | When to use |
|---|---|---|
| Fixed-size | Split every N characters | Simple, fast, baseline |
| Sentence-based | Split at sentence boundaries | Better for prose |
| Paragraph-based | Split at `\n\n` | Best for structured docs |
| Semantic | Split when topic changes (uses embeddings) | Advanced, expensive |
| Recursive | Try paragraph → sentence → word until fits | LangChain default, balanced |

**The magic numbers (from your implementation plan):**
> "Document chunking — why chunk size matters more than model choice"

This is the most underrated insight in RAG. Beginners obsess over which embedding model to use. The real performance lever is **chunk size**. A 512-token chunk with 50-token overlap, using a mediocre embedding model, will outperform a 4096-token chunk with a state-of-the-art model — because the smaller chunk is more precise.

**For your customer support agent:**
- `chunk_size = 512 tokens` (≈ 400 words)
- `chunk_overlap = 50 tokens` (≈ 40 words)
- Strategy: Recursive / paragraph-based

**Metadata per chunk:**
Every chunk needs metadata stored alongside it:
```json
{
  "text": "Refunds are processed within 5-7 business days...",
  "source": "return_policy_v2.pdf",
  "page": 3,
  "chunk_index": 12,
  "created_at": "2026-06-07"
}
```
**Why**: When you retrieve chunk 12, you want to tell the LLM *where* it came from. This enables citations: "According to `return_policy_v2.pdf`, page 3..."

---

### Concept 4: Embeddings

**What it is:**
Converting a text chunk (a string) into a **vector** — a list of numbers (e.g., 768 numbers) — that represents its *meaning* in mathematical space.

**Why it's needed:**
Computers can't compare meaning directly. They can compare numbers. Embeddings turn "What is your refund process?" and "How do I get my money back?" into vectors that are *close together in space* — because they mean the same thing — even though they share zero words.

This is how RAG finds relevant chunks without keyword matching.

**The mental model:**
Think of a 3D space (it's actually 768D, but imagine 3D). Every sentence is a point in this space. Sentences with similar meaning cluster together. "Dog" and "Puppy" are close. "Dog" and "Tax Return" are far apart.

**Cosine similarity:**
The distance metric used. It measures the *angle* between two vectors, not their length.
- Cosine similarity of `1.0` = identical meaning
- Cosine similarity of `0.0` = completely unrelated
- Cosine similarity of `-1.0` = opposite meaning (rare in practice)

**Why cosine, not Euclidean distance?**
Euclidean distance is affected by vector magnitude (length). Two documents of different length saying the same thing would have different magnitudes. Cosine similarity ignores magnitude — it only cares about direction = meaning.

**Your embedding model (from implementation plan):**
> `Gemini text-embedding-004` — Same SDK, 768 dimensions, asymmetric search support

**Asymmetric search** = the query embedding and the document embedding are generated differently. A question ("What is the refund policy?") and a document answer ("Refunds are processed in 5-7 days...") don't look the same syntactically, but an asymmetric model knows to make them close in vector space. Gemini `text-embedding-004` supports this via `task_type`:

```python
# For indexing (the document chunk)
embed_document("Refunds are processed...", task_type="RETRIEVAL_DOCUMENT")

# For querying (the user question)
embed_query("What is the refund policy?", task_type="RETRIEVAL_QUERY")
```

**What you DON'T do:**
You embed once during indexing. You do **not** re-embed the documents every time a user asks a question. That would be catastrophically slow and expensive.

---

### Concept 5: Vector Database (pgvector)

**What it is:**
A database that stores vectors (embeddings) and can efficiently search for the most similar vectors to a given query vector.

**Why it's needed:**
You have 1,000 chunks, each a 768-dimensional vector. A user asks a question. You embed their question (another 768D vector). Now you need to find which of your 1,000 stored vectors is *most similar* to the question vector. You cannot do a full scan of 1,000 vectors manually every time — you need a database optimized for this.

**Your stack (from implementation plan):**
> PostgreSQL + pgvector — One database for everything — relational data AND vectors

**Why pgvector over specialized vector DBs (Pinecone, Weaviate, Qdrant)?**
At your scale (thousands to tens of thousands of chunks), pgvector in PostgreSQL is:
- Cheaper (you already have PostgreSQL)
- Simpler (one DB, one connection, one backup)
- Good enough (pgvector's HNSW index handles millions of vectors fine)

You'd only switch to a dedicated vector DB at 10M+ vectors with sub-10ms latency requirements.

**The pgvector data model:**
```sql
CREATE TABLE document_chunks (
    id          SERIAL PRIMARY KEY,
    source      TEXT,           -- "return_policy_v2.pdf"
    page        INT,            -- 3
    chunk_index INT,            -- 12
    content     TEXT,           -- the actual chunk text
    embedding   vector(768),    -- the 768-dim embedding
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

**The similarity search query:**
```sql
SELECT content, source, page,
       1 - (embedding <=> query_embedding) AS similarity
FROM document_chunks
ORDER BY embedding <=> query_embedding   -- <=> is cosine distance
LIMIT 5;                                 -- top 5 most relevant chunks
```

**The `<=>` operator** = cosine distance in pgvector. Note: distance, not similarity. So `ORDER BY ASC` gives you the most similar (smallest distance).

**HNSW Index:**
For fast search, you create an index:
```sql
CREATE INDEX ON document_chunks
USING hnsw (embedding vector_cosine_ops);
```
HNSW (Hierarchical Navigable Small World) = approximate nearest neighbor algorithm. It trades tiny accuracy loss for massive speed gain. At 10K chunks, the difference is milliseconds. It matters at 1M+ chunks.

---

## PHASE 2: QUERYING (Runtime)

### Concept 6: Query Embedding

**What it is:**
At runtime, when a user sends a message, you embed that message using the same embedding model — but with `task_type="RETRIEVAL_QUERY"`.

**Why it's needed:**
To search the vector database, you need to convert the user's question into the same vector space as your stored chunks. You can't search a vector database with text — you need a vector.

**Key rule:**
Always use `RETRIEVAL_QUERY` task type for questions and `RETRIEVAL_DOCUMENT` task type for chunks during indexing. Mixing these up is a common bug that silently destroys retrieval quality.

---

### Concept 7: Retrieval (Similarity Search)

**What it is:**
Using the query vector to find the top-K most similar chunks in your vector database.

**Why it's needed:**
This is the "R" in RAG. Without retrieval, you're just "AG" — which is just a regular LLM call. Retrieval grounds the LLM's answer in *your* documents.

**The K parameter (top-K):**
How many chunks do you retrieve? This is a tradeoff:

| K value | Pros | Cons |
|---|---|---|
| Too small (K=1) | Precise, low token cost | Miss relevant info if best chunk is wrong |
| Too large (K=10) | Comprehensive | Noisy, expensive, dilutes the answer |
| **K=3-5** ✅ | Balanced | Sweet spot for most use cases |

**For your project:** Start with `K=5`.

**Similarity threshold:**
Don't just return the top K chunks blindly. If the *best* chunk has a similarity score of 0.3, that's very low — the document probably doesn't contain the answer. You filter by a **minimum similarity threshold**.

```python
MIN_SIMILARITY = 0.75

results = db.similarity_search(query_embedding, k=5)
relevant_chunks = [r for r in results if r.similarity >= MIN_SIMILARITY]

if not relevant_chunks:
    return "I don't have that information."
```

**This is the "I don't know" response** from your implementation plan:
> "The 'I don't know' response — when retrieval finds nothing relevant"

This is *crucial*. Without a threshold, your agent will always attempt to answer — hallucinating details from the closest-but-still-irrelevant chunk. With a threshold, it says "I don't have that info" instead of lying.

---

### Concept 8: Re-ranking (Optional but Powerful)

**What it is:**
After retrieving top-K chunks by vector similarity, running a second, more expensive model to re-score and reorder them by true relevance.

**Why it's needed:**
Vector similarity is fast but imperfect. Embeddings compress meaning into 768 numbers — information is lost. A cross-encoder re-ranker reads the full (query, chunk) pair and scores them much more accurately than vector similarity alone.

**The pipeline:**
```
Query → Vector Search (fast, approximate) → Top 20 candidates
→ Re-ranker (slow, precise) → Top 5 truly relevant chunks
```

**For your project:**
Skip re-ranking in Skill 3. Add it in Skill 6 (Reliability). Mention it here so you know it exists.

---

### Concept 9: Context Injection (Prompt Construction)

**What it is:**
Taking the retrieved chunks and inserting them into the LLM prompt before asking the LLM to answer.

**Why it's needed:**
The LLM doesn't know what you retrieved. You have to *tell* it. Context injection is how you bridge the retrieval step and the generation step. This is the "A" and "G" in RAG.

**The prompt structure:**
```
[SYSTEM PROMPT]
You are a customer support agent for Acme Corp.
Answer questions using ONLY the context provided below.
If the context does not contain the answer, say exactly:
"I don't have that information in my knowledge base."
Never guess. Never fabricate.

[RETRIEVED CONTEXT]
--- Source: return_policy_v2.pdf, Page 3 ---
Refunds are processed within 5-7 business days of receiving
the returned item. Items must be in original condition...

--- Source: return_policy_v2.pdf, Page 4 ---
Black Friday sale items are eligible for store credit only.
Cash refunds are not available for sale items purchased...

[USER QUESTION]
What is the refund policy for Black Friday items?
```

**Key design decisions:**

| Decision | Choice | Why |
|---|---|---|
| Where to put context | After system prompt, before user question | LLM processes it as background knowledge, not as a question |
| Include source metadata | Yes | Enables citations, helps LLM ground its answer |
| Separator between chunks | `---` with source label | Clearly separates chunks so LLM knows where each ends |
| Instruction to say "I don't know" | In system prompt | Repeat the guardrail where the LLM will see it right before answering |

**The negative constraint** (from Skill 1 — structured outputs):
> "Never guess. Never fabricate."

This is where Skill 1 and Skill 3 connect. The prompt engineering you learned in Skill 1 is the scaffolding that makes RAG trustworthy.

---

### Concept 10: Answer Generation

**What it is:**
The final LLM call with the injected context, where the model synthesizes an answer using *only* the retrieved information.

**Why it's needed:**
The LLM's job here is NOT to use its training data. Its job is to be a reading comprehension machine — read the provided context, find the answer, express it clearly. This is a fundamentally different mode than free-form generation.

**Prompt guardrails that enforce this:**
1. `"Answer using ONLY the context provided"` — blocks hallucination
2. `"If the context doesn't contain the answer, say 'I don't have that information'"` — blocks guessing
3. `"Cite the source when possible"` — enforces groundedness
4. `"Do not answer from your general knowledge"` — closes the loophole

**Structured output here (connecting to Skill 1):**
You can wrap the RAG answer in a Pydantic model:
```python
class RAGResponse(BaseModel):
    answer: str
    sources: list[str]       # ["return_policy_v2.pdf, Page 3"]
    confidence: float        # 0.0 - 1.0 — how confident is the LLM?
    retrieval_failed: bool   # True if no relevant chunks found
```

This gives you a clean, typed response with provenance — you know *why* the LLM said what it said.

---

## The Evaluation Framework (Your 20 Q&A Pairs)

From your implementation plan:
> **Eval:** 20 test Q&A pairs. Track two things separately:
> (1) retrieval accuracy — did we find the right chunks?
> (2) answer accuracy — did the LLM use them correctly? Target: 80%+ on both.

**Why track them separately?**
Because the failure modes are different:

| Failure | Retrieval Score | Answer Score | What to fix |
|---|---|---|---|
| Wrong chunks retrieved | Low | Low | Chunk size, overlap, similarity threshold |
| Right chunks, wrong answer | High | Low | System prompt, context injection format |
| No chunks returned (should have been) | Low | N/A | Lower similarity threshold |
| Chunks returned (should not have been) | High (wrong kind) | Low | Raise similarity threshold |

**Test case structure:**
```json
{
  "question": "What is the return window for Black Friday items?",
  "expected_answer": "Store credit only, no cash refunds",
  "expected_source": "return_policy_v2.pdf",
  "retrieval_should_succeed": true
}
```

---

## The Full RAG Flow — End to End

```
INDEXING (one time)
1. Load PDF → extract text
2. Clean text (remove headers, page numbers, noise)
3. Split into chunks (512 tokens, 50 overlap)
4. Add metadata (source, page, chunk_index)
5. Embed each chunk (task_type="RETRIEVAL_DOCUMENT")
6. Store (text + embedding + metadata) in PostgreSQL/pgvector

QUERYING (every user message)
1. User sends message: "What is your Black Friday return policy?"
2. Classify: does this need RAG? (from Skill 1 — use structured output)
3. Embed user question (task_type="RETRIEVAL_QUERY")
4. Vector search: find top 5 chunks with similarity >= 0.75
5. If no chunks meet threshold → return "I don't have that information"
6. Build prompt: [system] + [retrieved chunks with sources] + [user question]
7. Call Gemini → get RAGResponse (answer + sources + confidence)
8. Return answer to user with source citations
```

---

## Common Mistakes (And Why They Happen)

| Mistake | Why it's wrong | Fix |
|---|---|---|
| Chunk size too large (2000+ tokens) | Noisy embeddings, imprecise retrieval | Use 512 tokens |
| No overlap between chunks | Context lost at boundaries | Add 50-token overlap |
| Using `RETRIEVAL_DOCUMENT` task type for queries | Wrong embedding direction, bad retrieval | Use `RETRIEVAL_QUERY` for questions |
| No similarity threshold | Retrieves irrelevant chunks, LLM hallucinates | Set minimum 0.75 threshold |
| Not saying "I don't know" | Agent fabricates answers confidently | Add explicit guardrail in system prompt |
| Re-embedding documents every query | Extremely slow and expensive | Embed once during indexing, cache in DB |
| Storing embeddings in application memory | Lost on restart | Always persist to PostgreSQL |
| Not storing metadata with chunks | Can't cite sources | Always store source + page + chunk_index |

---

## What "Done" Looks Like (From Your Implementation Plan)

> Upload a PDF → ask questions → get accurate answers grounded in the doc.
> Ask something NOT in the PDF → get "I don't have that information."
> Tested against 20 Q&A pairs.

When you hit this — you've completed Skill 3.

---

## Tech Stack Summary for Skill 3

| Component | Tool | Why |
|---|---|---|
| Document loading | `PyMuPDF` (fitz) | Best PDF text extraction |
| Text chunking | Custom recursive chunker | Control over chunk_size and overlap |
| Embedding model | `Gemini text-embedding-004` | Same SDK, asymmetric support, 768D |
| Vector storage | `PostgreSQL + pgvector` | Already in stack, no new infra |
| Similarity search | pgvector `<=>` operator + HNSW index | Fast cosine distance at scale |
| LLM generation | `Gemini 2.5 Flash` | Same as Skills 1 & 2, no new SDK |
| Response schema | Pydantic `RAGResponse` | Typed, structured, auditable output |

---

## What Comes Next (Skill 4 Preview)

After RAG, you'll build the **Manual Agent Loop** — which wires Skill 1 (classify intent) + Skill 2 (tool calling) + Skill 3 (RAG) into a while loop. You'll see why managing state manually becomes painful at 3+ branches. That pain is intentional — it's what makes LangGraph (Skill 5) feel like relief, not overhead.

RAG is the last *isolated* skill. Starting Skill 4, everything connects.
