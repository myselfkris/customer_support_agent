"""
Skill 3: RAG Pipeline — Full Implementation
============================================

Two phases:
  PHASE 1 — INDEXING  (run once): load docs → chunk → embed → store in PostgreSQL/pgvector
  PHASE 2 — QUERYING  (per user): embed question → vector search → inject context → LLM answers

Usage:
  python rag_pipeline.py --index          # Index all docs in sample_docs/
  python rag_pipeline.py --query "..."    # Ask a question
  python rag_pipeline.py --reset          # Drop and recreate the table (re-index from scratch)
"""

import os
import re
import json
import argparse
import time
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")
DB_URL             = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5432/customer_support")

EMBEDDING_MODEL    = "text-embedding-004"
GENERATION_MODEL   = "gemini-2.5-flash"

CHUNK_SIZE         = 512    # characters (≈ 128 tokens — keeps chunks precise)
CHUNK_OVERLAP      = 80     # characters shared between consecutive chunks
TOP_K              = 5      # chunks to retrieve
MIN_SIMILARITY     = 0.50   # minimum cosine similarity to accept a chunk

DOCS_DIR           = Path(__file__).parent / "sample_docs"


# ============================================================
# PYDANTIC RESPONSE SCHEMA
# ============================================================

class RAGResponse(BaseModel):
    answer: str
    sources: list[str]          # e.g. ["return_policy.txt — chunk 3"]
    confidence: float           # 0.0 → 1.0 — LLM self-report
    retrieval_failed: bool      # True when no chunks met the threshold


# ============================================================
# DATABASE — setup and helpers
# ============================================================

def get_connection():
    """Open and return a psycopg2 connection with pgvector registered."""
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn


def setup_table(conn):
    """
    Create the document_chunks table if it doesn't exist.
    Enables pgvector extension and an HNSW index for fast cosine search.
    """
    with conn.cursor() as cur:
        # Enable the pgvector extension (safe to run repeatedly)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Main chunks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          SERIAL PRIMARY KEY,
                source      TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content     TEXT NOT NULL,
                embedding   vector(768),
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # HNSW index for fast approximate cosine search (created once)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops);
        """)

    conn.commit()
    print("[DB] Table and index ready.")


def reset_table(conn):
    """Drop and recreate the chunks table — use when re-indexing from scratch."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS document_chunks;")
    conn.commit()
    setup_table(conn)
    print("[DB] Table reset complete.")


def chunk_already_indexed(conn, source: str) -> bool:
    """Return True if this source file already has chunks in the DB (avoid re-indexing)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE source = %s;",
            (source,)
        )
        count = cur.fetchone()[0]
    return count > 0


# ============================================================
# PHASE 1A — DOCUMENT LOADING
# ============================================================

def load_text_file(path: Path) -> str:
    """Load a plain .txt file."""
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")


def load_document(path: Path) -> str:
    """Route to the correct loader based on file extension."""
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_text_file(path)
    elif ext == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ============================================================
# PHASE 1B — TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Remove noise that would pollute embeddings:
    - Collapse 3+ newlines to double newline (preserve paragraph breaks)
    - Strip leading/trailing whitespace per line
    - Remove page number patterns like "Page 3" or "- 3 -"
    """
    # Remove common page number patterns
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Collapse 3+ blank lines into a single blank line (keep paragraph structure)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing spaces from each line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


# ============================================================
# PHASE 1C — CHUNKING
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Paragraph-aware recursive chunker:
    1. Try to split on paragraph boundaries (\n\n) first — preserves semantic units
    2. If a paragraph is still larger than chunk_size, fall back to sentence splitting
    3. Merges small paragraphs into chunks up to chunk_size with overlap

    This is what your rag_deep_dive.md calls "Recursive / paragraph-based" chunking.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If the paragraph alone exceeds chunk_size, split it by sentences
        if len(para) > chunk_size:
            # Flush the current chunk before processing the long paragraph
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Carry over the last `overlap` characters into the next chunk
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk

            # Split long paragraph by sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = current_chunk[-overlap:] + " " + sentence
                    else:
                        # Single sentence longer than chunk_size — hard cut
                        chunks.append(sentence[:chunk_size].strip())
                        current_chunk = sentence[chunk_size - overlap:]
        else:
            # Normal paragraph — append to current chunk if it fits
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    current_chunk = para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# PHASE 1D — EMBEDDING
# ============================================================

def embed_documents(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of document chunks using RETRIEVAL_DOCUMENT task type.
    Called during INDEXING — embeds the stored knowledge.
    """
    embeddings = []
    for i, text in enumerate(texts):
        # Small delay to respect free tier rate limits
        if i > 0 and i % 5 == 0:
            time.sleep(1)
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings.append(response.embeddings[0].values)
    return embeddings


def embed_query(client: genai.Client, query: str) -> list[float]:
    """
    Embed a single user query using RETRIEVAL_QUERY task type.
    Called during QUERYING — must use a different task type than documents.

    WHY different task types:
    - RETRIEVAL_DOCUMENT: optimized for long-form content being stored
    - RETRIEVAL_QUERY: optimized for short questions being searched
    Mixing them silently destroys retrieval quality.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


# ============================================================
# PHASE 1E — STORING CHUNKS IN VECTOR DB
# ============================================================

def store_chunks(conn, source: str, chunks: list[str], embeddings: list[list[float]]):
    """
    Insert chunks + embeddings + metadata into PostgreSQL/pgvector.
    Uses execute_values for a single efficient bulk insert.
    """
    rows = [
        (source, idx, chunk, embedding)
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO document_chunks (source, chunk_index, content, embedding)
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s::vector)"
        )
    conn.commit()
    print(f"  [DB] Stored {len(chunks)} chunks from '{source}'")


# ============================================================
# PHASE 1 — INDEX ALL DOCUMENTS (entry point)
# ============================================================

def index_documents(client: genai.Client, conn, docs_dir: Path = DOCS_DIR):
    """
    Full indexing pipeline:
      For each document in docs_dir:
        load → clean → chunk → embed → store
    Skips files already indexed (idempotent).
    """
    setup_table(conn)

    supported_extensions = {".txt", ".pdf"}
    doc_files = [f for f in docs_dir.iterdir() if f.suffix.lower() in supported_extensions]

    if not doc_files:
        print(f"[INDEXING] No documents found in {docs_dir}")
        return

    print(f"\n[INDEXING] Found {len(doc_files)} document(s) to process.")

    for doc_path in doc_files:
        source_name = doc_path.name
        print(f"\n  Processing: {source_name}")

        # Skip if already indexed (idempotent)
        if chunk_already_indexed(conn, source_name):
            print(f"  [SKIP] '{source_name}' already indexed. Use --reset to re-index.")
            continue

        # Step 1: Load raw text
        raw_text = load_document(doc_path)
        print(f"  Loaded {len(raw_text)} characters")

        # Step 2: Clean
        clean = clean_text(raw_text)

        # Step 3: Chunk
        chunks = chunk_text(clean)
        print(f"  Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

        # Step 4: Embed
        print(f"  Embedding {len(chunks)} chunks...")
        embeddings = embed_documents(client, chunks)

        # Step 5: Store
        store_chunks(conn, source_name, chunks, embeddings)

    print("\n[INDEXING] Complete.")


# ============================================================
# PHASE 2A — RETRIEVAL (Vector Similarity Search)
# ============================================================

def retrieve_chunks(conn, query_embedding: list[float], top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY) -> list[dict]:
    """
    Find the top-K most similar chunks to the query embedding.

    Uses pgvector's <=> cosine DISTANCE operator (lower = more similar).
    Converts to similarity score: similarity = 1 - distance.

    Filters by min_similarity threshold BEFORE returning.
    If nothing meets the threshold → returns empty list (triggers "I don't know" path).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                source,
                chunk_index,
                content,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k)
        )
        rows = cur.fetchall()

    results = []
    for source, chunk_index, content, similarity in rows:
        if similarity >= min_similarity:
            results.append({
                "source": source,
                "chunk_index": chunk_index,
                "content": content,
                "similarity": round(float(similarity), 4),
            })

    return results


# ============================================================
# PHASE 2B — CONTEXT INJECTION (Prompt Construction)
# ============================================================

RAG_SYSTEM_PROMPT = """You are a customer support agent for Acme Corp.
Your ONLY job is to answer customer questions using the provided CONTEXT below.

STRICT RULES:
1. Answer ONLY using information from the CONTEXT. Never use your own training knowledge.
2. If the CONTEXT does not contain enough information to answer, respond EXACTLY with:
   "I don't have that information in my knowledge base."
3. Always cite the source document when you use information from it.
4. Never guess, fabricate, or infer beyond what the CONTEXT explicitly states.
5. Be concise, friendly, and professional.
"""

def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the context injection prompt.

    Structure:
      [CONTEXT]
      --- Source: return_policy.txt | Chunk 3 | Similarity: 0.91 ---
      <chunk text>

      [QUESTION]
      <user question>

    Key decisions:
    - Chunks ordered by similarity descending — most relevant first (then last due to Lost-in-Middle)
    - Source + chunk_index label on each chunk → enables citations
    - Similarity score shown → helps you audit retrieval quality during development
    """
    context_blocks = []
    for chunk in chunks:
        block = (
            f"--- Source: {chunk['source']} | Chunk {chunk['chunk_index']} | "
            f"Similarity: {chunk['similarity']} ---\n"
            f"{chunk['content']}"
        )
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    return f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"


# ============================================================
# PHASE 2C — ANSWER GENERATION
# ============================================================

def generate_answer(client: genai.Client, prompt: str, sources: list[str]) -> RAGResponse:
    """
    Call the LLM with the injected context and get a structured RAGResponse.
    Uses Gemini's response_schema to force valid JSON output.
    """
    response_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {
                "type": "array",
                "items": {"type": "string"}
            },
            "confidence": {"type": "number"},
            "retrieval_failed": {"type": "boolean"}
        },
        "required": ["answer", "sources", "confidence", "retrieval_failed"]
    }

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=RAG_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1,   # Low temp for factual retrieval tasks
        ),
    )

    data = json.loads(response.text)
    return RAGResponse(**data)


# ============================================================
# PHASE 2 — FULL QUERY PIPELINE (entry point)
# ============================================================

def query(client: genai.Client, conn, question: str, verbose: bool = True) -> RAGResponse:
    """
    Full RAG querying pipeline:
      1. Embed the user's question (RETRIEVAL_QUERY task type)
      2. Search vector DB for top-K chunks above min_similarity
      3. If no chunks found → return "I don't know" without calling LLM (saves cost!)
      4. Build the context injection prompt
      5. Call LLM → get RAGResponse (answer + sources + confidence)

    Returns a RAGResponse Pydantic object.
    """
    if verbose:
        print(f"\n[QUERY] '{question}'")

    # Step 1: Embed the question
    query_embedding = embed_query(client, question)

    # Step 2: Retrieve relevant chunks
    chunks = retrieve_chunks(conn, query_embedding)

    if verbose:
        print(f"  Retrieved {len(chunks)} chunk(s) above threshold ({MIN_SIMILARITY})")
        for c in chunks:
            print(f"    - {c['source']} chunk {c['chunk_index']}: similarity={c['similarity']}")

    # Step 3: No relevant chunks found → skip LLM call entirely (saves cost)
    if not chunks:
        if verbose:
            print("  [RETRIEVAL FAILED] No chunks met the similarity threshold.")
        return RAGResponse(
            answer="I don't have that information in my knowledge base.",
            sources=[],
            confidence=0.0,
            retrieval_failed=True,
        )

    # Step 4: Build prompt with injected context
    source_labels = [f"{c['source']} — chunk {c['chunk_index']}" for c in chunks]
    prompt = build_rag_prompt(question, chunks)

    # Step 5: Generate the answer
    result = generate_answer(client, prompt, source_labels)

    if verbose:
        print(f"  Confidence: {result.confidence}")
        print(f"  Sources: {result.sources}")
        print(f"\n  ANSWER: {result.answer}\n")

    return result


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline — Skill 3")
    parser.add_argument("--index",  action="store_true", help="Index all documents in sample_docs/")
    parser.add_argument("--reset",  action="store_true", help="Drop and recreate chunks table, then re-index")
    parser.add_argument("--query",  type=str,            help="Ask a question against the indexed documents")
    parser.add_argument("--stats",  action="store_true", help="Show indexing stats from the DB")
    args = parser.parse_args()

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        conn = get_connection()
    except Exception as e:
        print(f"[ERROR] Could not connect to PostgreSQL: {e}")
        print("Make sure PostgreSQL is running and DATABASE_URL is set in .env")
        return

    if args.reset:
        print("[RESET] Dropping and recreating table...")
        reset_table(conn)
        print("[RESET] Re-indexing all documents...")
        index_documents(client, conn)

    elif args.index:
        index_documents(client, conn)

    elif args.query:
        # Make sure table exists before querying
        setup_table(conn)
        query(client, conn, args.query)

    elif args.stats:
        setup_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT source, COUNT(*) FROM document_chunks GROUP BY source;")
            rows = cur.fetchall()
        print("\n[STATS] Indexed documents:")
        for source, count in rows:
            print(f"  {source}: {count} chunks")

    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
