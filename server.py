# server.py

import os
import re
import requests
import uuid
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text, Column, String, or_
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError

# Vector embeddings
from pgvector.sqlalchemy import Vector

# RAG deps
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import openai

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Globals ---
embedding_model = None
gemini_model = None
openai_client = None

# Thread pool for blocking DB operations
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

# --- Config / Env ---
DATABASE_URL = os.environ.get("DATABASE_URL")
print("Database URL:", DATABASE_URL)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()  # optional: 'openai' or 'gemini'

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set.")

# --- SQLAlchemy ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Model ---
class Paper(Base):
    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String)
    abstract = Column(String)
    authors = Column(String)
    url = Column(String)
    # Keep 384 dims to match MiniLM/bge-small/e5-small families (no schema change)
    embedding = Column(Vector(384))

# --- FastAPI app ---
app = FastAPI(
    title="The Aboutness Project Backend API",
    description="API for semantic search and RAG for academic papers.",
    version="0.3.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eohatdan.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utils ---

def chunk_text_words(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple word-based chunking with overlap to keep relationships local.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += step
    return chunks

def tokenize_query_for_keywords(q: str) -> List[str]:
    """
    Extract simple keywords (alnum + apostrophes) and drop very short tokens.
    """
    toks = re.findall(r"[A-Za-z0-9']+", q.lower())
    return [t for t in toks if len(t) >= 3]

# --- Dependencies ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    logger.info("Backend starting...")

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("DB connection OK")
    except OperationalError as e:
        logger.error(f"DB connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

    global embedding_model
    try:
        logger.info("Loading embedding model...")
        # 384-dim model to match schema; feel free to switch to 'bge-small-en-v1.5'
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.error(f"Embedding model load error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load embedding model.")

    global gemini_model
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini initialized.")
        except Exception as e:
            logger.error(f"Gemini init error: {e}")
            gemini_model = None
    else:
        logger.info("No GEMINI_API_KEY; Gemini disabled.")

    global openai_client
    if OPENAI_API_KEY:
        try:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized.")
        except Exception as e:
            logger.error(f"OpenAI init error: {e}")
            openai_client = None
    else:
        logger.info("No OPENAI_API_KEY; OpenAI disabled.")

    Base.metadata.create_all(bind=engine)
    logger.info("Schema ensured.")

# --- Root / Health ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to The Aboutness Project Backend API!"}

@app.get("/health")
async def health_check(db: SessionLocal = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database_connection": "successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

@app.get("/which-llm")
async def which_llm():
    # Provider preference (if set) + availability
    preferred = LLM_PROVIDER or ("gemini" if gemini_model else ("openai" if openai_client else "none"))
    return {
        "gemini_active": gemini_model is not None,
        "openai_active": openai_client is not None,
        "preferred": preferred,
    }

# --- Ingestion ---

class IngestGistFileRequest(BaseModel):
    gist_id: str
    filename: str

@app.post("/bulk-load-gist", status_code=status.HTTP_201_CREATED)
async def bulk_load_gist(gist_id: str, db: SessionLocal = Depends(get_db)):
    """
    Drops all existing data, fetches content from a multi-file Gist,
    CHUNKS each file, and ingests each chunk as a separate row.
    """
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

    # Reset table
    logger.info("Dropping and recreating 'papers' for bulk load...")
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("'papers' table reset.")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to reset table: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {e}")

    # Fetch gist metadata
    api_url = f"https://api.github.com/gists/{gist_id}"
    logger.info(f"Fetching gist metadata: {api_url}")
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        gist_data = response.json()
        files = gist_data.get("files", {})
    except requests.exceptions.RequestException as e:
        logger.error(f"Gist fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch Gist metadata: {e}")

    if not files:
        raise HTTPException(status_code=404, detail="No files found in the specified Gist.")

    to_ingest: List[Paper] = []

    for filename, file_info in files.items():
        raw_url = file_info.get("raw_url")
        if not raw_url:
            logger.warning(f"File '{filename}' has no raw_url; skipping.")
            continue

        logger.info(f"Fetching file: {filename}")
        try:
            content_resp = requests.get(raw_url)
            content_resp.raise_for_status()
            content = content_resp.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Content fetch error for {raw_url}: {e}")
            continue

        # --- CHUNKING ---
        chunks = chunk_text_words(content, size=1000, overlap=200)
        if not chunks:
            # fallback: at least create one chunk if empty split
            chunks = [content]

        for i, chunk in enumerate(chunks):
            emb = embedding_model.encode(chunk).tolist()
            to_ingest.append(
                Paper(
                    title=f"{filename}::chunk_{i:03d}",
                    abstract=chunk,
                    authors="Gist User",
                    url=raw_url,
                    embedding=emb,
                )
            )

    logger.info(f"Ingesting {len(to_ingest)} chunks...")
    try:
        db.bulk_save_objects(to_ingest)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database ingestion failed: {e}")

    return {"message": f"Ingested {len(to_ingest)} chunks from gist {gist_id}.", "chunks_ingested": len(to_ingest)}

@app.post("/ingest-gist-file", status_code=status.HTTP_201_CREATED)
async def ingest_gist_file(request_body: IngestGistFileRequest, db: SessionLocal = Depends(get_db)):
    """
    Fetches a single file from a Gist, CHUNKS it, and ingests chunks.
    """
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

    api_url = f"https://api.github.com/gists/{request_body.gist_id}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        gist_data = response.json()
        files = gist_data.get("files", {})
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Gist metadata: {e}")

    file_info = files.get(request_body.filename)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File '{request_body.filename}' not found in the gist.")

    raw_url = file_info.get("raw_url")
    if not raw_url:
        raise HTTPException(status_code=500, detail=f"Raw URL not found for '{request_body.filename}'.")

    try:
        content_response = requests.get(raw_url)
        content_response.raise_for_status()
        content = content_response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch content from {raw_url}: {e}")

    chunks = chunk_text_words(content, size=1000, overlap=200)
    if not chunks:
        chunks = [content]

    objects = []
    for i, chunk in enumerate(chunks):
        emb = embedding_model.encode(chunk).tolist()
        objects.append(
            Paper(
                title=f"{request_body.filename}::chunk_{i:03d}",
                abstract=chunk,
                authors="Gist User",
                url=raw_url,
                embedding=emb,
            )
        )

    try:
        db.bulk_save_objects(objects)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database ingestion failed: {e}")

    return {"message": f"Ingested {len(objects)} chunks from '{request_body.filename}'."}

# --- Query / Similarity ---

class QueryRequest(BaseModel):
    query: str
    limit: int = 5

class SimilarGistDocsRequest(BaseModel):
    gist_id: str
    document_title: str
    limit: int = 3

class SimilarLLMRequest(BaseModel):
    document_title: str
    limit: int = 3
    llm_type: str = "gemini"

def search_db_for_vectors_sync(db: SessionLocal, query_embedding_list: List[float], limit: int) -> List[Paper]:
    try:
        return (
            db.query(Paper)
            .order_by(Paper.embedding.cosine_distance(query_embedding_list))
            .limit(limit)
            .all()
        )
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []

def search_db_for_vectors_filtered_sync(db: SessionLocal, query_embedding_list: List[float], limit: int, exclude_title: str) -> List[Paper]:
    try:
        return (
            db.query(Paper)
            .filter(Paper.title != exclude_title)
            .order_by(Paper.embedding.cosine_distance(query_embedding_list))
            .limit(limit)
            .all()
        )
    except Exception as e:
        logger.error(f"Filtered vector search error: {e}")
        return []

def search_db_by_keywords_sync(db: SessionLocal, query_text: str, limit: int) -> List[Paper]:
    """
    Simple keyword fallback using ILIKE on title/abstract.
    """
    tokens = tokenize_query_for_keywords(query_text)
    if not tokens:
        return []
    ilike_terms = [Paper.title.ilike(f"%{t}%") for t in tokens] + [Paper.abstract.ilike(f"%{t}%") for t in tokens]
    try:
        return (
            db.query(Paper)
            .filter(or_(*ilike_terms))
            .limit(limit)
            .all()
        )
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []

@app.post("/query")
async def query_data(request_body: QueryRequest, db: SessionLocal = Depends(get_db)):
    """
    Hybrid RAG:
      - Embed query (cosine distance)
      - Pull a wider vector pool (~4x limit)
      - Pull keyword matches
      - Merge & dedupe, then trim to final limit
      - Strict prompt permits inverse family relations
      - Deterministic LLM (temperature=0)
    """
    global embedding_model, gemini_model, openai_client

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")
    if gemini_model is None and openai_client is None:
        raise HTTPException(status_code=500, detail="No LLM API configured.")
    if not request_body.query:
        raise HTTPException(status_code=400, detail="Query parameter is required in the body.")

    logger.info(f"Query: {request_body.query}")
    loop = asyncio.get_event_loop()

    # 1) Embed and vector-retrieve a larger pool
    q_emb = embedding_model.encode([request_body.query])[0].tolist()
    pool_k = max(request_body.limit, 5) * 4  # e.g., 5 -> 20
    vec_pool = await loop.run_in_executor(executor, search_db_for_vectors_sync, db, q_emb, pool_k)

    # 2) Keyword fallback pool
    kw_pool = await loop.run_in_executor(executor, search_db_by_keywords_sync, db, request_body.query, pool_k)

    # 3) Combine (vector order first), dedupe by id
    combined = []
    seen = set()
    for p in vec_pool + kw_pool:
        if p.id not in seen:
            combined.append(p)
            seen.add(p.id)

    if not combined:
        return {"response": "I could not find any relevant documents in the knowledge base to answer your question."}

    # 4) Trim to final limit for LLM context
    relevant = combined[:request_body.limit]
    documents_text = "\n\n".join([f"Title: {p.title}\nAuthors: {p.authors}\nAbstract: {p.abstract}" for p in relevant])

    # 5) Strict prompt allowing inverse family relations
    augmented_prompt = (
        "You are a factual QA assistant.\n"
        "RULES:\n"
        "1) Answer ONLY with facts from the Documents.\n"
        "2) You may infer an inverse family relationship if one direction is explicitly stated.\n"
        "   Examples: if A is the father of B, then B's father is A; if A is the son of B, then B is A's parent.\n"
        "3) If the Documents are insufficient, reply exactly: \"I don't know based on the provided documents.\"\n"
        "4) After your answer, include a short 'Sources:' list citing the document titles you used.\n\n"
        "Documents:\n"
        "```\n"
        f"{documents_text}\n"
        "```\n\n"
        "Question:\n"
        f"{request_body.query}\n"
    )

    logger.info("Sending to LLM...")
    try:
        llm_used = None
        llm_response = None

        provider = LLM_PROVIDER or ("gemini" if gemini_model else ("openai" if openai_client else ""))
        if provider == "openai" and openai_client:
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Follow the user's RULES exactly."},
                    {"role": "user", "content": augmented_prompt},
                ],
            )
            llm_response = completion.choices[0].message.content
            llm_used = "openai:gpt-3.5-turbo"
        elif provider == "gemini" and gemini_model:
            gen_cfg = {"temperature": 0.0, "top_p": 0.1, "top_k": 1, "max_output_tokens": 512}
            g = gemini_model.generate_content(augmented_prompt, generation_config=gen_cfg)
            llm_response = g.text
            llm_used = "gemini-2.0-flash"
        else:
            # fallback to whichever is available
            if gemini_model:
                gen_cfg = {"temperature": 0.0, "top_p": 0.1, "top_k": 1, "max_output_tokens": 512}
                g = gemini_model.generate_content(augmented_prompt, generation_config=gen_cfg)
                llm_response = g.text
                llm_used = "gemini-2.0-flash"
            elif openai_client:
                completion = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "Follow the user's RULES exactly."},
                        {"role": "user", "content": augmented_prompt},
                    ],
                )
                llm_response = completion.choices[0].message.content
                llm_used = "openai:gpt-3.5-turbo"

        return {
            "response": llm_response,
            "llm_used": llm_used,
            "relevant_documents": [
                {"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in relevant
            ],
        }
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

@app.post("/find-similar-gist-docs")
async def find_similar_gist_docs(request_body: SimilarGistDocsRequest, db: SessionLocal = Depends(get_db)):
    """
    Finds documents similar to a given document title (now chunk-friendly).
    If you pass a base filename ('file.txt'), it will match chunked titles like 'file.txt::chunk_000'.
    """
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

    # More flexible match against chunked titles
    target_paper = (
        db.query(Paper)
        .filter(Paper.title.ilike(f"{request_body.document_title}%"))
        .first()
    )

    if not target_paper:
        raise HTTPException(status_code=404, detail=f"Document with title starting '{request_body.document_title}' not found.")

    target_embedding = target_paper.embedding

    relevant_papers = await asyncio.get_event_loop().run_in_executor(
        executor, search_db_for_vectors_filtered_sync, db, target_embedding, request_body.limit + 1, target_paper.title
    )

    filtered_results = [p for p in relevant_papers if p.title != target_paper.title]
    if not filtered_results:
        return {"message": f"No similar documents found for '{request_body.document_title}'."}

    return {
        "similar_documents": [
            {"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in filtered_results
        ]
    }

@app.post("/find-similar-llm")
async def find_similar_llm(request_body: SimilarLLMRequest, db: SessionLocal = Depends(get_db)):
    """
    Uses an LLM to generate a search query from a document title and then finds similar articles.
    """
    global embedding_model, gemini_model, openai_client

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")
    if gemini_model is None and openai_client is None:
        raise HTTPException(status_code=500, detail="No LLM API configured.")

    prompt = (
        f"Generate a semantic search query for academic papers about the topic of: '{request_body.document_title}'. "
        "The query should be a single, concise sentence."
    )

    try:
        llm_response = None
        if request_body.llm_type == "gemini" and gemini_model:
            gen_cfg = {"temperature": 0.0, "top_p": 0.1, "top_k": 1, "max_output_tokens": 128}
            g = gemini_model.generate_content(prompt, generation_config=gen_cfg)
            llm_response = g.text
        elif request_body.llm_type == "openai" and openai_client:
            c = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            llm_response = c.choices[0].message.content
        else:
            raise HTTPException(status_code=400, detail=f"Invalid LLM type '{request_body.llm_type}' or API not configured.")

        if not llm_response:
            raise HTTPException(status_code=500, detail="LLM failed to generate a query.")

        generated_query = llm_response.strip().strip('"')
    except Exception as e:
        logger.error(f"LLM query generation error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM query generation failed: {e}")

    q_emb = embedding_model.encode([generated_query])[0].tolist()
    relevant = await asyncio.get_event_loop().run_in_executor(
        executor, search_db_for_vectors_sync, db, q_emb, request_body.limit
    )

    if not relevant:
        return {"message": "No similar documents found based on the LLM-generated query."}

    return {
        "similar_documents": [
            {"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in relevant
        ]
    }
