# server.py

import os
import re
import requests
import uuid
import logging
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import defaultdict, deque

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text, Column, String, or_
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError

from pgvector.sqlalchemy import Vector

from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import openai

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- Globals --------
embedding_model = None
gemini_model = None
openai_client = None
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

# -------- Env / Config --------
DATABASE_URL = os.environ.get("DATABASE_URL")
print("Database URL:", DATABASE_URL)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()  # optional: 'openai' or 'gemini'

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set.")

# -------- DB setup --------
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Paper(Base):
    __tablename__ = "papers"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String)
    abstract = Column(String)
    authors = Column(String)
    url = Column(String)
    embedding = Column(Vector(384))  # keep 384d

# -------- FastAPI --------
app = FastAPI(
    title="The Aboutness Project Backend API",
    description="API for semantic search and RAG for academic papers.",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eohatdan.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Utils: chunking, tokenizing --------
def chunk_text_words(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    words = text.split()
    if not words:
        return []
    out = []
    i = 0
    step = max(1, size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += step
    return out

def tokenize_query_for_keywords(q: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9']+", q.lower())
    return [t for t in toks if len(t) >= 3]

# -------- Relation Extraction (regex) --------
# Supported relations: father, mother, parent, son, daughter, child, husband, wife, spouse
NAME = r"[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}"  # Up to 4 tokens
OF = r"\s+of\s+"
IS_THE = r"\s+is\s+the\s+"
WAS_THE = r"\s+was\s+the\s+"
IS = r"\s+is\s+"
WAS = r"\s+was\s+"

REL_PATTERNS = [
    # A is the father of B / A was the father of B
    (re.compile(fr"({NAME})(?:{IS_THE}|{WAS_THE})(father|mother|son|daughter|husband|wife|spouse|parent|child){OF}({NAME})", re.IGNORECASE), "dir"),
    # A is B's father / A was B's father
    (re.compile(fr"({NAME})(?:{IS}|{WAS})(?:{NAME})'?s\s+(father|mother|son|daughter|husband|wife|spouse|parent|child)", re.IGNORECASE), "poss"),
    # B's father is A
    (re.compile(fr"({NAME})'?s\s+(father|mother|son|daughter|husband|wife|spouse|parent|child){IS}({NAME})", re.IGNORECASE), "inv"),
]

# Canonical relation directions we store in the graph:
# parent_of(A,B), child_of(A,B), spouse_of(A,B) (undirected)
def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def rel_to_edges(subj: str, rel: str, obj: str) -> List[Tuple[str, str, str]]:
    s = normalize_name(subj)
    o = normalize_name(obj)
    r = rel.lower()
    edges = []
    if r in ["father", "mother", "parent"]:
        edges.append(("parent_of", s, o))
        edges.append(("child_of", o, s))
    elif r in ["son", "daughter", "child"]:
        edges.append(("child_of", s, o))
        edges.append(("parent_of", o, s))
    elif r in ["husband", "wife", "spouse"]:
        # store as undirected: add both ways
        edges.append(("spouse_of", s, o))
        edges.append(("spouse_of", o, s))
    return edges

def extract_relations(text: str) -> List[Tuple[str, str, str]]:
    triples = []
    for pat, kind in REL_PATTERNS:
        for m in pat.finditer(text):
            if kind == "dir":
                a, r, b = m.group(1), m.group(2), m.group(3)
                triples.extend(rel_to_edges(a, r, b))
            elif kind == "poss":
                # "A is B's father" -> subject=A, rel=father, object=B
                a, b, r = m.group(1), m.group(2), m.group(3)
                # careful: groups swapped; pattern: (A) is (B)'s (rel)
                # Our regex was (A)(is/was)(B)'s (rel) — groups(1)=A, (2)=B, (3)=rel
                triples.extend(rel_to_edges(a, r, b))
            elif kind == "inv":
                # "B's father is A" -> subject=B, rel=father, object=A
                b, r, a = m.group(1), m.group(2), m.group(3)
                triples.extend(rel_to_edges(a, r, b))  # A is parent_of B, etc.
    return triples

# Build a tiny graph and reason simple queries (parent/child/spouse and grandparents)
class MiniGraph:
    def __init__(self):
        self.parents: Dict[str, Set[str]] = defaultdict(set)   # parents[child] -> {parent1, parent2}
        self.children: Dict[str, Set[str]] = defaultdict(set)  # children[parent] -> {child1, child2}
        self.spouses: Dict[str, Set[str]]  = defaultdict(set)  # spouses[a] -> {b,...}
        self.edge_sources: Dict[Tuple[str,str,str], Set[str]] = defaultdict(set)  # (rel, a, b) -> {source_titles}

    def add(self, rel: str, a: str, b: str, source_title: str):
        if rel == "parent_of":
            self.parents[b].add(a)
            self.children[a].add(b)
        elif rel == "child_of":
            self.children[b].add(a)
            self.parents[a].add(b)
        elif rel == "spouse_of":
            self.spouses[a].add(b)
            self.spouses[b].add(a)
        self.edge_sources[(rel, a, b)].add(source_title)

    def get_parents(self, person: str) -> Set[str]:
        return self.parents.get(person, set())

    def get_children(self, person: str) -> Set[str]:
        return self.children.get(person, set())

    def get_spouses(self, person: str) -> Set[str]:
        return self.spouses.get(person, set())

    def get_grandparents(self, person: str) -> Set[str]:
        gps = set()
        for p in self.get_parents(person):
            gps.update(self.get_parents(p))
        return gps

    def sources_for(self, rel: str, a: str, b: str) -> Set[str]:
        return self.edge_sources.get((rel, a, b), set())

# Parse natural questions into (target_person, relation)
REL_ALIASES = {
    "father": "father",
    "mother": "mother",
    "parent": "parent",
    "son": "son",
    "daughter": "daughter",
    "child": "child",
    "husband": "spouse",
    "wife": "spouse",
    "spouse": "spouse",
    "grandfather": "grandfather",
    "grandmother": "grandmother",
    "grandparent": "grandparent",
}

Q_PATTERNS = [
    # who is A's <rel> ?
    re.compile(r"who\s+is\s+(.+?)'s\s+([a-z]+)\??", re.IGNORECASE),
    # who is the <rel> of A ?
    re.compile(r"who\s+is\s+the\s+([a-z]+)\s+of\s+(.+?)\??", re.IGNORECASE),
]

def parse_question(q: str) -> Tuple[str, str]:
    q = q.strip()
    for pat in Q_PATTERNS:
        m = pat.search(q)
        if m:
            if pat.pattern.startswith("who\\s+is\\s+(.+?)'s"):
                person, rel = m.group(1), m.group(2).lower()
            else:
                rel, person = m.group(1).lower(), m.group(2)
            rel = REL_ALIASES.get(rel, rel)
            return (normalize_name(person), rel)
    return ("", "")

def answer_from_graph(G: MiniGraph, person: str, rel: str) -> Tuple[List[str], List[str]]:
    """
    Returns (answers, sources_list). answers is list of names.
    sources_list is a list of source titles used (deduped).
    """
    answers: Set[str] = set()
    sources: Set[str] = set()

    if not person or not rel:
        return ([], [])

    if rel in ["father", "mother", "parent"]:
        parents = G.get_parents(person)
        if rel == "father":
            # we don't track gender; return all parents, LLM or downstream UI can display both
            for p in parents:
                answers.add(p)
                sources.update(G.sources_for("parent_of", p, person))
        elif rel == "mother":
            for p in parents:
                answers.add(p)
                sources.update(G.sources_for("parent_of", p, person))
        else:  # parent
            for p in parents:
                answers.add(p)
                sources.update(G.sources_for("parent_of", p, person))

    elif rel in ["son", "daughter", "child"]:
        children = G.get_children(person)
        for c in children:
            answers.add(c)
            sources.update(G.sources_for("parent_of", person, c))

    elif rel == "spouse":
        spouses = G.get_spouses(person)
        for s in spouses:
            answers.add(s)
            # spouses stored undirected; record any spouse_of edge source
            sources.update(G.sources_for("spouse_of", person, s))
            sources.update(G.sources_for("spouse_of", s, person))

    elif rel in ["grandfather", "grandmother", "grandparent"]:
        gps = G.get_grandparents(person)
        for gp in gps:
            answers.add(gp)
            # collect sources via parent links (two hops)
            for p in G.get_parents(person):
                sources.update(G.sources_for("parent_of", p, person))
                for gpp in G.get_parents(p):
                    if gpp == gp:
                        sources.update(G.sources_for("parent_of", gp, p))

    return (sorted(answers), sorted(set(sources)))

# -------- Dependencies --------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------- Startup --------
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
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
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

# -------- Basic endpoints --------
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
    preferred = LLM_PROVIDER or ("gemini" if gemini_model else ("openai" if openai_client else "none"))
    return {"gemini_active": gemini_model is not None, "openai_active": openai_client is not None, "preferred": preferred}

# -------- Ingestion --------
class IngestGistFileRequest(BaseModel):
    gist_id: str
    filename: str

@app.post("/bulk-load-gist", status_code=status.HTTP_201_CREATED)
async def bulk_load_gist(gist_id: str, db: SessionLocal = Depends(get_db)):
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")
    logger.info("Resetting 'papers' for bulk load...")
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {e}")

    api_url = f"https://api.github.com/gists/{gist_id}"
    logger.info(f"Fetching gist metadata: {api_url}")
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        gist_data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Gist metadata: {e}")

    files = gist_data.get("files", {})
    if not files:
        raise HTTPException(status_code=404, detail="No files found in the specified Gist.")

    to_ingest: List[Paper] = []
    for filename, file_info in files.items():
        raw_url = file_info.get("raw_url")
        if not raw_url:
            continue
        try:
            content_resp = requests.get(raw_url)
            content_resp.raise_for_status()
            content = content_resp.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Content fetch error for {raw_url}: {e}")
            continue

        chunks = chunk_text_words(content, size=1000, overlap=200) or [content]
        for i, chunk in enumerate(chunks):
            emb = embedding_model.encode(chunk).tolist()
            to_ingest.append(Paper(
                title=f"{filename}::chunk_{i:03d}",
                abstract=chunk,
                authors="Gist User",
                url=raw_url,
                embedding=emb,
            ))

    try:
        db.bulk_save_objects(to_ingest)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database ingestion failed: {e}")

    return {"message": f"Ingested {len(to_ingest)} chunks from gist {gist_id}.", "chunks_ingested": len(to_ingest)}

@app.post("/ingest-gist-file", status_code=status.HTTP_201_CREATED)
async def ingest_gist_file(request_body: IngestGistFileRequest, db: SessionLocal = Depends(get_db)):
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")
    api_url = f"https://api.github.com/gists/{request_body.gist_id}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        gist_data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Gist metadata: {e}")

    file_info = gist_data.get("files", {}).get(request_body.filename)
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

    chunks = chunk_text_words(content, size=1000, overlap=200) or [content]
    objs = []
    for i, chunk in enumerate(chunks):
        emb = embedding_model.encode(chunk).tolist()
        objs.append(Paper(
            title=f"{request_body.filename}::chunk_{i:03d}",
            abstract=chunk,
            authors="Gist User",
            url=raw_url,
            embedding=emb,
        ))
    try:
        db.bulk_save_objects(objs)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database ingestion failed: {e}")

    return {"message": f"Ingested {len(objs)} chunks from '{request_body.filename}'."}

# -------- Querying --------
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
    tokens = tokenize_query_for_keywords(query_text)
    if not tokens:
        return []
    ilikes = [Paper.title.ilike(f"%{t}%") for t in tokens] + [Paper.abstract.ilike(f"%{t}%") for t in tokens]
    try:
        return db.query(Paper).filter(or_(*ilikes)).limit(limit).all()
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []

def build_graph_from_chunks(chunks: List[Paper]) -> MiniGraph:
    G = MiniGraph()
    for p in chunks:
        triples = extract_relations(p.abstract)
        for (rel, a, b) in triples:
            G.add(rel, a, b, p.title)
    return G

@app.post("/query")
async def query_data(request_body: QueryRequest, db: SessionLocal = Depends(get_db)):
    global embedding_model, gemini_model, openai_client

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")
    if gemini_model is None and openai_client is None:
        raise HTTPException(status_code=500, detail="No LLM API configured.")
    if not request_body.query:
        raise HTTPException(status_code=400, detail="Query parameter is required in the body.")

    loop = asyncio.get_event_loop()
    q_emb = embedding_model.encode([request_body.query])[0].tolist()
    pool_k = max(request_body.limit, 5) * 4

    vec_pool = await loop.run_in_executor(executor, search_db_for_vectors_sync, db, q_emb, pool_k)
    kw_pool  = await loop.run_in_executor(executor, search_db_by_keywords_sync, db, request_body.query, pool_k)

    # Merge & dedupe by id (vector order first)
    combined, seen = [], set()
    for p in vec_pool + kw_pool:
        if p.id not in seen:
            combined.append(p)
            seen.add(p.id)

    if not combined:
        return {"response": "I could not find any relevant documents in the knowledge base to answer your question."}

    relevant = combined[:request_body.limit]
    documents_text = "\n\n".join([f"Title: {p.title}\nAuthors: {p.authors}\nAbstract: {p.abstract}" for p in relevant])

    # ---------- NEW: Graph reasoning pass ----------
    person, rel = parse_question(request_body.query)
    graph_answers, graph_sources = ([], [])
    if person and rel:
        G = build_graph_from_chunks(relevant)
        graph_answers, graph_sources = answer_from_graph(G, person, rel)

    if graph_answers:
        # Deterministic graph-derived answer with sources (no LLM)
        return {
            "response": ", ".join(graph_answers),
            "llm_used": "graph",
            "sources": graph_sources,
            "relevant_documents": [{"title": p.title, "authors": p.authors, "url": p.url} for p in relevant],
        }

    # ---------- Fall back to LLM with stricter prompt ----------
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
            "relevant_documents": [{"title": p.title, "authors": p.authors, "url": p.url} for p in relevant],
        }
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

# -------- Similarity helpers --------
@app.post("/find-similar-gist-docs")
async def find_similar_gist_docs(request_body: SimilarGistDocsRequest, db: SessionLocal = Depends(get_db)):
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

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
    filtered = [p for p in relevant_papers if p.title != target_paper.title]
    if not filtered:
        return {"message": f"No similar documents found for '{request_body.document_title}'."}
    return {"similar_documents": [{"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in filtered]}

@app.post("/find-similar-llm")
async def find_similar_llm(request_body: SimilarLLMRequest, db: SessionLocal = Depends(get_db)):
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")
    if gemini_model is None and openai_client is None:
        raise HTTPException(status_code=500, detail="No LLM API configured.")

    prompt = (f"Generate a semantic search query for academic papers about the topic of: '{request_body.document_title}'. "
              "The query should be a single, concise sentence.")
    try:
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
    return {"similar_documents": [{"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in relevant]}
