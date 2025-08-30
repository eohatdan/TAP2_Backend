# server.py

import os
import re
import requests
import uuid
import logging
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import defaultdict

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
# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Globals ----------------
embedding_model = None
gemini_model = None
openai_client = None
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

# ---------------- Env / Config ----------------
DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set.")

# ---------------- DB setup ----------------
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
    embedding = Column(Vector(384))  # E5-small-v2 is 384-d

# Auto-parsed facts (overwritten on ingest)
class Relation(Base):
    __tablename__ = "relations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent = Column(String, index=True)
    child  = Column(String, index=True)
    source = Column(String)

class Spouse(Base):
    __tablename__ = "spouses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    a = Column(String, index=True)
    b = Column(String, index=True)
    source = Column(String)

class Nickname(Base):
    __tablename__ = "nicknames"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alias    = Column(String, index=True)
    fullname = Column(String, index=True)
    source   = Column(String)

# Manual overrides (NEVER cleared on ingest)
class ManualRelation(Base):
    __tablename__ = "manual_relations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent = Column(String, index=True)
    child  = Column(String, index=True)
    note   = Column(String, default="")  # optional
    source = Column(String, default="manual")

class ManualSpouse(Base):
    __tablename__ = "manual_spouses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    a = Column(String, index=True)
    b = Column(String, index=True)
    note   = Column(String, default="")
    source = Column(String, default="manual")

class ManualNickname(Base):
    __tablename__ = "manual_nicknames"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alias    = Column(String, index=True)
    fullname = Column(String, index=True)
    note     = Column(String, default="")
    source   = Column(String, default="manual")

# ---------------- FastAPI ----------------
app = FastAPI(
    title="The Aboutness Project Backend API",
    description="API for semantic search and RAG for academic papers.",
    version="0.6.0",
)

from fastapi.middleware.cors import CORSMiddleware

ALLOWED_ORIGINS = ["https://eohatdan.github.io"]  # exact origin, no trailing slash

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],  # include OPTIONS
    allow_headers=["*"],                       # allow Content-Type, etc.
    expose_headers=["Content-Type"],
    max_age=86400,
)

from fastapi import Response

@app.options("/{path:path}")
def preflight_cors(path: str, response: Response):
    origin = "https://eohatdan.github.io"
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
    response.headers["Access-Control-Max-Age"] = "86400"
    return Response(status_code=204)


# ---------------- Utilities ----------------
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

def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

# --- Retrieval-tuned embeddings (E5: prefixes + normalization) ---
def embed_doc(text: str) -> List[float]:
    return embedding_model.encode("passage: " + text, normalize_embeddings=True).tolist()

def embed_query(text: str) -> List[float]:
    return embedding_model.encode("query: " + text, normalize_embeddings=True).tolist()

# ---------------- Relation Extraction ----------------
NAME = r"[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}"
OF = r"\s+of\s+"
IS_THE = r"\s+is\s+the\s+"
WAS_THE = r"\s+was\s+the\s+"
IS = r"\s+is\s+"
WAS = r"\s+was\s+"

REL_PATTERNS = [
    (re.compile(fr"({NAME})(?:{IS_THE}|{WAS_THE})(father|mother|son|daughter|husband|wife|spouse|parent|child){OF}({NAME})", re.IGNORECASE), "dir"),
    (re.compile(fr"({NAME})(?:{IS}|{WAS})({NAME})'?s\s+(father|mother|son|daughter|husband|wife|spouse|parent|child)", re.IGNORECASE), "poss"),
    (re.compile(fr"({NAME})'?s\s+(father|mother|son|daughter|husband|wife|spouse|parent|child){IS}({NAME})", re.IGNORECASE), "inv"),
]

CHILDREN_BLOCK = re.compile(
    fr"({NAME})\s+has\s+(?:one|two|three|four|five|\d+)\s+children?:\s*(.+?)(?:(?:\n\s*\n)|$)",
    re.IGNORECASE | re.DOTALL
)
SINGLE_CHILD = re.compile(
    fr"({NAME})\s+has\s+one\s+child:\s*({NAME})\s*\((son|daughter)\.?\)\.?",
    re.IGNORECASE
)

NICK_BOTH = re.compile(fr"({NAME})\s+is\s+the\s+nickname\s+of\s+both\s+({NAME})\s+and\s+({NAME})", re.IGNORECASE)
NICK_ONE  = re.compile(fr"({NAME})\s+is\s+the\s+nickname\s+of\s+({NAME})", re.IGNORECASE)

PERSON = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}$")  # 1–4 capitalized tokens

BAD_TOKENS = (
    "High School","University","College","WV","San Jose","Huntington","California","West Virginia",
    "degree","bachelor","masters","master’s","information","sciences","born","on","July","August",
    "County","State","Road","Street","Avenue","Vin son","Vinson"
)

def is_person(name: str) -> bool:
    n = normalize_name(name)
    if any(x in n for x in BAD_TOKENS):
        return False
    return bool(PERSON.match(n))


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
        edges.append(("spouse_of", s, o))
        edges.append(("spouse_of", o, s))
    return edges

def _parse_children_lines(parent: str, block_text: str) -> List[Tuple[str, str, str]]:
    edges: List[Tuple[str,str,str]] = []
    candidates = re.split(r"[\n,]+", block_text)
    for c in candidates:
        m = re.search(fr"({NAME})\s*\(\s*(son|daughter)\.?\s*\)\.?", c.strip(), re.IGNORECASE)
        if m:
            child_name = normalize_name(m.group(1))
            edges.extend(rel_to_edges(child_name, "child", parent))
    return edges

def extract_relations(text: str) -> Tuple[List[Tuple[str, str, str]], Dict[str, List[str]]]:
    triples: List[Tuple[str,str,str]] = []
    nicknames: Dict[str, List[str]] = {}

    for pat, kind in REL_PATTERNS:
        for m in pat.finditer(text):
            if kind == "dir":
                a, r, b = m.group(1), m.group(2), m.group(3)
                triples.extend(rel_to_edges(a, r, b))
            elif kind == "poss":
                a, b, r = m.group(1), m.group(2), m.group(3)
                triples.extend(rel_to_edges(a, r, b))
            elif kind == "inv":
                b, r, a = m.group(1), m.group(2), m.group(3)
                triples.extend(rel_to_edges(a, r, b))

    for m in CHILDREN_BLOCK.finditer(text):
        parent = normalize_name(m.group(1))
        block  = m.group(2)
        triples.extend(_parse_children_lines(parent, block))

    for m in SINGLE_CHILD.finditer(text):
        parent = normalize_name(m.group(1))
        child  = normalize_name(m.group(2))
        triples.extend(rel_to_edges(child, "child", parent))

    for m in NICK_BOTH.finditer(text):
        alias = normalize_name(m.group(1))
        a = normalize_name(m.group(2))
        b = normalize_name(m.group(3))
        nicknames.setdefault(alias, [])
        for full in (a, b):
            if full not in nicknames[alias]:
                nicknames[alias].append(full)
    for m in NICK_ONE.finditer(text):
        alias = normalize_name(m.group(1))
        full  = normalize_name(m.group(2))
        nicknames.setdefault(alias, [])
        if full not in nicknames[alias]:
            nicknames[alias].append(full)

    return triples, nicknames

# ---------------- Mini Graph (for fallback reasoning) ----------------
class MiniGraph:
    def __init__(self):
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.spouses: Dict[str, Set[str]]  = defaultdict(set)
        self.edge_sources: Dict[Tuple[str,str,str], Set[str]] = defaultdict(set)

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

def build_graph_from_chunks(chunks: List['Paper']) -> Tuple['MiniGraph', Dict[str, List[str]]]:
    G = MiniGraph()
    nick_map: Dict[str, List[str]] = {}
    for p in chunks:
        triples, nicks = extract_relations(p.abstract)
        for (rel, a, b) in triples:
            G.add(rel, a, b, p.title)
        for alias, fulls in nicks.items():
            nick_map.setdefault(alias, [])
            for f in fulls:
                if f not in nick_map[alias]:
                    nick_map[alias].append(f)
    return G, nick_map

def graph_all_people(G: 'MiniGraph') -> Set[str]:
    names = set(G.parents.keys()) | set(G.children.keys()) | set(G.spouses.keys())
    for kids in G.parents.values(): names |= kids
    for kids in G.children.values(): names |= kids
    for sps in G.spouses.values(): names |= sps
    return names

def expand_person_candidates(person: str, G: 'MiniGraph', nick_map: Dict[str, List[str]]) -> List[str]:
    cands = set([person])
    names = graph_all_people(G)
    if person in nick_map:
        cands.update(nick_map[person])
    tokens = person.lower().split()
    if len(tokens) >= 2:
        for full in names:
            ftoks = full.lower().split()
            i = 0; ok = True
            for t in tokens:
                try: j = ftoks.index(t, i); i = j + 1
                except ValueError: ok = False; break
            if ok: cands.add(full)
    if len(tokens) == 1:
        token = tokens[0]
        for full in names:
            if any(t == token for t in full.lower().split()):
                cands.add(full)
    return sorted(cands)

REL_ALIASES = {
    "father": "father", "mother": "mother", "parent": "parent",
    "son": "son", "daughter": "daughter", "child": "child",
    "husband": "spouse", "wife": "spouse", "spouse": "spouse",
    "grandfather": "grandparent", "grandmother": "grandparent", "grandparent": "grandparent",
}

# accept straight ' and curly ’; apostrophe optional before s
Q_PATTERNS = [
    re.compile(r"who\s+is\s+(.+?)[’']?s\s+([a-z]+)\??", re.IGNORECASE),
    re.compile(r"who\s+is\s+the\s+([a-z]+)\s+of\s+(.+?)\??", re.IGNORECASE),
]


def parse_question(q: str) -> Tuple[str, str]:
    q = q.strip()
    for pat in Q_PATTERNS:
        m = pat.search(q)
        if m:
            if pat is Q_PATTERNS[0]:
                person, rel = m.group(1), m.group(2).lower()
            else:
                rel, person = m.group(1).lower(), m.group(2)
            rel = REL_ALIASES.get(rel, rel)
            return (normalize_name(person), rel)
    return ("", "")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- Startup ----------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting backend...")
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

    global embedding_model
    embedding_model = SentenceTransformer("intfloat/e5-small-v2")  # retrieval-tuned
    global gemini_model
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception:
            gemini_model = None
    global openai_client
    if OPENAI_API_KEY:
        try:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            openai_client = None

    # Ensure schema exists (creates tables if missing; does not drop)
    Base.metadata.create_all(bind=engine)
    logger.info("Startup OK.")

# ---------------- Basic endpoints ----------------
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
    return {
        "gemini_active": gemini_model is not None,
        "openai_active": openai_client is not None,
        "preferred": preferred
    }

# ---------------- Ingestion ----------------
class IngestGistFileRequest(BaseModel):
    gist_id: str
    filename: str

def _clear_auto_tables(db):
    # Clear auto-parsed content only; keep manual_* tables intact
    db.query(Paper).delete()
    db.query(Relation).delete()
    db.query(Spouse).delete()
    db.query(Nickname).delete()
    db.commit()

@app.post("/bulk-load-gist", status_code=status.HTTP_201_CREATED)
async def bulk_load_gist(gist_id: str, db: SessionLocal = Depends(get_db)):
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

    # Only clear auto tables; manual_* tables persist
    _clear_auto_tables(db)

    api_url = f"https://api.github.com/gists/{gist_id}"
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
    inserted_rel = 0
    inserted_sp  = 0
    inserted_nk  = 0

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
            title = f"{filename}::chunk_{i:03d}"
            # 1) Store Paper + embedding
            emb = embed_doc(chunk)
            to_ingest.append(Paper(
                title=title,
                abstract=chunk,
                authors="Gist User",
                url=raw_url,
                embedding=emb,
            ))
            # 2) Extract relations + nicknames for deterministic SQL answering
            triples, nicks = extract_relations(chunk)
            for (rel, a, b) in triples:
                if rel == "parent_of":
                    db.add(Relation(parent=a, child=b, source=title)); inserted_rel += 1
                elif rel == "spouse_of":
                    db.add(Spouse(a=a, b=b, source=title)); inserted_sp += 1
            for alias, fulls in nicks.items():
                for full in fulls:
                    db.add(Nickname(alias=alias, fullname=full, source=title)); inserted_nk += 1

    try:
        db.bulk_save_objects(to_ingest)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database ingestion failed: {e}")

    return {
        "message": f"Ingested {len(to_ingest)} chunks from gist {gist_id}.",
        "chunks_ingested": len(to_ingest),
        "relations": inserted_rel,
        "spouses": inserted_sp,
        "nicknames": inserted_nk
    }

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
        title = f"{request_body.filename}::chunk_{i:03d}"
        emb = embed_doc(chunk)
        objs.append(Paper(
            title=title,
            abstract=chunk,
            authors="Gist User",
            url=raw_url,
            embedding=emb,
        ))
        triples, nicks = extract_relations(chunk)
        for (rel, a, b) in triples:
            if rel == "parent_of":
                db.add(Relation(parent=a, child=b, source=title))
            elif rel == "spouse_of":
                db.add(Spouse(a=a, b=b, source=title))
        for alias, fulls in nicks.items():
            for full in fulls:
                db.add(Nickname(alias=alias, fullname=full, source=title))

    try:
        db.bulk_save_objects(objs)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database ingestion failed: {e}")

    return {"message": f"Ingested {len(objs)} chunks from '{request_body.filename}'."}

# ---------------- SQL-first answering ----------------
def sql_person_candidates(db, person: str):
    # 1) manual nicknames first
    rows = db.execute(text("""
        SELECT fullname FROM manual_nicknames WHERE alias = :p
        UNION SELECT :p AS fullname
    """), {"p": person}).fetchall()
    cands = {r[0] for r in rows}

    # 2) auto nicknames
    rows = db.execute(text("""
        SELECT fullname FROM nicknames WHERE alias = :p
    """), {"p": person}).fetchall()
    cands.update({r[0] for r in rows})

    # 3) token-subsequence match across known names
    tokens = person.lower().split()
    if tokens:
        name_rows = db.execute(text("""
            SELECT parent FROM relations
            UNION SELECT child FROM relations
            UNION SELECT a FROM spouses
            UNION SELECT b FROM spouses
            UNION SELECT parent FROM manual_relations
            UNION SELECT child FROM manual_relations
            UNION SELECT a FROM manual_spouses
            UNION SELECT b FROM manual_spouses
        """)).fetchall()
        for (full,) in name_rows:
            ftoks = full.lower().split()
            i = 0; ok = True
            for t in tokens:
                try:
                    j = ftoks.index(t, i); i = j + 1
                except ValueError:
                    ok = False; break
            if ok:
                cands.add(full)
    return sorted(cands)

def answer_with_sql(db, person: str, rel: str):
    cands = sql_person_candidates(db, person)

    def dedup(items):
        out, seen = [], set()
        for it in items:
            if it not in seen:
                out.append(it); seen.add(it)
        return out

    used_manual_any = False
    answers, sources = [], []

    def fetch_manual_then_auto(sql_manual: str, sql_auto: str, params: dict):
        nonlocal used_manual_any
        rows_m = db.execute(text(sql_manual), params).fetchall()
        if rows_m:
            used_manual_any = True
            return rows_m
        return db.execute(text(sql_auto), params).fetchall()

    if rel in ("father","mother","parent"):
        for c in cands:
            rows = fetch_manual_then_auto(
                "SELECT parent, source FROM manual_relations WHERE child = :c",
                "SELECT parent, source FROM relations WHERE child = :c",
                {"c": c}
            )
            answers += [r[0] for r in rows]; sources += [r[1] for r in rows]

    elif rel in ("son","daughter","child"):
        for c in cands:
            rows = fetch_manual_then_auto(
                "SELECT child, source FROM manual_relations WHERE parent = :c",
                "SELECT child, source FROM relations WHERE parent = :c",
                {"c": c}
            )
            answers += [r[0] for r in rows]; sources += [r[1] for r in rows]

    elif rel == "spouse":
        for c in cands:
            rows_m = db.execute(text("""
                SELECT b, source FROM manual_spouses WHERE a = :c
                UNION SELECT a, source FROM manual_spouses WHERE b = :c
            """), {"c": c}).fetchall()
            if rows_m:
                used_manual_any = True
                rows = rows_m
            else:
                rows = db.execute(text("""
                    SELECT b, source FROM spouses WHERE a = :c
                    UNION SELECT a, source FROM spouses WHERE b = :c
                """), {"c": c}).fetchall()
            answers += [r[0] for r in rows]; sources += [r[1] for r in rows]

    elif rel == "grandparent":
        for c in cands:
            rows_m = db.execute(text("""
                SELECT DISTINCT gp.parent, p.source, gp.source
                FROM manual_relations p
                JOIN manual_relations gp ON gp.child = p.parent
                WHERE p.child = :c
            """), {"c": c}).fetchall()
            if rows_m:
                used_manual_any = True
                answers += [r[0] for r in rows_m]
                for r in rows_m: sources += [r[1], r[2]]
            else:
                rows_a = db.execute(text("""
                    SELECT DISTINCT gp.parent, p.source, gp.source
                    FROM relations p
                    JOIN relations gp ON gp.child = p.parent
                    WHERE p.child = :c
                """), {"c": c}).fetchall()
                answers += [r[0] for r in rows_a]
                for r in rows_a: sources += [r[1], r[2]]

    answers = dedup(answers)
    sources = dedup([s for s in sources if s])
    provenance = "manual" if used_manual_any else ("auto" if answers else "none")
    return answers, sources, provenance


# ---------------- Retrieval helpers ----------------
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

# ---------------- Query endpoint ----------------
class QueryRequest(BaseModel):
    query: str
    limit: int = 5

@app.post("/query")
async def query_data(request_body: QueryRequest, db: SessionLocal = Depends(get_db)):
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")
    if not request_body.query:
        raise HTTPException(status_code=400, detail="Query parameter is required in the body.")

    # 1) SQL-first deterministic answering for family relations
    person, rel = parse_question(request_body.query)
    if person and rel:
        names, srcs, prov = answer_with_sql(db, person, rel)
        if names:
            return {
                "response": ", ".join(names),
                "llm_used": "sql-graph",
                "provenance": prov,   # "manual" or "auto"
                "sources": srcs
            }

    # 2) If not a relation (or no SQL facts), do hybrid retrieval
    loop = asyncio.get_event_loop()
    q_emb = embed_query(request_body.query)
    pool_k = max(request_body.limit, 5) * 8
    vec_pool = await loop.run_in_executor(executor, search_db_for_vectors_sync, db, q_emb, pool_k)
    kw_pool  = await loop.run_in_executor(executor, search_db_by_keywords_sync, db, request_body.query, pool_k)

    combined, seen = [], set()
    for p in vec_pool + kw_pool:
        if p.id not in seen:
            combined.append(p); seen.add(p.id)

    if not combined:
        return {"response": "I could not find any relevant documents in the knowledge base to answer your question."}

    # Build graph on a wider pool for robustness
    max_context = 100
    context_pool = combined[:max_context]
    relevant = combined[:request_body.limit]
    documents_text = "\n\n".join([f"Title: {p.title}\nAuthors: {p.authors}\nAbstract: {p.abstract}" for p in relevant])

    # 3) Graph fallback for relations if SQL had nothing
    if person and rel:
        G, nick_map = build_graph_from_chunks(context_pool)
        candidate_persons = expand_person_candidates(person, G, nick_map)
        all_answers, all_sources = set(), set()
        for cand in candidate_persons:
            ans, srcs2 = answer_from_graph(G, cand, rel)
            all_answers.update(ans); all_sources.update(srcs2)
        if all_answers:
            return {
                "response": ", ".join(sorted(all_answers)),
                "llm_used": "graph",
                "sources": sorted(all_sources),
                "relevant_documents": [{"title": p.title, "authors": p.authors, "url": p.url} for p in relevant],
            }

    # 4) LLM fallback (deterministic)
    if gemini_model is None and openai_client is None:
        raise HTTPException(status_code=500, detail="No LLM API configured.")

    provider = LLM_PROVIDER or ("gemini" if gemini_model else ("openai" if openai_client else ""))

    augmented_prompt = (
        "You are a factual QA assistant.\n"
        "RULES:\n"
        "1) Answer ONLY with facts from the Documents.\n"
        "2) You may infer an inverse family relationship if one direction is explicitly stated.\n"
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
        if provider == "openai" and openai_client:
            c = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Follow the user's RULES exactly."},
                    {"role": "user", "content": augmented_prompt},
                ],
            )
            answer = c.choices[0].message.content
            llm_used = "openai:gpt-3.5-turbo"
        elif provider == "gemini" and gemini_model:
            gen_cfg = {"temperature": 0.0, "top_p": 0.1, "top_k": 1, "max_output_tokens": 512}
            g = gemini_model.generate_content(augmented_prompt, generation_config=gen_cfg)
            answer = g.text
            llm_used = "gemini-2.0-flash"
        else:
            # fallback to whichever exists
            if gemini_model:
                gen_cfg = {"temperature": 0.0, "top_p": 0.1, "top_k": 1, "max_output_tokens": 512}
                g = gemini_model.generate_content(augmented_prompt, generation_config=gen_cfg)
                answer = g.text
                llm_used = "gemini-2.0-flash"
            else:
                c = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "Follow the user's RULES exactly."},
                        {"role": "user", "content": augmented_prompt},
                    ],
                )
                answer = c.choices[0].message.content
                llm_used = "openai:gpt-3.5-turbo"

        return {
            "response": answer,
            "llm_used": llm_used,
            "relevant_documents": [{"title": p.title, "authors": p.authors, "url": p.url} for p in relevant],
        }
    except Exception as e:
        # Keep request successful from the browser POV; show the error
        return {"response": "LLM error", "error": str(e), "llm_used": None, "relevant_documents": [...]}

# ---------------- Similarity helpers (unchanged endpoints) ----------------
class SimilarGistDocsRequest(BaseModel):
    gist_id: str
    document_title: str
    limit: int = 3

class SimilarLLMRequest(BaseModel):
    document_title: str
    limit: int = 3
    llm_type: str = "gemini"

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

    q_emb = embed_query(generated_query)
    relevant = await asyncio.get_event_loop().run_in_executor(
        executor, search_db_for_vectors_sync, db, q_emb, request_body.limit
    )
    if not relevant:
        return {"message": "No similar documents found based on the LLM-generated query."}
    return {"similar_documents": [{"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in relevant]}
# --- debug: what DB am I on & which tables exist?
@app.get("/debug/db")
def debug_db(db: SessionLocal = Depends(get_db)):
    rows = db.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
    )).fetchall()
    from urllib.parse import urlparse
    u = urlparse(os.environ.get("DATABASE_URL", ""))
    return {
        "app_version": app.version,                 # should be 0.6.0 in the file I sent
        "db": u.path.lstrip("/"),
        "host": u.hostname,
        "tables": [r[0] for r in rows],
    }

# --- admin: create any missing tables (safe; no drops)
@app.post("/admin/create-missing-tables")
def create_missing_tables():
    Base.metadata.create_all(bind=engine)
    return {"status": "created_if_missing"}
