# server.py

import os
import re
import uuid
import logging
import asyncio
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, text, Column, String, or_, insert
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError

from pgvector.sqlalchemy import Vector

from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import openai

# --- Relation & gender maps (define once) ---
REL_MAP = {
    # direct
    "parent":"parent", "mother":"parent", "father":"parent",
    "spouse":"spouse", "husband":"spouse", "wife":"spouse", "partner":"spouse",
    "child":"child", "children":"child", "son":"child", "daughter":"child",
    "sibling":"sibling", "siblings":"sibling", "brother":"sibling", "sister":"sibling",
    # grand* aliases
    "grandparent":"grandparent", "grandparents":"grandparent",
    "grandfather":"grandparent", "grandmother":"grandparent",
    "grandchild":"grandchild", "grandchildren":"grandchild",
}

GENDER_HINT = {
    "father":"M", "mother":"F",
    "grandfather":"M", "grandmother":"F",
    "husband":"M", "wife":"F",
    "son":"M", "daughter":"F",
    "brother":"M", "sister":"F",
}

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tap2")

# ---------------- Env / Config ----------------
DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var not set.")

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
    embedding = Column(Vector(384))

# Auto-parsed facts (cleared on ingest)
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
    note   = Column(String, default="")
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

# ---------------- App ----------------
app = FastAPI(
    title="The Aboutness Project Backend API",
    description="API for semantic search and RAG",
    version="0.8.0",
)

# CORS: explicit origin + OPTIONS catch-all
ALLOWED_ORIGINS = ["https://eohatdan.github.io"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
    max_age=86400,
)

@app.options("/{path:path}")
def preflight_cors(path: str, response: Response):
    origin = ALLOWED_ORIGINS[0]
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
    response.headers["Access-Control-Max-Age"] = "86400"
    return Response(status_code=204)

# ---------------- Globals ----------------
embedding_model = None
gemini_model = None
openai_client = None
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

# ---------------- Utils ----------------
def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

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

# E5 retrieval-tuned embeddings
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
    # A is the father of B
    (re.compile(fr"({NAME})(?:{IS_THE}|{WAS_THE})(father|mother|son|daughter|husband|wife|spouse|parent|child){OF}({NAME})", re.IGNORECASE), "dir"),
    # A is B’s father  (apostrophe REQUIRED)
    (re.compile(fr"({NAME})(?:{IS}|{WAS})({NAME})[’']s\s+(father|mother|son|daughter|husband|wife|spouse|parent|child)", re.IGNORECASE), "poss"),
    # A’s father is B  (apostrophe REQUIRED)
    (re.compile(fr"({NAME})[’']s\s+(father|mother|son|daughter|husband|wife|spouse|parent|child){IS}({NAME})", re.IGNORECASE), "inv"),
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

PERSON = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}$")
BAD_TOKENS = (
    "High School","University","College","WV","San Jose","Huntington","California","West Virginia",
    "degree","bachelor","masters","master’s","information","sciences","born","on","July","August",
    "County","State","Road","Street","Avenue","Vin son","Vinson"
)
PRONOUNS = {"I","You","He","She","We","They","His","Her","Their","Our","Your","Hers","Him"}
import re

# Optional: a few common synonyms -> canonical keys you already use in REL_MAP
_REL_SYNONYMS = {
    "granddad": "grandfather",
    "grand-dad": "grandfather",
    "grandpa": "grandfather",
    "grandmom": "grandmother",
    "grandma": "grandmother",
    "mum": "mother",
    "dad": "father",
    "bro": "brother",
    "sis": "sister",
    "kids": "children",
    "child": "children",   # treat singular as children for list-y queries
}

def _canon_rel(rel: str) -> str:
    rel = rel.strip().lower()
    if rel in _REL_SYNONYMS:
        rel = _REL_SYNONYMS[rel]
    # if your REL_MAP uses singular keys like 'child' not 'children', flip here
    return rel

# --- minimal kinship classifier ---
def classify_question(q: str):
    """
    Returns (is_kinship: bool, subject: str|None, relation: str|None, remainder: str|None)
    Supports:
      - "Who is Matthew Clarke's grandfather?"
      - "Who is the grandfather of Matthew Clarke?"
      - "What company did Matthew Clarke's grandfather work for?"
    """
    import re
    original = q
    ql = re.sub(r"\s+", " ", (q or "").strip().lower())

    # who is the <rel> of <name>
    m = re.search(r"\bwho\s+is\s+(?:the\s+)?(?P<rel>[a-z\- ]+)\s+of\s+(?P<name>[a-z .'\-]+)", ql)
    if not m:
        # who is <name>'s <rel>
        m = re.search(r"\bwho\s+is\s+(?P<name>[a-z .'\-]+?)'s\s+(?P<rel>[a-z\- ]+)", ql)
    if not m:
        # composite: <name>'s <rel> ...
        m2 = re.search(r"(?P<name>[a-z .'\-]+?)'s\s+(?P<rel>[a-z\- ]+)", ql)
        if not m2:
            return (False, None, None, None)
        subject = m2.group("name").strip(" ?.,")
        relation = m2.group("rel").strip()
        start, end = m2.span()
        remainder = (original[:start] + original[end:]).strip(" ?.,")
        return (True, subject, relation, remainder or None)

    subject = m.group("name").strip(" ?.,")
    relation = m.group("rel").strip()
    start, end = m.span()
    remainder = (original[:start] + original[end:]).strip(" ?.,")
    return (True, subject, relation, remainder or None)
# --- end classifier ---


def is_person(name: str) -> bool:
    n = normalize_name(name)
    if n in PRONOUNS:
        return False
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
            if is_person(child_name) and is_person(parent):
                edges.extend(rel_to_edges(child_name, "child", parent))
    return edges

def extract_relations(text: str) -> Tuple[List[Tuple[str, str, str]], Dict[str, List[str]]]:
    triples: List[Tuple[str,str,str]] = []
    nicknames: Dict[str, List[str]] = {}

    for pat, kind in REL_PATTERNS:
        for m in pat.finditer(text):
            if kind == "dir":
                a0, r, b0 = m.group(1), m.group(2), m.group(3)
            elif kind == "poss":
                a0, b0, r = m.group(1), m.group(2), m.group(3)
            elif kind == "inv":
                b0, r, a0 = m.group(1), m.group(2), m.group(3)
            for (rel, a, b) in rel_to_edges(a0, r, b0):
                if is_person(a) and is_person(b):
                    triples.append((rel, a, b))

    for m in CHILDREN_BLOCK.finditer(text):
        parent = normalize_name(m.group(1))
        block  = m.group(2)
        triples.extend(_parse_children_lines(parent, block))

    for m in SINGLE_CHILD.finditer(text):
        parent = normalize_name(m.group(1))
        child  = normalize_name(m.group(2))
        if is_person(child) and is_person(parent):
            triples.extend(rel_to_edges(child, "child", parent))

    for m in NICK_BOTH.finditer(text):
        alias = normalize_name(m.group(1))
        a = normalize_name(m.group(2))
        b = normalize_name(m.group(3))
        if is_person(a): nicknames.setdefault(alias, []).append(a)
        if is_person(b): nicknames.setdefault(alias, []).append(b)
    for m in NICK_ONE.finditer(text):
        alias = normalize_name(m.group(1))
        full  = normalize_name(m.group(2))
        if is_person(full): nicknames.setdefault(alias, []).append(full)

    for k, v in list(nicknames.items()):
        nicknames[k] = sorted(set(v))
    return triples, nicknames

# ---------------- Mini Graph (fallback reasoning) ----------------
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

from sqlalchemy import text

def _manual_person_ids(db, name: str):
    """
    Resolve a name to 0..N manual_persons.id.
    Supports exact full-name and first+last-only matches (first_last_norm).
    """
    rows = db.execute(text("""
        WITH q AS (SELECT :name AS n)
        SELECT id
        FROM manual_persons
        WHERE lower(full_name) = lower((SELECT n FROM q))
        UNION
        SELECT id
        FROM manual_persons
        WHERE first_last_norm(full_name) = first_last_norm((SELECT n FROM q))
    """), {"name": name}).fetchall()
    return [r[0] for r in rows]

def _manual_spouses_by_name(db, name: str):
    """
    Return list of (spouse_fullname, source) using manual_unions.
    """
    ids = _manual_person_ids(db, name)
    if not ids:
        return []
    rows = db.execute(text("""
        WITH x AS (SELECT unnest(:ids::uuid[]) AS id)
        SELECT p.full_name AS spouse, u.source
          FROM manual_unions u
          JOIN x ON x.id = u.spouse1
          JOIN manual_persons p ON p.id = u.spouse2
        UNION ALL
        SELECT p.full_name AS spouse, u.source
          FROM manual_unions u
          JOIN x ON x.id = u.spouse2
          JOIN manual_persons p ON p.id = u.spouse1
    """), {"ids": ids}).fetchall()
    return rows


def graph_all_people(G: 'MiniGraph') -> Set[str]:
    names = set(G.parents.keys()) | set(G.children.keys()) | set(G.spouses.keys())
    for kids in G.parents.values(): names |= kids
    for kids in G.children.values(): names |= kids
    for sps in G.spouses.values(): names |= sps
    return names

def answer_from_graph(G: 'MiniGraph', person: str, rel: str):
    ans, srcs = set(), set()
    if rel in ("father","mother","parent"):
        for p in G.get_parents(person):
            ans.add(p)
            srcs |= G.sources_for("parent_of", p, person)
    elif rel in ("son","daughter","child"):
        for c in G.get_children(person):
            ans.add(c)
            srcs |= G.sources_for("parent_of", person, c)
    elif rel == "spouse":
        for s in G.get_spouses(person):
            ans.add(s)
            srcs |= G.sources_for("spouse_of", person, s)
    elif rel == "grandparent":
        for gp in G.get_grandparents(person):
            ans.add(gp)
    return ans, srcs

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
        token = token.lower()
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

# Require apostrophe for possessive; also support "the father of X"
Q_PATTERNS = [
    re.compile(r"who\s+is\s+(.+?)[’']s\s+([a-z]+)\??", re.IGNORECASE),
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

# ---------------- DI ----------------
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
        raise HTTPException(status_code=500, detail=f"DB connection failed: {e}")

    global embedding_model
    embedding_model = SentenceTransformer("intfloat/e5-small-v2")

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

    Base.metadata.create_all(bind=engine)
    logger.info("Startup OK.")

# ---------------- Basic endpoints ----------------
@app.get("/")
async def read_root():
    return {"message": "Welcome to TAP2 Backend!"}

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
    db.execute(text("DELETE FROM papers"))
    db.execute(text("DELETE FROM relations"))
    db.execute(text("DELETE FROM spouses"))
    db.execute(text("DELETE FROM nicknames"))
    db.commit()

@app.post("/bulk-load-gist", status_code=status.HTTP_201_CREATED)
async def bulk_load_gist(gist_id: str, db: SessionLocal = Depends(get_db)):
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

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

    paper_rows, rel_rows, sp_rows, nk_rows = [], [], [], []

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
            emb = embed_doc(chunk)

            paper_rows.append({
                "id": str(uuid.uuid4()),
                "title": title,
                "abstract": chunk,
                "authors": "Gist User",
                "url": raw_url,
                "embedding": emb,
            })

            triples, nicks = extract_relations(chunk)
            for (rel, a, b) in triples:
                if rel == "parent_of":
                    rel_rows.append({"id": str(uuid.uuid4()), "parent": a, "child": b, "source": title})
                elif rel == "spouse_of":
                    sp_rows.append({"id": str(uuid.uuid4()), "a": a, "b": b, "source": title})
            for alias, fulls in nicks.items():
                for full in fulls:
                    nk_rows.append({"id": str(uuid.uuid4()), "alias": alias, "fullname": full, "source": title})

    # explicit inserts (no empty rows)
    if paper_rows:
        db.execute(insert(Paper.__table__), paper_rows)
    if rel_rows:
        db.execute(insert(Relation.__table__), rel_rows)
    if sp_rows:
        db.execute(insert(Spouse.__table__), sp_rows)
    if nk_rows:
        db.execute(insert(Nickname.__table__), nk_rows)
    db.commit()

    counts = db.execute(text("""
      SELECT
        (SELECT COUNT(*) FROM papers)    AS papers,
        (SELECT COUNT(*) FROM relations) AS relations,
        (SELECT COUNT(*) FROM spouses)   AS spouses,
        (SELECT COUNT(*) FROM nicknames) AS nicknames
    """)).mappings().first()

    return {
        "message": f"Loaded gist {gist_id}",
        "papers": counts["papers"],
        "relations": counts["relations"],
        "spouses": counts["spouses"],
        "nicknames": counts["nicknames"]
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
    paper_rows, rel_rows, sp_rows, nk_rows = [], [], [], []
    for i, chunk in enumerate(chunks):
        title = f"{request_body.filename}::chunk_{i:03d}"
        emb = embed_doc(chunk)
        paper_rows.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "abstract": chunk,
            "authors": "Gist User",
            "url": raw_url,
            "embedding": emb,
        })
        triples, nicks = extract_relations(chunk)
        for (rel, a, b) in triples:
            if rel == "parent_of":
                rel_rows.append({"id": str(uuid.uuid4()), "parent": a, "child": b, "source": title})
            elif rel == "spouse_of":
                sp_rows.append({"id": str(uuid.uuid4()), "a": a, "b": b, "source": title})
        for alias, fulls in nicks.items():
            for full in fulls:
                nk_rows.append({"id": str(uuid.uuid4()), "alias": alias, "fullname": full, "source": title})

    if paper_rows:
        db.execute(insert(Paper.__table__), paper_rows)
    if rel_rows:
        db.execute(insert(Relation.__table__), rel_rows)
    if sp_rows:
        db.execute(insert(Spouse.__table__), sp_rows)
    if nk_rows:
        db.execute(insert(Nickname.__table__), nk_rows)
    db.commit()

    return {"message": f"Ingested {len(paper_rows)} chunks from '{request_body.filename}'."}

# ---------------- SQL-first answering ----------------
def sql_person_candidates(db, person: str):
    rows = db.execute(text("""
        SELECT fullname FROM manual_nicknames WHERE alias = :p
        UNION SELECT :p AS fullname
    """), {"p": person}).fetchall()
    cands = {r[0] for r in rows}

    rows = db.execute(text("""
        SELECT fullname FROM nicknames WHERE alias = :p
    """), {"p": person}).fetchall()
    cands.update({r[0] for r in rows})

    tokens = person.lower().split()
    if tokens:
        name_rows = db.execute(text("""
    -- Names from lightweight extracted tables (if ever populated)
    SELECT parent  AS fullname FROM relations
    UNION SELECT child       FROM relations
    UNION SELECT a           FROM spouses
    UNION SELECT b           FROM spouses

    -- Names from new normalized manual model
    UNION SELECT parent FROM manual_relations_v
    UNION SELECT child  FROM manual_relations_v
    UNION
    SELECT p1.full_name FROM manual_unions u
      JOIN manual_persons p1 ON p1.id = u.spouse1
    UNION
    SELECT p2.full_name FROM manual_unions u
      JOIN manual_persons p2 ON p2.id = u.spouse2
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
                "SELECT parent, source FROM manual_relations_v WHERE child = :c",
                "SELECT parent, source FROM relations WHERE child = :c",
                {"c": c}
            )
            answers += [r[0] for r in rows]; sources += [r[1] for r in rows]

    elif rel in ("son","daughter","child"):
        for c in cands:
            rows = fetch_manual_then_auto(
                "SELECT child, source FROM manual_relations_v WHERE parent = :c",
                "SELECT child, source FROM relations WHERE parent = :c",
                {"c": c}
            )
            answers += [r[0] for r in rows]; sources += [r[1] for r in rows]

    elif rel == "spouse":
        for c in cands:
            # Try the normalized manual schema first
            rows_m = db.execute(text("""
                WITH ids AS (
                  SELECT id FROM manual_persons
                   WHERE lower(full_name) = lower(:c)
                  UNION
                  SELECT id FROM manual_persons
                   WHERE first_last_norm(full_name) = first_last_norm(:c)
                )
                SELECT p.full_name AS spouse, u.source
                FROM manual_unions u
                JOIN ids i ON (i.id = u.spouse1 OR i.id = u.spouse2)
                JOIN manual_persons p
                  ON p.id = CASE WHEN u.spouse1 = i.id THEN u.spouse2 ELSE u.spouse1 END
            """), {"c": c}).fetchall()
    
            if rows_m:
                used_manual_any = True
                rows = rows_m
            else:
                # Fallback to the lightweight extracted table (if you still use it)
                rows = db.execute(text("""
                    SELECT b, source FROM spouses WHERE a = :c
                    UNION ALL
                    SELECT a, source FROM spouses WHERE b = :c
                """), {"c": c}).fetchall()

        answers += [r[0] for r in rows]
        sources += [r[1] for r in rows]


    elif rel in ("grandparent",):
        for c in cands:
            rows_m = db.execute(text("""
                SELECT DISTINCT
                       gp.parent AS person,
                       p.source  AS src1,
                       gp.source AS src2
                FROM manual_relations_v p               -- p: (parent -> child)
                JOIN manual_relations_v gp              -- gp: (grandparent -> parent)
                  ON lower(gp.child) = lower(p.parent)
                WHERE lower(p.child) = lower(:c)
            """), {"c": c}).fetchall()

        if rows_m:
            used_manual_any = True
            for person, s1, s2 in rows_m:
                answers.append(person)
                sources.append(_merge_src(s1, s2))

    elif rel in ("grandchild",):
        for c in cands:
            rows_m = db.execute(text("""
                SELECT DISTINCT
                       gc.child  AS person,
                       p.source  AS src1,
                       gc.source AS src2
                FROM manual_relations_v p               -- p: (parent -> child) where parent = candidate
                JOIN manual_relations_v gc              -- gc: (child -> grandchild)
                  ON lower(gc.parent) = lower(p.child)
                WHERE lower(p.parent) = lower(:c)
            """), {"c": c}).fetchall()

        if rows_m:
            used_manual_any = True
            for person, s1, s2 in rows_m:
                answers.append(person)
                sources.append(_merge_src(s1, s2))


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
        
def _merge_src(s1, s2):
    parts = { (s1 or "").strip(), (s2 or "").strip() }
    parts.discard("")
    return "; ".join(sorted(parts)) or "manual"

# ---------------- Query endpoint ----------------
class QueryRequest(BaseModel):
    query: str
    limit: int = 5

from fastapi import HTTPException
from sqlalchemy import text
from typing import Dict, Any

@app.post("/query")
def query(request_body: "QueryRequest") -> Dict[str, Any]:  # keep your existing Pydantic model
    user_q = (getattr(request_body, "query", None) or "").strip()
    if not user_q:
        raise HTTPException(status_code=400, detail="Query parameter is required in the body.")

    # ---- Stage 1: SQL kinship gate ------------------------------------------
    is_kin, subject, relation, remainder = classify_question(user_q)
    known_fact = ""
    sql_sources = []
    stage = None

    with engine.begin() as db:
        if is_kin and subject and relation:
            names, srcs, prov = answer_with_sql(db, subject, relation)
            if names:
                if not remainder or not remainder.strip():
                    # Pure kinship -> return immediately
                    return {
                        "response": ", ".join(names),
                        "stage": "sql_only",
                        "sources": srcs,
                        "provenance": prov,
                        "relevant_documents": [],
                    }
                # Composite question -> carry forward a known fact for Stage 2
                known_fact = f"{subject}'s {relation} is {', '.join(names)}."
                sql_sources = srcs

        # ---- Stage 2: RAG over stories only (rag_story_docs_v) ---------------
        effective_q = user_q if not known_fact else f"{user_q}\n\nKnown facts:\n- {known_fact}"
        try:
            q_vec = embed_query(effective_q)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

        k = 8
        rows = db.execute(text("""
           SELECT
              id,
              title,
              COALESCE(content, abstract, '') AS content,
              url,
              1.0 - (embedding <=> (%(qvec)s)::vector) AS score
            FROM rag_story_docs_v
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> (%(qvec)s)::vector
            LIMIT %(k)s;
        """), {"qvec": q_vec, "k": k}).mappings().all()

    hits = [
        {
            "doc_id": r["id"],
            "title": r["title"],
            "content": r["content"] or "",
            "url": r["url"],
            "score": float(r["score"]),
        }
        for r in rows
    ]

    if not hits:
        # No documents to support an answer
        return {
            "response": "I don't know based on the provided documents.",
            "stage": "none",
            "sources": sql_sources,
            "relevant_documents": [],
            "augmented_query": effective_q,
        }

    stage = "sql+rag" if known_fact else "rag"

    # Build the exact same augmented prompt style you showed
    top = hits[:5]
    documents_text = "\n\n".join(
        f"Title: {h['title']}\n{(h['content'] or '')[:1200]}"
        for h in top
    )

    augmented_prompt = (
        "You are a factual QA assistant.\n"
        "\"RULES:\"\n"
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
        + (f"\nKnown facts:\n- {known_fact}\n" if known_fact else "")
    )

    # ---- Stage 3: LLM composition (same pattern you already use) ------------
    answer_text = "I don't know based on the provided documents."
    llm_used = None

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
            answer_text = c.choices[0].message.content
            llm_used = "openai:gpt-3.5-turbo"
        else:
            # If you support other providers (e.g., Gemini), add those branches here.
            llm_used = "none"
            # Leave the default 'I don't know...' if no provider is configured.
    except Exception as e:
        answer_text = f"I don't know based on the provided documents. (LLM error: {e})"
        llm_used = "error"

    return {
        "response": answer_text,
        "stage": stage,
        "llm_used": llm_used,
        "sources": sql_sources,
        "relevant_documents": [
            {"title": h["title"], "url": h["url"], "score": h["score"]} for h in top
        ],
        "augmented_query": effective_q,
    }
