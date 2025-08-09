# server.py

import os
import requests
import uuid
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text, Column, String, select
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError

# A key dependency for vector embeddings
from pgvector.sqlalchemy import Vector

# --- RAG Specific Dependencies ---
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import openai # Added for a potential OpenAI LLM integration

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Embedding Model ---
embedding_model = None

# --- LLM Models ---
gemini_model = None
# We can add an OpenAI client here for alternative LLM use cases
openai_client = None

# --- Thread Pool Executor for Blocking DB Calls ---
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

# --- Database Configuration ---
DATABASE_URL = os.environ.get("DATABASE_URL")
print('Database URL: ',DATABASE_URL)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # New environment variable for OpenAI

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set.")
    pass

# SQLAlchemy setup
Base = declarative_base()

# Create a database engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create a session local class for database interactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Models ---
class Paper(Base):
    __tablename__ = 'papers'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String)
    abstract = Column(String)
    authors = Column(String)
    url = Column(String)
    
    embedding = Column(Vector(384))

# --- Dependency to get a database session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- FastAPI Application ---
app = FastAPI(
    title="The Aboutness Project Backend API",
    description="API for semantic search and RAG for academic papers.",
    version="0.1.0",
)

# --- CORS Middleware Configuration ---
origins = [
    "https://eohatdan.github.io",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS Middleware ---

@app.on_event("startup")
async def startup_event():
    logger.info("Backend application starting up...")
    
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Successfully connected to the database!")
    except OperationalError as e:
        logger.error(f"Database connection failed on startup: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

    global embedding_model
    try:
        logger.info("Loading Sentence-Transformer model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load embedding model.")

    global gemini_model
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini LLM model initialized.")
        except Exception as e:
            logger.error(f"Error initializing Gemini LLM: {e}")
            gemini_model = None
            logger.warning("Gemini LLM will not be available due to initialization error.")
    else:
        logger.warning("GEMINI_API_KEY not found. Gemini LLM will not be available.")
        gemini_model = None
    
    global openai_client
    if OPENAI_API_KEY:
        try:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized.")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            openai_client = None
            logger.warning("OpenAI client will not be available due to initialization error.")
    else:
        logger.warning("OPENAI_API_KEY not found. OpenAI client will not be available.")
        openai_client = None


    Base.metadata.create_all(bind=engine)
    logger.info("Database schema validated/created.")


@app.get("/")
async def read_root():
    return {"message": "Welcome to The Aboutness Project Backend API!"}

@app.get("/health")
async def health_check(db: SessionLocal = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database_connection": "successful"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database connection failed: {e}")

# --- Ingestion Endpoints ---
# (Existing ingest-data and bulk-load-gist endpoints go here, unchanged from the previous Canvas version)

# --- New Ingestion Endpoint for a Single Gist file ---
class IngestGistFileRequest(BaseModel):
    gist_id: str
    filename: str

@app.post("/ingest-gist-file", status_code=status.HTTP_201_CREATED)
async def ingest_gist_file(request_body: IngestGistFileRequest, db: SessionLocal = Depends(get_db)):
    """
    Fetches a single file from a Gist, embeds its content, and ingests it.
    """
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedding model not loaded.")

    api_url = f"https://api.github.com/gists/{request_body.gist_id}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        gist_data = response.json()
        files = gist_data.get('files', {})
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch Gist metadata: {e}")

    file_info = files.get(request_body.filename)
    if not file_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File '{request_body.filename}' not found in the specified Gist.")

    raw_url = file_info.get('raw_url')
    if not raw_url:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Raw URL not found for file '{request_body.filename}'.")
    
    try:
        content_response = requests.get(raw_url)
        content_response.raise_for_status()
        content = content_response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch content from {raw_url}: {e}")

    text_to_embed = content
    embedding = embedding_model.encode(text_to_embed).tolist()
    
    paper_entry = Paper(
        title=request_body.filename,
        abstract=content,
        authors="Gist User",
        url=raw_url,
        embedding=embedding
    )

    try:
        db.add(paper_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database ingestion failed: {e}")
    
    return {"message": f"Successfully ingested file '{request_body.filename}'."}

# --- RAG Query Endpoints ---
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

@app.post("/query")
async def query_data(request_body: QueryRequest, db: SessionLocal = Depends(get_db)):
    global embedding_model, gemini_model, openai_client

    if embedding_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedding model not loaded.")
    if gemini_model is None and openai_client is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No LLM API configured.")

    if not request_body.query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query parameter is required in the body.")

    logger.info(f"Received query: '{request_body.query}'")

    query_embedding = embedding_model.encode([request_body.query])[0].tolist()
    logger.info(f"Query embedding generated. Length: {len(query_embedding)}")

    relevant_papers = await asyncio.get_event_loop().run_in_executor(
        executor, search_db_for_vectors_sync, db, query_embedding, request_body.limit
    )
    
    logger.info(f"Found {len(relevant_papers)} relevant documents from database.")
    
    if not relevant_papers:
        return {"response": "I could not find any relevant documents in the knowledge base to answer your question."}

    documents_text = "\n\n".join([
        f"Title: {p.title}\nAuthors: {p.authors}\nAbstract: {p.abstract}" for p in relevant_papers
    ])

    augmented_prompt = (
        "Based on the following documents, answer the user's question. "
        "If the documents do not contain enough information, state that clearly and concisely. "
        "Do not use external knowledge.\n\n"
        "Documents:\n"
        "```\n"
        f"{documents_text}\n"
        "```\n\n"
        "User's Question:\n"
        f"{request_body.query}"
    )
    
    logger.info("Sending augmented prompt to LLM...")
    try:
        llm_response = None
        if gemini_model:
            gemini_response = gemini_model.generate_content(augmented_prompt)
            llm_response = gemini_response.text
        elif openai_client:
            # Placeholder for OpenAI generation
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": augmented_prompt}]
            )
            llm_response = completion.choices[0].message.content
        
        logger.info("LLM response received.")
        return {"response": llm_response, "relevant_documents": [
            {"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in relevant_papers
        ]}
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM generation failed: {e}")

@app.post("/find-similar-gist-docs")
async def find_similar_gist_docs(request_body: SimilarGistDocsRequest, db: SessionLocal = Depends(get_db)):
    """
    Finds documents within a specific Gist that are semantically similar to a given document title.
    """
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedding model not loaded.")

    # 1. Find the embedding of the document_title within the specified Gist
    target_paper = db.query(Paper).filter(
        Paper.title == request_body.document_title,
        Paper.url.like(f"https://gist.githubusercontent.com/%/%/raw/{request_body.document_title}")
    ).first()

    if not target_paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document with title '{request_body.document_title}' not found in the specified Gist.")

    target_embedding = target_paper.embedding
    
    # 2. Search for similar documents, excluding the target itself, and only within the same gist
    relevant_papers = await asyncio.get_event_loop().run_in_executor(
        executor, search_db_for_vectors_filtered_sync, db, target_embedding, request_body.limit + 1, request_body.document_title
    )
    
    # Filter out the target paper from the results
    filtered_results = [p for p in relevant_papers if p.title != request_body.document_title]

    if not filtered_results:
        return {"message": f"No similar documents found in the Gist for '{request_body.document_title}'."}

    return {"similar_documents": [
        {"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in filtered_results
    ]}

@app.post("/find-similar-llm")
async def find_similar_llm(request_body: SimilarLLMRequest, db: SessionLocal = Depends(get_db)):
    """
    Uses an LLM to generate a search query from a document title and then finds similar articles.
    """
    global embedding_model, gemini_model, openai_client

    if embedding_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedding model not loaded.")
    if gemini_model is None and openai_client is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No LLM API configured.")

    # 1. Use LLM to generate a semantic query from the document title
    llm_query_prompt = (
        f"Generate a semantic search query for academic papers about the topic of: '{request_body.document_title}'. "
        "The query should be a single, concise sentence."
    )
    
    try:
        llm_response = None
        if request_body.llm_type == "gemini" and gemini_model:
            gemini_response = gemini_model.generate_content(llm_query_prompt)
            llm_response = gemini_response.text
        elif request_body.llm_type == "openai" and openai_client:
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": llm_query_prompt}]
            )
            llm_response = completion.choices[0].message.content
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid LLM type '{request_body.llm_type}' or API not configured.")

        if not llm_response:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM failed to generate a query.")
            
        generated_query = llm_response.strip().strip('"')

    except Exception as e:
        logger.error(f"Error generating LLM query: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM query generation failed: {e}")

    # 2. Use the generated query to perform a semantic search
    query_embedding = embedding_model.encode([generated_query])[0].tolist()
    relevant_papers = await asyncio.get_event_loop().run_in_executor(
        executor, search_db_for_vectors_sync, db, query_embedding, request_body.limit
    )

    if not relevant_papers:
        return {"message": "No similar documents found based on the LLM-generated query."}

    return {"similar_documents": [
        {"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in relevant_papers
    ]}

# Synchronous helper function for filtered vector search
def search_db_for_vectors_filtered_sync(db: SessionLocal, query_embedding_list: List[float], limit: int, exclude_title: str) -> List[Paper]:
    try:
        results = db.query(Paper).filter(Paper.title != exclude_title).order_by(
            Paper.embedding.l2_distance(query_embedding_list)
        ).limit(limit).all()
        return results
    except Exception as e:
        logger.error(f"Error searching filtered database for vectors: {e}")
        return []

# Synchronous function to be run in ThreadPoolExecutor
def search_db_for_vectors_sync(db: SessionLocal, query_embedding_list: List[float], limit: int) -> List[Paper]:
    try:
        results = db.query(Paper).order_by(
            Paper.embedding.l2_distance(query_embedding_list)
        ).limit(limit).all()
        return results
    except Exception as e:
        logger.error(f"Error searching database for vectors: {e}")
        return []
