"""
src/rag.py
RAG system: document loading, vector store creation/loading, and retrieval.
Uses ChromaDB as the vector store and HuggingFace embeddings (no API key needed).
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

# ── Embedding model (free, runs locally, no API key required) ─────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 3


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store() -> Chroma:
    """
    Load documents from data/documents/, chunk them, embed, and persist to
    data/vector_store/. Should be run once; subsequent runs load the existing store.
    """
    print("[RAG] Loading documents from", DOCUMENTS_DIR)
    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"[RAG] Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] Split into {len(chunks)} chunks.")

    embeddings = _get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    vectorstore.persist()
    print(f"[RAG] Vector store persisted to {VECTOR_STORE_DIR}")
    return vectorstore


def load_vector_store() -> Chroma:
    """Load the existing persisted vector store from disk."""
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings,
    )
    return vectorstore


def get_vector_store() -> Chroma:
    """
    Return the vector store, building it if it does not yet exist.
    This is the main entry point used by the rest of the application.
    """
    # Check if vector store already exists
    if VECTOR_STORE_DIR.exists() and any(VECTOR_STORE_DIR.iterdir()):
        print("[RAG] Loading existing vector store.")
        return load_vector_store()
    else:
        print("[RAG] Vector store not found – building for the first time.")
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        return build_vector_store()


def retrieve(query: str, vectorstore: Chroma = None, k: int = TOP_K_RESULTS) -> str:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query:       The user's natural-language question.
        vectorstore: A Chroma instance. If None, the store is loaded from disk.
        k:           Number of chunks to retrieve.

    Returns:
        A single string concatenating the retrieved passages, ready to be
        injected into the LLM prompt as context.
    """
    if vectorstore is None:
        vectorstore = get_vector_store()

    results: List = vectorstore.similarity_search(query, k=k)

    if not results:
        return "No relevant information found in the knowledge base."

    passages = []
    for i, doc in enumerate(results, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        passages.append(f"[Source {i}: {source}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(passages)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    vs = get_vector_store()
    test_query = "What audio features are most important for song popularity?"
    print("\nQuery:", test_query)
    print("\nRetrieved context:\n", retrieve(test_query, vs))
