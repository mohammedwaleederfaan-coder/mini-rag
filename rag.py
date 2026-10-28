"""RAG service: retrieval + grounded answer generation."""
import logging
import os

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

logger = logging.getLogger("mini_rag")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hands_on_ml_book")
TOP_K = int(os.getenv("TOP_K", "5"))

PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant answering questions about a machine learning book.

Use ONLY the context below. Do NOT add any information, technique, or term
that is not explicitly stated in the context.
If the context only partially answers the question, answer only that part.
If the context does not contain the answer, say:
"I don't know based on the provided book context."

Answer completely: cover every part of the question that the context supports.
Keep the answer clear and educational.

Context:
{context}

Question:
{question}

Answer:"""
)


def get_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def page_of(doc) -> str:
    return str(doc.metadata.get("page_label", doc.metadata.get("page", "Unknown")))


def format_docs(docs) -> str:
    return "\n\n".join(f"[Page {page_of(d)}]\n{d.page_content}" for d in docs)


def build_sources(docs) -> list[dict]:
    """One entry per distinct page, in retrieval order."""
    seen, sources = set(), []
    for d in docs:
        page = page_of(d)
        if page in seen:
            continue
        seen.add(page)
        sources.append({"page": page, "snippet": d.page_content[:200]})
    return sources


class RagService:
    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        self.chain = PROMPT | llm | StrOutputParser()

    def chunk_count(self) -> int:
        return self.vector_store._collection.count()

    def ask(self, question: str) -> dict:
        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(
            {"context": format_docs(docs), "question": question}
        )
        return {"answer": answer.strip(), "sources": build_sources(docs)}