import os
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

COMPLIANCE_DIR = Path("Compliance_files")
COMPLIANCE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3

embed_model = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# LLM Client
# ---------------------------------------------------------

def get_llm_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to your environment or .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


# ---------------------------------------------------------
# PDF Loading & Chunking
# ---------------------------------------------------------

def load_all_pdf_chunks(folder=COMPLIANCE_DIR):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    all_chunks = []
    all_sources = []

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for pdf_path in folder.glob("*.pdf"):
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):

                text = page.extract_text() or ""

                if not text.strip():
                    continue

                page_chunks = splitter.split_text(text)

                for chunk_number, chunk in enumerate(
                    page_chunks,
                    start=1,
                ):
                    all_chunks.append(chunk)

                    all_sources.append(
                        {
                            "filename": pdf_path.name,
                            "page": page_number,
                            "chunk": chunk_number,
                        }
                    )

    return all_chunks, all_sources


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

def build_vector_store(chunks):
    if not chunks:
        return None

    vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    dimension = vectors.shape[1]

    # Inner product on normalized embeddings = cosine similarity
    vector_index = faiss.IndexFlatIP(dimension)

    vector_index.add(
        vectors.astype("float32")
    )

    return vector_index


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_context(
    question,
    index,
    chunks,
    sources,
    k=DEFAULT_TOP_K,
):
    if not question.strip():
        return ""

    if index is None or not chunks:
        return ""

    k = min(k, len(chunks))

    question_vector = embed_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        question_vector,
        k,
    )

    retrieved_chunks = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):
        if idx < 0 or idx >= len(chunks):
            continue

        source = sources[idx]

        retrieved_chunks.append(
            {
                "text": chunks[idx],
                "source": source,
                "score": float(score),
            }
        )

    context_sections = []

    for item in retrieved_chunks:
        source = item["source"]

        context_sections.append(
            f"[Source: {source['filename']} | "
            f"Page: {source['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(context_sections)


# ---------------------------------------------------------
# Compliance Question Answering
# ---------------------------------------------------------

def query_compliance(
    question: str,
    context: str,
) -> str:

    if not context.strip():
        return (
            "I could not find relevant information "
            "in the indexed compliance documents."
        )

    client = get_llm_client()

    system_prompt = """
You are a banking compliance research assistant.

Follow these rules carefully:

1. Answer only using the supplied document context.
2. Do not invent regulations, requirements, or facts.
3. If the context does not contain enough information,
   clearly state that the available documents are insufficient.
4. When possible, mention the source document and page.
5. Keep the answer concise and factual.
"""

    user_prompt = f"""
Retrieved Context:

{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        stream=False,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# Add Newly Uploaded PDF
# ---------------------------------------------------------

def add_pdf_to_index(pdf_path):
    global index
    global chunks
    global sources

    pdf_path = Path(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    new_chunks = []
    new_sources = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            page_chunks = splitter.split_text(text)

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1,
            ):

                new_chunks.append(chunk)

                new_sources.append(
                    {
                        "filename": pdf_path.name,
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )

    if not new_chunks:
        return 0

    new_vectors = embed_model.encode(
        new_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    if index is None:
        dimension = new_vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)

    index.add(new_vectors)

    chunks.extend(new_chunks)
    sources.extend(new_sources)

    return len(new_chunks)


# ---------------------------------------------------------
# Initialize Knowledge Base
# ---------------------------------------------------------

chunks, sources = load_all_pdf_chunks()

index = build_vector_store(chunks)
