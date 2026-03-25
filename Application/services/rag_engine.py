import os
import pdfplumber
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set Groq API client with OpenAI-compatible interface
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

#  Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load and chunk all PDFs from folder
def load_all_pdf_chunks(folder="Compliance_files/"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    sources = []

    for fname in os.listdir(folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(folder, fname)
            with pdfplumber.open(path) as pdf:
                full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])

            chunks = splitter.split_text(full_text)
            all_chunks.extend(chunks)
            sources.extend([fname] * len(chunks))

    return all_chunks, sources


#  Embed and store in FAISS index
def build_vector_store(chunks):
    vectors = embed_model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, vectors


# Retrieve top-k relevant chunks
def retrieve_context(question, index, chunks, sources, k=3):
    q_vec = embed_model.encode([question])
    _, I = index.search(q_vec, k)
    results = [f"[{sources[i]}]\n{chunks[i]}" for i in I[0]]
    return "\n\n".join(results)


#  Query Groq LLM with the retrieved context (OpenAI SDK 1.x style)
def query_compliance(question: str, context: str) -> str:
    prompt = f"""You are a compliance assistant. Answer based only on the provided context below.

Context:
{context}

Question: {question}
Answer:"""

    try:
        response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system", "content": "You are a strict, reliable compliance analyst. Respond only using the context provided."},
        {"role": "user", "content": prompt}
    ],
    stream=False  #  add this if not using streaming
)

        return response.choices[0].message.content
    except Exception as e:
        return f"Error from Groq API: {e}"


# ➕ Add a new uploaded PDF into the vector DB
def add_pdf_to_index(pdf_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join([p.extract_text() or "" for p in pdf.pages])

    new_chunks = splitter.split_text(full_text)
    new_sources = [os.path.basename(pdf_path)] * len(new_chunks)

    new_vectors = embed_model.encode(new_chunks)
    index.add(np.array(new_vectors))

    chunks.extend(new_chunks)
    sources.extend(new_sources)

    return len(new_chunks)


#  Initialize global memory (must come last)
chunks, sources = load_all_pdf_chunks("Compliance_files/")
index, _ = build_vector_store(chunks)
