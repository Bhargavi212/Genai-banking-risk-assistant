from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from Application.services import rag_engine
import shutil

from Application.services.rag_engine import (
    retrieve_context,
    query_compliance,
    add_pdf_to_index,
)

router = APIRouter()

class QARequest(BaseModel):
    question: str

@router.post("/compliance-qa")
def compliance_qa(req: QARequest):
      # import inside to use latest chunks/index
    context = retrieve_context(req.question, rag_engine.index, rag_engine.chunks, rag_engine.sources)
    answer = query_compliance(req.question, context)
    return {"answer": answer}

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    filepath = f"Compliance_files/{file.filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks_added = add_pdf_to_index(filepath)
    return {"message": f"Uploaded and indexed {file.filename}", "chunks": chunks_added}
