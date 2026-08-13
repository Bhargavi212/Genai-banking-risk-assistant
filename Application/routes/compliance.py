from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from Application.services import rag_engine
from Application.services.rag_engine import (
    add_pdf_to_index,
    query_compliance,
    retrieve_context,
)

router = APIRouter(
    prefix="/compliance",
    tags=["Compliance"],
)

UPLOAD_DIR = Path("Compliance_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QARequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Compliance-related question to answer using retrieved document context.",
    )


@router.post("/qa")
def compliance_qa(req: QARequest):
    try:
        context = retrieve_context(
            req.question,
            rag_engine.index,
            rag_engine.chunks,
            rag_engine.sources,
        )

        if not context:
            return {
                "question": req.question,
                "answer": "No relevant compliance context was found in the indexed documents.",
                "retrieval_status": "no_context_found",
            }

        answer = query_compliance(req.question, context)

        return {
            "question": req.question,
            "answer": answer,
            "retrieval_status": "success",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the compliance question.",
        ) from exc


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    safe_filename = Path(file.filename).name
    filepath = UPLOAD_DIR / safe_filename

    try:
        with filepath.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks_added = add_pdf_to_index(str(filepath))

        return {
            "message": f"Uploaded and indexed {safe_filename}",
            "filename": safe_filename,
            "chunks_added": chunks_added,
        }

    except Exception as exc:
        if filepath.exists():
            filepath.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Unable to upload and index the PDF.",
        ) from exc

    finally:
        await file.close()
