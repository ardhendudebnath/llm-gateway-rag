"""RAG endpoints. Everything is scoped to the caller's tenant: documents, search and answers."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_services, rate_limited
from app.core.container import Services
from app.core.security import Principal
from app.rag.documents import DocumentRecord
from app.rag.ingestion import DocumentTooLargeError, DocumentUpload
from app.rag.parsing import EmptyDocumentError, UnsupportedDocumentError
from app.rag.schemas import AnswerRequest, AnswerResponse, SearchHit, SearchRequest, SearchResponse

router = APIRouter(prefix="/v1/rag", tags=["rag"])


@router.post(
    "/documents",
    response_model=DocumentRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF, Markdown or text file: parse, chunk, embed and index it",
)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None, max_length=200),
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> DocumentRecord:
    ingestion = services.rag.ingestion
    # Read one byte past the limit: enough to reject an oversized upload without reading it all.
    data = await file.read(ingestion.max_bytes + 1)
    try:
        return await ingestion.ingest(
            principal.tenant_id,
            DocumentUpload(
                filename=file.filename or "upload",
                content_type=file.content_type,
                data=data,
                title=title,
            ),
        )
    except DocumentTooLargeError as e:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail=str(e)) from e
    except UnsupportedDocumentError as e:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e)) from e
    except EmptyDocumentError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e


@router.get(
    "/documents", response_model=list[DocumentRecord], summary="Your documents, newest first"
)
async def list_documents(
    principal: Principal = Depends(rate_limited), services: Services = Depends(get_services)
) -> list[DocumentRecord]:
    return await services.rag.documents.list(principal.tenant_id)


@router.get("/documents/{doc_id}", response_model=DocumentRecord)
async def get_document(
    doc_id: str,
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> DocumentRecord:
    record = await services.rag.documents.get(principal.tenant_id, doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such document")
    return record


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> None:
    if not await services.rag.ingestion.delete(principal.tenant_id, doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such document")


@router.post("/search", response_model=SearchResponse, summary="Top-k passages, reranked")
async def search(
    body: SearchRequest,
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> SearchResponse:
    retriever = services.rag.retriever
    hits = await retriever.search(principal.tenant_id, body.query, body.top_k, rerank=body.rerank)
    return SearchResponse(
        query=body.query,
        reranked=body.rerank and retriever.reranker is not None,
        hits=[
            SearchHit(
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                chunk_index=h.chunk.chunk_index,
                page=h.chunk.page,
                heading=h.chunk.heading,
                text=h.chunk.text,
                vector_score=h.vector_score,
                rerank_score=h.rerank_score,
            )
            for h in hits
        ],
    )


@router.post(
    "/answer",
    response_model=AnswerResponse,
    summary="Answer a question from your documents, with numbered citations",
)
async def answer(
    body: AnswerRequest,
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> AnswerResponse:
    return await services.rag.answers.answer(principal, body)
