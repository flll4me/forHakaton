"""HTTP-эндпоинты: POST /triage (контракт Grand Mobile), GET /health, GET /version."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.models.schemas import TriageRequest, TriageResponse

router = APIRouter()


def _pipeline(request: Request):
    return request.app.state.pipeline


@router.post("/triage", response_model=TriageResponse, response_model_exclude_none=False)
def triage(req: TriageRequest, request: Request, debug: bool = Query(False, description="добавить диагностику в поле _debug")):
    resp, dbg = _pipeline(request).triage(req)
    if debug:
        return JSONResponse({**resp.model_dump(), "_debug": dbg.model_dump()})
    return resp


@router.get("/health")
def health(request: Request):
    p = getattr(request.app.state, "pipeline", None)
    if p is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok", "components": p.status(), "load_seconds": p.load_s}


@router.get("/version")
def version(request: Request):
    p = getattr(request.app.state, "pipeline", None)
    meta = p.classifier.meta if p else {}
    s = get_settings()
    return {"app_version": s.app_version, "classifier": meta, "embedding_model": s.embedding_model if s.embeddings_enabled else None,
            "llm_provider": s.llm_provider, "llm_model": s.llm_model or None}
