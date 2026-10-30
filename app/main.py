"""FastAPI-приложение Grand Mobile Support Copilot.

Запуск: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.triage import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.pipeline import TriagePipeline

setup_logging()
log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "pipeline", None) is None:
        app.state.pipeline = TriagePipeline()
    yield


app = FastAPI(title="Grand Mobile Support Copilot", version=get_settings().app_version, lifespan=lifespan)
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # некорректный JSON / отсутствующие поля → 422 с понятным описанием, сервис продолжает работать
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
    return JSONResponse({"error": "invalid_request", "details": errors}, status_code=422)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    log.exception("unhandled error: %s", exc)
    return JSONResponse({"error": "internal_error"}, status_code=500)
