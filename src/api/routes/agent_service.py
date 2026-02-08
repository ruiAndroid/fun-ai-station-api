from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.core.config import get_settings

router = APIRouter(prefix="/agent-service", tags=["agent-service"])


def _service_base() -> str:
    settings = get_settings()
    return settings.FUN_AGENT_SERVICE_URL.rstrip("/")


@router.get("/agents")
async def list_agents(request: Request):
    url = f"{_service_base()}/agents"
    trace_id = request.headers.get("x-trace-id")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={"x-trace-id": trace_id} if trace_id else None)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Agent service error: {exc}")
    try:
        data = resp.json()
    except Exception:
        data = resp.text
    return JSONResponse(status_code=resp.status_code, content=data)


@router.post("/agents/{agent}/execute")
async def execute_agent(agent: str, request: Request):
    url = f"{_service_base()}/agents/{agent}/execute"
    trace_id = request.headers.get("x-trace-id")
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"x-trace-id": trace_id} if trace_id else None,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Agent service error: {exc}")

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    if resp.status_code >= 500:
        raise HTTPException(status_code=502, detail="Agent service unavailable")

    return JSONResponse(status_code=resp.status_code, content=data)


@router.get("/config/llm")
async def get_llm_config(request: Request):
    url = f"{_service_base()}/config/llm"
    trace_id = request.headers.get("x-trace-id")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={"x-trace-id": trace_id} if trace_id else None)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Agent service error: {exc}")
    try:
        data = resp.json()
    except Exception:
        data = resp.text
    return JSONResponse(status_code=resp.status_code, content=data)


@router.put("/config/llm")
async def update_llm_config(request: Request):
    url = f"{_service_base()}/config/llm"
    trace_id = request.headers.get("x-trace-id")
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.put(
                url,
                json=payload,
                headers={"x-trace-id": trace_id} if trace_id else None,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Agent service error: {exc}")

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    if resp.status_code >= 500:
        raise HTTPException(status_code=502, detail="Agent service unavailable")

    return JSONResponse(status_code=resp.status_code, content=data)