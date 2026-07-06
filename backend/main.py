"""FastAPI application entry point -- AI video metadata reverse-engineering tool.

v1.3: uses the :class:`AnalysisOrchestrator` to run multi-module analysis
asynchronously in the background and streams real-time progress via WebSocket.
"""
from __future__ import annotations
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.orchestrator import AnalysisOrchestrator, WebSocketManager

app = FastAPI(title="AI Video Metadata Extractor", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------
orchestrator = AnalysisOrchestrator()
ws_manager = WebSocketManager()

# Legacy in-memory state kept in sync by the orchestrator
_analysis_states: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the video file")
    modules: list[str] = Field(
        default=["tech", "visual", "audio", "ai", "source_recovery"],
        description="Enabled module list",
    )


class AnalyzeResponse(BaseModel):
    file_hash: str
    message: str = ""
    cached: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cache_service():
    """Import CacheService lazily (Task 6). Returns None if unavailable."""
    try:
        from backend.services.cache_service import CacheService
        return CacheService()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.3.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(req: AnalyzeRequest):
    """Start a video analysis job.

    Validates the file, computes the SHA-256 hash, checks the cache, and
    launches the pipeline as an async background task.  Returns immediately
    with the file_hash so the frontend can subscribe to the WebSocket.
    """
    from backend.utils.file_utils import validate_video_file, compute_sha256

    ok, err = validate_video_file(req.file_path)
    if not ok:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": err})

    file_hash = compute_sha256(req.file_path)

    # Cache hit → return immediately
    cache = _get_cache_service()
    if cache and cache.exists(file_hash):
        return AnalyzeResponse(file_hash=file_hash, cached=True,
                                message="分析结果已缓存")

    # Initialise state so /status returns something useful immediately
    _analysis_states[file_hash] = {
        "status": "running",
        "file_path": req.file_path,
        "modules": req.modules,
    }

    # Launch the pipeline in the background
    asyncio.create_task(
        _run_analysis_background(req.file_path, req.modules, file_hash)
    )

    return AnalyzeResponse(file_hash=file_hash, message="分析已开始")


@app.get("/analyze/{file_hash}/status")
async def get_analysis_status(file_hash: str):
    """Query the status of an analysis job."""
    state = _analysis_states.get(file_hash)
    if state is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "未找到该分析任务"})
    return state


@app.get("/cache/{file_hash}")
async def get_cached_result(file_hash: str):
    """Read a cached analysis result."""
    cache = _get_cache_service()
    if cache is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"error": "缓存服务未就绪"},
        )
    result = cache.get(file_hash)
    if result is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "缓存未命中"})
    return result


@app.delete("/cache/{file_hash}")
async def delete_cached_result(file_hash: str):
    """Delete a cached analysis result."""
    cache = _get_cache_service()
    if cache is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"error": "缓存服务未就绪"},
        )
    cache.delete(file_hash)
    return {"deleted": True}


@app.get("/export/{file_hash}")
async def export_result(file_hash: str, format: str = "json"):
    """Export analysis result in specified format."""
    cache = _get_cache_service()
    if cache is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"error": "缓存服务未就绪"},
        )
    result = cache.get(file_hash)
    if result is None:
        raise HTTPException(404, "结果不存在")
    from backend.services.export_service import ExportService
    exporter = ExportService()
    # Reconstruct AnalysisResult from cached dict
    from backend.models.schemas import AnalysisResult
    analysis_result = AnalysisResult(**result)
    content = exporter.export_all(analysis_result).get(format)
    if content is None:
        raise HTTPException(400, f"不支持的格式: {format}")
    return {"format": format, "content": content}


@app.get("/cache")
async def list_cached_results():
    """List all cached analysis results."""
    cache = _get_cache_service()
    if cache is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"error": "缓存服务未就绪"},
        )
    items = []
    for entry in cache.list_all():
        items.append({
            "file_hash": entry.get("file_hash", ""),
            "file_path": entry.get("file_path", ""),
            "cached_at": entry.get("cached_at", ""),
        })
    return {"count": len(items), "items": items}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{file_hash}")
async def websocket_progress(websocket: WebSocket, file_hash: str):
    """WebSocket endpoint --- pushes real-time analysis progress.

    The orchestrator broadcasts ``AnalysisProgress`` JSON messages after
    each pipeline step.
    """
    await websocket.accept()
    ws_manager.register(file_hash, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        ws_manager.unregister(file_hash, websocket)


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------

async def _run_analysis_background(
    file_path: str,
    modules: list[str],
    file_hash: str,
) -> None:
    """Run the full pipeline in the background and update in-memory state."""
    try:
        result = await orchestrator.run_analysis(
            file_path, modules, ws_manager=ws_manager,
        )
        _analysis_states[file_hash] = {
            "status": "completed",
            "file_path": file_path,
            "modules": modules,
            "result": result.model_dump(),
        }
    except Exception as exc:
        _analysis_states[file_hash] = {
            "status": "failed",
            "file_path": file_path,
            "modules": modules,
            "error": str(exc),
        }
