"""FastAPI 应用入口 — AI短视频元数据逆向提取工具后端。"""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI Video Metadata Extractor", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 简易内存状态（后续 task 集成 orchestrator）
_analysis_states: dict[str, dict] = {}
_ws_connections: dict[str, list[WebSocket]] = {}


class AnalyzeRequest(BaseModel):
    file_path: str = Field(..., description="视频文件绝对路径")
    modules: list[str] = Field(
        default=["tech", "visual", "audio", "ai", "source_recovery"],
        description="启用的模块列表",
    )


class AnalyzeResponse(BaseModel):
    file_hash: str
    message: str


# ---- /cache deferred imports (CacheService from Task 6) ----

def _get_cache_service():
    """Import CacheService lazily (Task 6). Returns None if unavailable."""
    try:
        from backend.services.cache_service import CacheService
        return CacheService()
    except ImportError:
        return None


# ---- Endpoints ----

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.3.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(req: AnalyzeRequest):
    """启动视频分析任务。返回文件哈希作为追踪 ID。"""
    from backend.utils.file_utils import validate_video_file, compute_sha256

    ok, err = validate_video_file(req.file_path)
    if not ok:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": err})

    file_hash = compute_sha256(req.file_path)
    _analysis_states[file_hash] = {
        "status": "pending",
        "file_path": req.file_path,
        "modules": req.modules,
    }
    return AnalyzeResponse(file_hash=file_hash, message="分析任务已创建")


@app.get("/analyze/{file_hash}/status")
async def get_analysis_status(file_hash: str):
    """查询分析任务状态。"""
    state = _analysis_states.get(file_hash)
    if state is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "未找到该分析任务"})
    return state


@app.websocket("/ws/{file_hash}")
async def websocket_progress(websocket: WebSocket, file_hash: str):
    """WebSocket 端点 — 推送实时分析进度。"""
    await websocket.accept()
    conns = _ws_connections.setdefault(file_hash, [])
    conns.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        conns.remove(websocket)


@app.get("/cache/{file_hash}")
async def get_cached_result(file_hash: str):
    """读取缓存的分析结果。"""
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
    """删除缓存的分析结果。"""
    cache = _get_cache_service()
    if cache is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"error": "缓存服务未就绪"},
        )
    cache.delete(file_hash)
    return {"deleted": True}
