"""Pipeline Orchestrator — coordinates the multi-module analysis DAG.

DAG topology (v1.3):
    Validate + Hash → Tech ─┬→ Visual ∥ Audio ─┬→ AI ∥ SourceRecovery → Aggregate → Cache
                            └────────────────────┘

Progress is streamed via WebSocket after each step.
Graceful degradation: skipped modules are marked as such, failures are
isolated and do not stop the pipeline.
"""
from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from backend.models.schemas import (
    TechMetadata,
    VisualAnalysis,
    AudioAnalysis,
    AIInference,
    SourceRecoveryHit,
    ModuleStatusValue,
    AnalysisResult,
    AnalysisProgress,
)
from backend.modules.tech_extractor import TechExtractor
from backend.modules.visual_analyzer import VisualAnalyzer
from backend.modules.audio_analyzer import AudioAnalyzer
from backend.modules.ai_inferrer import AIInferrer
from backend.modules.source_recovery import SourceRecoveryMatcher
from backend.services.cache_service import CacheService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default TechMetadata sentinel used for the *input* side of tech extraction,
# where we do not yet have real metadata.  The tech extractor ignores it and
# fetches its own data from ffprobe.
# ---------------------------------------------------------------------------
_DUMMY_TECH = TechMetadata(
    container_format="",
    video_codec="",
    video_profile="",
    resolution_width=0,
    resolution_height=0,
    frame_rate=0.0,
    total_bitrate_bps=0,
    video_bitrate_bps=0,
    audio_codec="",
    audio_sample_rate_hz=0,
    audio_bitrate_bps=0,
    gop_structure="",
    color_space="",
    hdr_info="",
    duration=0.0,
    file_size_bytes=0,
    platform_fingerprint=None,
)


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------

class WebSocketManager:
    """Simple pub-sub manager that tracks WebSocket connections keyed by
    ``file_hash`` and broadcasts JSON messages to all subscribers."""

    def __init__(self) -> None:
        self._connections: dict[str, list] = {}

    def register(self, file_hash: str, websocket) -> None:
        conns = self._connections.setdefault(file_hash, [])
        conns.append(websocket)

    def unregister(self, file_hash: str, websocket) -> None:
        conns = self._connections.get(file_hash)
        if conns and websocket in conns:
            conns.remove(websocket)

    async def broadcast(self, file_hash: str, message: str) -> None:
        """Send *message* (a JSON string) to every websocket listening for
        ``file_hash``.  Disconnected clients are pruned automatically."""
        conns = list(self._connections.get(file_hash, []))
        dead = []
        for ws in conns:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)  # client disconnected
        for ws in dead:
            self.unregister(file_hash, ws)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AnalysisOrchestrator:
    """Coordinates the analysis pipeline DAG:

        Tech → (Visual ∥ Audio) → (AI ∥ SourceRecovery) → Aggregate → Cache

    Progress events are pushed via *ws_manager* after every step.
    """

    def __init__(self) -> None:
        self.tech_extractor = TechExtractor()
        self.visual_analyzer = VisualAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.ai_inferrer = AIInferrer()
        self.source_matcher = SourceRecoveryMatcher()
        self.cache = CacheService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_analysis(
        self,
        file_path: str,
        modules: list[str],
        ws_manager: Optional[WebSocketManager] = None,
    ) -> AnalysisResult:
        """Execute the full pipeline and stream progress via WebSocket.

        Parameters
        ----------
        file_path : str
            Absolute path to the video file (already validated by caller).
        modules : list[str]
            Subset of ``["tech", "visual", "audio", "ai", "source_recovery"]``
            specifying which modules to run.
        ws_manager : WebSocketManager or None
            If given, progress messages are broadcast after every step.

        Returns
        -------
        AnalysisResult
            The aggregated analysis result with per-module status.
        """
        from backend.utils.file_utils import compute_sha256

        file_hash = compute_sha256(file_path)
        module_set = set(modules)

        # ---- cache check --------------------------------------------------
        if self.cache.exists(file_hash):
            cached = self.cache.get(file_hash)
            if cached:
                logger.info("Cache hit for %s — returning cached result", file_hash)
                result = AnalysisResult(**cached)
                await self._broadcast(ws_manager, file_hash, "cache", "completed",
                                       100.0, message="从缓存加载")
                return result

        module_status: dict[str, ModuleStatusValue] = {}

        # ---- step 1: tech (always) ----------------------------------------
        tech_dict = await self._run_tech(
            file_path, file_hash, ws_manager, module_status
        )

        # Build the real TechMetadata from the extracted dict
        tech_meta = TechMetadata(**tech_dict) if tech_dict else _DUMMY_TECH

        # ---- step 2: visual ∥ audio --------------------------------------
        visual_dict: Optional[dict] = None
        audio_dict: Optional[dict] = None
        visual_tasks = []

        async def _run_visual() -> Optional[dict]:
            if "visual" not in module_set:
                module_status["visual"] = "skipped"
                return None
            return await self._run_visual(
                file_path, tech_meta, file_hash, ws_manager, module_status
            )

        async def _run_audio() -> Optional[dict]:
            if "audio" not in module_set:
                module_status["audio"] = "skipped"
                return None
            return await self._run_audio(
                file_path, tech_meta, file_hash, ws_manager, module_status
            )

        visual_dict, audio_dict = await asyncio.gather(
            _run_visual(), _run_audio()
        )

        # ---- step 3: ai ∥ source_recovery --------------------------------
        ai_dict: Optional[dict] = None
        source_hits: Optional[list[dict]] = None

        # Gather inputs needed by downstream modules
        keyframe_paths = self._extract_keyframe_paths(visual_dict)
        audio_text = self._extract_audio_text(audio_dict)

        async def _run_ai() -> Optional[dict]:
            if "ai" not in module_set:
                module_status["ai"] = "skipped"
                return None
            return await self._run_ai(
                file_path, tech_meta, keyframe_paths, audio_text,
                file_hash, ws_manager, module_status,
            )

        async def _run_source() -> Optional[list[dict]]:
            if "source_recovery" not in module_set:
                module_status["source_recovery"] = "skipped"
                return None
            return await self._run_source_recovery(
                keyframe_paths, file_hash, ws_manager, module_status,
            )

        ai_dict, source_hits = await asyncio.gather(
            _run_ai(), _run_source()
        )

        # ---- step 4: aggregate -------------------------------------------
        await self._broadcast(ws_manager, file_hash, "aggregate", "running",
                               90.0, message="正在聚合结果...")

        best_hit = self._pick_best_hit(source_hits) if source_hits else None

        result = AnalysisResult(
            file_hash=file_hash,
            file_path=file_path,
            analyzed_at=datetime.now(timezone.utc),
            schema_version="1.3.0",
            tech_metadata=tech_meta,
            visual_analysis=VisualAnalysis(**visual_dict) if visual_dict else None,
            audio_analysis=AudioAnalysis(**audio_dict) if audio_dict else None,
            ai_inference=AIInference(**ai_dict) if ai_dict else None,
            source_recovery=best_hit,
            module_status=module_status,
        )

        # ---- step 5: cache ------------------------------------------------
        self.cache.put(file_hash, json.loads(result.model_dump_json()))
        logger.info("Cached result for %s", file_hash)

        await self._broadcast(ws_manager, file_hash, "aggregate", "completed",
                               100.0, partial_result=json.loads(result.model_dump_json()),
                               message="分析完成")

        return result

    # ------------------------------------------------------------------
    # Per-module runners (graceful degradation)
    # ------------------------------------------------------------------

    async def _run_tech(
        self,
        file_path: str,
        file_hash: str,
        ws_manager: Optional[WebSocketManager],
        module_status: dict[str, ModuleStatusValue],
    ) -> Optional[dict]:
        await self._broadcast(ws_manager, file_hash, "tech", "running", 0.0,
                               message="开始技术提取...")
        try:
            result = await self.tech_extractor.extract(
                file_path, _DUMMY_TECH,
                progress_cb=self._make_cb(ws_manager, file_hash),
            )
            module_status["tech"] = "completed"
            await self._broadcast(ws_manager, file_hash, "tech", "completed",
                                   100.0, message="技术提取完成")
            return result
        except Exception as exc:
            logger.exception("Tech extraction failed: %s", exc)
            module_status["tech"] = "failed"
            await self._broadcast(ws_manager, file_hash, "tech", "failed",
                                   0.0, message=f"技术提取失败: {exc}")
            return None

    async def _run_visual(
        self,
        file_path: str,
        tech_meta: TechMetadata,
        file_hash: str,
        ws_manager: Optional[WebSocketManager],
        module_status: dict[str, ModuleStatusValue],
    ) -> Optional[dict]:
        await self._broadcast(ws_manager, file_hash, "visual", "running", 0.0,
                               message="开始视觉分析...")
        try:
            result = await asyncio.wait_for(
                self.visual_analyzer.extract(
                    file_path, tech_meta,
                    progress_cb=self._make_cb(ws_manager, file_hash),
                ),
                timeout=120.0,
            )
            module_status["visual"] = "completed"
            await self._broadcast(ws_manager, file_hash, "visual", "completed",
                                   100.0, message="视觉分析完成")
            return result
        except Exception as exc:
            logger.exception("Visual analysis failed: %s", exc)
            module_status["visual"] = "failed"
            await self._broadcast(ws_manager, file_hash, "visual", "failed",
                                   0.0, message=f"视觉分析失败: {exc}")
            return None

    async def _run_audio(
        self,
        file_path: str,
        tech_meta: TechMetadata,
        file_hash: str,
        ws_manager: Optional[WebSocketManager],
        module_status: dict[str, ModuleStatusValue],
    ) -> Optional[dict]:
        await self._broadcast(ws_manager, file_hash, "audio", "running", 0.0,
                               message="开始音频分析...")
        try:
            result = await asyncio.wait_for(
                self.audio_analyzer.extract(
                    file_path, tech_meta,
                    progress_cb=self._make_cb(ws_manager, file_hash),
                ),
                timeout=120.0,
            )
            module_status["audio"] = "completed"
            await self._broadcast(ws_manager, file_hash, "audio", "completed",
                                   100.0, message="音频分析完成")
            return result
        except Exception as exc:
            logger.exception("Audio analysis failed: %s", exc)
            module_status["audio"] = "failed"
            await self._broadcast(ws_manager, file_hash, "audio", "failed",
                                   0.0, message=f"音频分析失败: {exc}")
            return None

    async def _run_ai(
        self,
        file_path: str,
        tech_meta: TechMetadata,
        keyframe_paths: list[str],
        audio_text: str,
        file_hash: str,
        ws_manager: Optional[WebSocketManager],
        module_status: dict[str, ModuleStatusValue],
    ) -> Optional[dict]:
        await self._broadcast(ws_manager, file_hash, "ai", "running", 0.0,
                               message="开始AI推断...")
        try:
            result = await asyncio.wait_for(
                self.ai_inferrer.extract(
                    file_path, tech_meta,
                    progress_cb=self._make_cb(ws_manager, file_hash),
                    keyframe_paths=keyframe_paths,
                    audio_text=audio_text,
                ),
                timeout=90.0,
            )
            module_status["ai"] = "completed"
            await self._broadcast(ws_manager, file_hash, "ai", "completed",
                                   100.0, message="AI推断完成")
            return result
        except Exception as exc:
            logger.exception("AI inference failed: %s", exc)
            module_status["ai"] = "failed"
            await self._broadcast(ws_manager, file_hash, "ai", "failed",
                                   0.0, message=f"AI推断失败: {exc}")
            return None

    async def _run_source_recovery(
        self,
        keyframe_paths: list[str],
        file_hash: str,
        ws_manager: Optional[WebSocketManager],
        module_status: dict[str, ModuleStatusValue],
    ) -> Optional[list[dict]]:
        await self._broadcast(ws_manager, file_hash, "source_recovery", "running",
                               0.0, message="开始源回捞匹配...")
        try:
            hits = await asyncio.wait_for(
                self.source_matcher.match(
                    keyframe_paths,
                    progress_cb=self._make_cb(ws_manager, file_hash),
                ),
                timeout=45.0,
            )
            module_status["source_recovery"] = "completed"
            await self._broadcast(ws_manager, file_hash, "source_recovery",
                                   "completed", 100.0, message="源回捞完成")
            return [json.loads(h.model_dump_json()) for h in hits] if hits else []
        except Exception as exc:
            logger.exception("Source recovery failed: %s", exc)
            module_status["source_recovery"] = "failed"
            await self._broadcast(ws_manager, file_hash, "source_recovery",
                                   "failed", 0.0, message=f"源回捞失败: {exc}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keyframe_paths(visual_dict: Optional[dict]) -> list[str]:
        """Pull keyframe / representative frame paths from visual result."""
        if not visual_dict:
            return []
        paths = []
        for key in ("keyframe_grid_paths", "representative_frames"):
            for p in visual_dict.get(key, []) or []:
                if p and p not in paths:
                    paths.append(p)
        return paths

    @staticmethod
    def _extract_audio_text(audio_dict: Optional[dict]) -> str:
        """Pull full text from audio result."""
        if not audio_dict:
            return ""
        return audio_dict.get("full_text", "") or ""

    @staticmethod
    def _pick_best_hit(hits: Optional[list[dict]]) -> Optional[SourceRecoveryHit]:
        """Pick the top source-recovery hit by confidence_score."""
        best = None
        best_score = -1.0
        for h in (hits or []):
            score = h.get("confidence_score", 0.0)
            if score > best_score:
                best_score = score
                best = h
        return SourceRecoveryHit(**best) if best else None

    def _make_cb(self, ws_manager: Optional[WebSocketManager], file_hash: str):
        """Return a ProgressCallback that forwards through the WebSocket manager."""
        if ws_manager is None:
            return None

        async def _cb(module: str, progress: float, message: str) -> None:
            await self._broadcast(
                ws_manager, file_hash, module, "running", progress, message=message
            )

        return _cb

    @staticmethod
    async def _broadcast(
        ws_manager: Optional[WebSocketManager],
        file_hash: str,
        module: str,
        status: ModuleStatusValue,
        progress: float,
        partial_result: Optional[dict] = None,
        message: str = "",
    ) -> None:
        """Send an ``AnalysisProgress`` message to all WebSocket listeners."""
        if ws_manager is None:
            return
        msg = AnalysisProgress(
            file_hash=file_hash,
            module=module,
            status=status,
            progress_pct=progress,
            message=message,
            partial_result=partial_result,
        )
        await ws_manager.broadcast(file_hash, msg.model_dump_json())
