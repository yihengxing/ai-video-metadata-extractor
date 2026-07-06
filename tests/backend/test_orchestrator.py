"""Integration tests for AnalysisOrchestrator (Tasks 38-40).

Covers:
    - Tech-only pipeline execution
    - Full pipeline DAG ordering with mocked modules
    - Cache hit short-circuit
    - Module failure graceful degradation
"""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import hashlib
import json
import os
import tempfile

import pytest
import numpy as np

from backend.models.schemas import (
    TechMetadata,
    VisualAnalysis,
    AudioAnalysis,
    AIInference,
    SourceRecoveryHit,
    AnalysisResult,
    AnalysisProgress,
    ModuleStatusValue,
)
from backend.orchestrator import AnalysisOrchestrator, WebSocketManager, _DUMMY_TECH


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_test_video():
    """Generate a small synthetic MP4 with 2 distinct scenes (red -> blue)."""
    import cv2
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp.name, fourcc, 30.0, (640, 480))
    for _ in range(60):
        out.write(np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8))
    for _ in range(60):
        out.write(np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8))
    out.release()
    return tmp.name


def _make_tech_dict() -> dict:
    return {
        "container_format": "mp4",
        "video_codec": "H.264",
        "video_profile": "High",
        "resolution_width": 640,
        "resolution_height": 480,
        "frame_rate": 30.0,
        "total_bitrate_bps": 1000000,
        "video_bitrate_bps": 800000,
        "audio_codec": "AAC",
        "audio_sample_rate_hz": 44100,
        "audio_bitrate_bps": 128000,
        "gop_structure": "GOP=30",
        "color_space": "BT.709",
        "hdr_info": "SDR",
        "duration": 4.0,
        "file_size_bytes": 500000,
        "platform_fingerprint": None,
    }


def _make_visual_dict() -> dict:
    return {
        "shots": [
            {"index": 0, "start_time": 0.0, "end_time": 2.0, "duration": 2.0,
             "thumbnail_path": "/tmp/shot0.jpg", "is_representative": False},
            {"index": 1, "start_time": 2.0, "end_time": 4.0, "duration": 2.0,
             "thumbnail_path": "/tmp/shot1.jpg", "is_representative": False},
        ],
        "shot_count": 2,
        "avg_shot_duration": 2.0,
        "transitions": ["hard cut"],
        "keyframe_grid_paths": ["/tmp/kf_1.jpg", "/tmp/kf_2.jpg"],
        "representative_frames": ["/tmp/rep_1.jpg"],
        "color_summary": None,
        "text_regions": [],
        "face_detections": [],
        "object_detections": [],
        "motion_summary": None,
    }


def _make_audio_dict() -> dict:
    return {
        "full_text": "这是测试音频文本",
        "text_segments": [{"text": "这是测试", "start": 0.0, "end": 1.0}],
        "speech_rate": 5.0,
        "speech_emotion": "calm",
        "bgm_title": None,
        "bgm_artist": None,
        "bgm_style_tags": [],
        "bgm_emotion": None,
        "bgm_bpm": None,
        "sound_events": [],
        "voice_to_bg_ratio": None,
        "audio_structure": None,
    }


def _make_ai_dict() -> dict:
    return {
        "inferred_tool": "Kling 1.6",
        "inferred_tool_confidence": 0.8,
        "inferred_prompt": "a beautiful sunset",
        "inferred_prompt_confidence": 0.7,
        "style_tags": ["cinematic"],
        "inferred_workflow": None,
        "inferred_workflow_confidence": 0.0,
        "imitation_suggestions": [],
        "model_recommendations": [],
        "overall_confidence": 0.75,
    }


def _make_source_hits() -> list[dict]:
    return [{
        "status": "partial_match",
        "source_url": "https://civitai.com/images/999",
        "similarity": 85.0,
        "hit_keyframes": 1,
        "total_keyframes_sent": 5,
        "workflow_json": "{}",
        "prompt": "a sunset",
        "seed": 42,
        "sampler": "dpmpp_2m",
        "steps": 20,
        "cfg_scale": 7.0,
        "model_name": "sd_xl.safetensors",
        "confidence_score": 0.72,
        "source_trust": "civitai",
    }]


# Pre-computed hash for deterministic cache tests
_FAKE_HASH = hashlib.sha256(b"fake_path").hexdigest()


# ---------------------------------------------------------------------------
# Stub WebSocket for tracking broadcasts
# ---------------------------------------------------------------------------

class StubWebSocket:
    """Collects sent messages in a list for assertion."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, message: str) -> None:
        self.sent.append(message)


def _fake_compute_sha256(file_path: str) -> str:
    return hashlib.sha256(file_path.encode()).hexdigest()


# ===================================================================
# Test 1: Tech-only pipeline (requires real ffmpeg + real file)
# ===================================================================

@pytest.mark.asyncio
async def test_orchestrator_runs_tech_only():
    """Verify pipeline runs successfully with only the 'tech' module selected."""
    from backend.services.ffmpeg_service import FFmpegService

    # Skip if ffmpeg is not available
    if not FFmpegService.is_installed():
        pytest.skip("FFmpeg/ffprobe not installed")

    video_path = _make_test_video()
    ws = StubWebSocket()
    wsm = WebSocketManager()
    wsm.register("test_hash", ws)

    try:
        orch = AnalysisOrchestrator()
        result = await orch.run_analysis(
            video_path, modules=["tech"], ws_manager=wsm,
        )

        assert isinstance(result, AnalysisResult)
        assert result.tech_metadata is not None
        assert result.tech_metadata.container_format == "mp4"
        assert result.module_status["tech"] == "completed"
        assert result.module_status.get("visual", "skipped") == "skipped"
        assert result.module_status.get("audio", "skipped") == "skipped"

        # Verify progress messages were sent
        assert len(ws.sent) > 0
        msgs = [AnalysisProgress(**json.loads(m)) for m in ws.sent]
        modules_seen = {m.module for m in msgs}
        assert "tech" in modules_seen
        assert "aggregate" in modules_seen
    finally:
        os.unlink(video_path)


# ===================================================================
# Test 2: Full pipeline DAG ordering
# ===================================================================

@pytest.mark.asyncio
@patch("backend.utils.file_utils.compute_sha256", side_effect=_fake_compute_sha256)
async def test_orchestrator_runs_full_pipeline(_mock_sha):
    """Mock all modules and verify the DAG runs in the correct order:
    tech -> (visual || audio) -> (ai || source_recovery) -> aggregate.
    """
    orch = AnalysisOrchestrator()

    tech_dict = _make_tech_dict()
    vis_dict = _make_visual_dict()
    aud_dict = _make_audio_dict()
    ai_dict = _make_ai_dict()
    src_hits = _make_source_hits()

    # Track call order via a shared list
    call_order: list[str] = []

    async def _tech_extract(file_path, tech_meta, progress_cb=None):
        call_order.append("tech_start")
        return tech_dict

    async def _visual_extract(file_path, tech_meta, progress_cb=None):
        call_order.append("visual_start")
        return vis_dict

    async def _audio_extract(file_path, tech_meta, progress_cb=None):
        call_order.append("audio_start")
        return aud_dict

    async def _ai_extract(file_path, tech_meta, progress_cb=None,
                          keyframe_paths=None, audio_text=""):
        call_order.append("ai_start")
        return ai_dict

    async def _source_match(keyframe_paths, progress_cb=None):
        call_order.append("source_start")
        return [SourceRecoveryHit(**h) for h in src_hits]

    # Mock cache to avoid real filesystem hits
    cache = MagicMock()
    cache.exists.return_value = False

    orch.tech_extractor.extract = _tech_extract
    orch.visual_analyzer.extract = _visual_extract
    orch.audio_analyzer.extract = _audio_extract
    orch.ai_inferrer.extract = _ai_extract
    orch.source_matcher.match = _source_match
    orch.cache = cache

    fake_file = "/fake/video.mp4"
    expected_hash = _fake_compute_sha256(fake_file)

    ws = StubWebSocket()
    wsm = WebSocketManager()
    wsm.register(expected_hash, ws)

    result = await orch.run_analysis(
        fake_file,
        modules=["tech", "visual", "audio", "ai", "source_recovery"],
        ws_manager=wsm,
    )

    # Assert DAG order: tech first, then visual+audio (concurrent), then
    # ai+source (concurrent)
    assert call_order[0] == "tech_start"

    # After tech, both visual and audio must have started
    post_tech = set(call_order[1:3])
    assert post_tech == {"visual_start", "audio_start"}

    # After visual+audio, both ai and source must have started
    post_vis_aud = set(call_order[3:])
    assert post_vis_aud == {"ai_start", "source_start"}

    # Verify result structure
    assert isinstance(result, AnalysisResult)
    assert result.tech_metadata.container_format == "mp4"
    assert result.visual_analysis is not None
    assert result.audio_analysis is not None
    assert result.ai_inference is not None
    assert result.source_recovery is not None
    assert result.module_status["tech"] == "completed"
    assert result.module_status["visual"] == "completed"
    assert result.module_status["audio"] == "completed"
    assert result.module_status["ai"] == "completed"
    assert result.module_status["source_recovery"] == "completed"

    # Cache.put should have been called
    cache.put.assert_called_once()

    # Progress was broadcast
    assert len(ws.sent) > 0


# ===================================================================
# Test 3: Cache hit short-circuit
# ===================================================================

@pytest.mark.asyncio
@patch("backend.utils.file_utils.compute_sha256", side_effect=_fake_compute_sha256)
async def test_orchestrator_cache_hit(_mock_sha):
    """Run analysis once -- second call should return cached result
    without re-running extractors."""
    orch = AnalysisOrchestrator()

    fake_file = "/fake/v.mp4"
    fake_hash = _fake_compute_sha256(fake_file)

    # Pre-populate a cached result
    cached_result = {
        "file_hash": fake_hash,
        "file_path": fake_file,
        "schema_version": "1.3.0",
        "tech_metadata": _make_tech_dict(),
        "visual_analysis": None,
        "audio_analysis": None,
        "ai_inference": None,
        "source_recovery": None,
        "module_status": {"tech": "completed", "visual": "skipped",
                          "audio": "skipped", "ai": "skipped",
                          "source_recovery": "skipped"},
        "analyzed_at": "2025-01-01T00:00:00Z",
    }

    call_count = 0

    async def _tech_extract(file_path, tech_meta, progress_cb=None):
        nonlocal call_count
        call_count += 1
        return _make_tech_dict()

    orch.tech_extractor.extract = _tech_extract
    orch.visual_analyzer.extract = AsyncMock()
    orch.audio_analyzer.extract = AsyncMock()
    orch.ai_inferrer.extract = AsyncMock()
    orch.source_matcher.match = AsyncMock()

    cache = MagicMock()
    orch.cache = cache

    # --- First call: cache miss, should run modules ---
    cache.exists.return_value = False
    cache.get.return_value = None

    ws1 = StubWebSocket()
    wsm1 = WebSocketManager()
    wsm1.register(fake_hash, ws1)

    result1 = await orch.run_analysis(
        fake_file,
        modules=["tech"],
        ws_manager=wsm1,
    )
    assert result1 is not None
    assert call_count == 1

    # --- Second call: cache hit, should NOT run modules ---
    cache.exists.return_value = True
    cache.get.return_value = cached_result

    ws2 = StubWebSocket()
    wsm2 = WebSocketManager()
    wsm2.register(fake_hash, ws2)

    result2 = await orch.run_analysis(
        fake_file,
        modules=["tech"],
        ws_manager=wsm2,
    )
    assert result2 is not None
    # Extractors should NOT have been called again
    assert call_count == 1

    # The result from the cache should match the cached data
    assert result2.tech_metadata.container_format == "mp4"

    # Verify a "completed" broadcast was sent for the cache hit
    cache_msgs = [json.loads(m) for m in ws2.sent]
    assert any(m["module"] == "cache" and m["status"] == "completed"
               for m in cache_msgs)


# ===================================================================
# Test 4: Module failure graceful degradation
# ===================================================================

@pytest.mark.asyncio
@patch("backend.utils.file_utils.compute_sha256", side_effect=_fake_compute_sha256)
async def test_orchestrator_module_failure_doesnt_stop_pipeline(_mock_sha):
    """When one module (visual) raises an exception, the orchestrator must
    mark it as 'failed' and continue executing the remaining modules."""
    orch = AnalysisOrchestrator()

    tech_dict = _make_tech_dict()
    aud_dict = _make_audio_dict()
    ai_dict = _make_ai_dict()
    src_hits = _make_source_hits()

    # Shared tracking
    audio_called = asyncio.Event()

    async def _tech_extract(file_path, tech_meta, progress_cb=None):
        return tech_dict

    async def _visual_extract(file_path, tech_meta, progress_cb=None):
        raise RuntimeError("Simulated scene-detect failure")

    async def _audio_extract(file_path, tech_meta, progress_cb=None):
        audio_called.set()
        return aud_dict

    async def _ai_extract(file_path, tech_meta, progress_cb=None,
                          keyframe_paths=None, audio_text=""):
        return ai_dict

    async def _source_match(keyframe_paths, progress_cb=None):
        return [SourceRecoveryHit(**h) for h in src_hits]

    cache = MagicMock()
    cache.exists.return_value = False

    orch.tech_extractor.extract = _tech_extract
    orch.visual_analyzer.extract = _visual_extract
    orch.audio_analyzer.extract = _audio_extract
    orch.ai_inferrer.extract = _ai_extract
    orch.source_matcher.match = _source_match
    orch.cache = cache

    ws = StubWebSocket()
    wsm = WebSocketManager()
    wsm.register("fail_hash", ws)

    result = await orch.run_analysis(
        "/fake/v.mp4",
        modules=["tech", "visual", "audio", "ai", "source_recovery"],
        ws_manager=wsm,
    )

    # Visual must be marked as failed
    assert result.module_status["visual"] == "failed"

    # Audio must have been called (pipeline continued) and completed
    assert audio_called.is_set()
    assert result.module_status["audio"] == "completed"

    # AI and source_recovery should also have completed (they run after)
    assert result.module_status["ai"] == "completed"
    assert result.module_status["source_recovery"] == "completed"

    # Tech must have completed
    assert result.module_status["tech"] == "completed"

    # Visual result should be None
    assert result.visual_analysis is None

    # Audio and AI results should be present
    assert result.audio_analysis is not None
    assert result.ai_inference is not None

    # Cache was still populated
    cache.put.assert_called_once()


# ===================================================================
# Test 5: Skipped modules
# ===================================================================

@pytest.mark.asyncio
@patch("backend.utils.file_utils.compute_sha256", side_effect=_fake_compute_sha256)
async def test_orchestrator_skips_unselected_modules(_mock_sha):
    """Modules not in the 'modules' list must be marked 'skipped' and
    not be called at all."""
    orch = AnalysisOrchestrator()

    visual_called = False

    async def _tech_extract(file_path, tech_meta, progress_cb=None):
        return _make_tech_dict()

    async def _visual_extract(file_path, tech_meta, progress_cb=None):
        nonlocal visual_called
        visual_called = True
        return _make_visual_dict()

    orch.tech_extractor.extract = _tech_extract
    orch.visual_analyzer.extract = _visual_extract
    orch.audio_analyzer.extract = AsyncMock()
    orch.ai_inferrer.extract = AsyncMock()
    orch.source_matcher.match = AsyncMock()

    cache = MagicMock()
    cache.exists.return_value = False
    orch.cache = cache

    result = await orch.run_analysis(
        "/fake/v.mp4",
        modules=["tech"],  # only tech
    )

    assert not visual_called, "Visual should NOT have been called"
    assert result.module_status["tech"] == "completed"
    assert result.module_status.get("visual", "skipped") == "skipped"
    assert result.module_status.get("audio", "skipped") == "skipped"
    assert result.module_status.get("ai", "skipped") == "skipped"
    assert result.module_status.get("source_recovery", "skipped") == "skipped"
    assert result.visual_analysis is None
    assert result.audio_analysis is None


# ===================================================================
# Test 6: AIInferrer receives keyframes + audio text
# ===================================================================

@pytest.mark.asyncio
@patch("backend.utils.file_utils.compute_sha256", side_effect=_fake_compute_sha256)
async def test_orchestrator_passes_keyframes_and_audio_to_ai(_mock_sha):
    """Verify the orchestrator plucks keyframe paths from visual result
    and audio text from audio result before calling the AI inferrer."""
    orch = AnalysisOrchestrator()

    captured_keyframes = None
    captured_audio_text = None

    async def _tech_extract(file_path, tech_meta, progress_cb=None):
        return _make_tech_dict()

    async def _visual_extract(file_path, tech_meta, progress_cb=None):
        return _make_visual_dict()

    async def _audio_extract(file_path, tech_meta, progress_cb=None):
        return _make_audio_dict()

    async def _ai_extract(file_path, tech_meta, progress_cb=None,
                          keyframe_paths=None, audio_text=""):
        nonlocal captured_keyframes, captured_audio_text
        captured_keyframes = keyframe_paths
        captured_audio_text = audio_text
        return _make_ai_dict()

    async def _source_match(keyframe_paths, progress_cb=None):
        return [SourceRecoveryHit(**_make_source_hits()[0])]

    cache = MagicMock()
    cache.exists.return_value = False

    orch.tech_extractor.extract = _tech_extract
    orch.visual_analyzer.extract = _visual_extract
    orch.audio_analyzer.extract = _audio_extract
    orch.ai_inferrer.extract = _ai_extract
    orch.source_matcher.match = _source_match
    orch.cache = cache

    result = await orch.run_analysis(
        "/fake/v.mp4",
        modules=["tech", "visual", "audio", "ai", "source_recovery"],
    )

    # AI should have received keyframe paths from visual
    assert captured_keyframes is not None
    assert len(captured_keyframes) > 0
    assert "/tmp/kf_" in captured_keyframes[0] or "/tmp/rep_" in captured_keyframes[0]

    # AI should have received audio text
    assert captured_audio_text == "这是测试音频文本"


# ===================================================================
# Test 7: WebSocketManager integration
# ===================================================================

@pytest.mark.asyncio
async def test_websocket_manager_broadcast():
    """WebSocketManager should send messages to all registered clients
    and prune disconnected ones gracefully."""
    wsm = WebSocketManager()

    ws1 = StubWebSocket()
    ws2 = StubWebSocket()

    wsm.register("hash_a", ws1)
    wsm.register("hash_a", ws2)

    await wsm.broadcast("hash_a", '{"hello": "world"}')

    assert len(ws1.sent) == 1
    assert len(ws2.sent) == 1
    assert json.loads(ws1.sent[0]) == {"hello": "world"}

    # Unregister and broadcast again
    wsm.unregister("hash_a", ws1)
    await wsm.broadcast("hash_a", '{"second": true}')

    assert len(ws1.sent) == 1  # unchanged
    assert len(ws2.sent) == 2  # received second message

    # Broadcast to unknown hash should not raise
    await wsm.broadcast("nonexistent", "{}")
