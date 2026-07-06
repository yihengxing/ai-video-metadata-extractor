"""Tests for VisualAnalyzer — scene detection & keyframe extraction (Task 9)."""
from __future__ import annotations
import pytest
import numpy as np
import tempfile
import os
from backend.models.schemas import TechMetadata


def _make_test_video():
    """Generate a small synthetic MP4 with 2 distinct scenes (red -> blue)."""
    import cv2
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp.name, fourcc, 30.0, (640, 480))
    for _ in range(60):  # 2 seconds of red
        out.write(np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8))
    for _ in range(60):  # 2 seconds of blue
        out.write(np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8))
    out.release()
    return tmp.name


@pytest.mark.asyncio
async def test_visual_analyzer_module_name():
    """Verify the module_name property returns "visual"."""
    from backend.modules.visual_analyzer import VisualAnalyzer
    analyzer = VisualAnalyzer()
    assert analyzer.module_name == "visual"


@pytest.mark.asyncio
async def test_extract_detects_shots():
    """Create a two-scene video and verify shot_count >= 2 with populated shots."""
    from backend.modules.visual_analyzer import VisualAnalyzer

    video_path = _make_test_video()
    try:
        dummy_meta = TechMetadata(
            container_format="mp4",
            video_codec="H.264",
            video_profile="High",
            resolution_width=640,
            resolution_height=480,
            frame_rate=30.0,
            total_bitrate_bps=1_000_000,
            video_bitrate_bps=800_000,
            audio_codec="AAC",
            audio_sample_rate_hz=44100,
            audio_bitrate_bps=128_000,
            gop_structure="GOP=30",
            color_space="BT.709",
            hdr_info="SDR",
            duration=4.0,
            file_size_bytes=500_000,
            platform_fingerprint=None,
        )

        analyzer = VisualAnalyzer()
        progress_log: list[tuple] = []

        def _progress(module: str, pct: float, msg: str) -> None:
            progress_log.append((module, pct, msg))

        result = await analyzer.extract(video_path, dummy_meta, progress_cb=_progress)

        # Verify return dict structure
        assert "shots" in result
        assert "shot_count" in result
        assert "avg_shot_duration" in result
        assert "transitions" in result
        assert "keyframe_grid_paths" in result
        assert "representative_frames" in result
        assert "color_summary" in result
        assert "text_regions" in result
        assert "face_detections" in result
        assert "object_detections" in result
        assert "motion_summary" in result

        # Scene detection should find at least 2 shots for red->blue cut
        assert result["shot_count"] >= 2, f"Expected >= 2 shots, got {result['shot_count']}"
        assert len(result["shots"]) == result["shot_count"]

        # Each shot dict should have required keys
        for shot in result["shots"]:
            assert "index" in shot
            assert "start_time" in shot
            assert "end_time" in shot
            assert "duration" in shot
            assert "thumbnail_path" in shot
            assert "is_representative" in shot

        # avg_shot_duration should be positive
        assert result["avg_shot_duration"] > 0

        # transitions should be all "硬切" for now
        assert all(t == "硬切" for t in result["transitions"])

        # keyframe_grid_paths should be non-empty
        assert len(result["keyframe_grid_paths"]) > 0, "Expected keyframe grid thumbnails"

        # representative_frames should have at most 5
        assert 0 < len(result["representative_frames"]) <= 5

        # Future fields should be None or empty
        assert result["color_summary"] is None
        assert result["text_regions"] == []
        assert result["face_detections"] == []
        assert result["object_detections"] == []
        assert result["motion_summary"] is None

        # Progress should have been reported
        assert len(progress_log) > 0
        pcts = [p for _, p, _ in progress_log]
        assert pcts[0] >= 0
        assert pcts[-1] >= 90  # should have reached near-completion

    finally:
        # Clean up temp video
        try:
            os.unlink(video_path)
        except OSError:
            pass
