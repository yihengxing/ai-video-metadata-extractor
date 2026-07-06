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

        # Task 10 fields should be populated (not None)
        assert result["color_summary"] is not None, "color_summary should be populated by Task 10"
        assert "dominant_hue" in result["color_summary"]
        assert isinstance(result["face_detections"], list)
        assert isinstance(result["object_detections"], list)
        # style_tags should exist (new field from Task 10)
        assert "style_tags" in result
        assert isinstance(result["style_tags"], dict)
        # Task 11 fields should be populated (no longer empty placeholders)
        assert isinstance(result["text_regions"], list)
        assert isinstance(result["motion_summary"], str)
        # motion_summary should contain expected Chinese labels
        assert "静态" in result["motion_summary"] or "运动" in result["motion_summary"]

        # Progress should have been reported
        assert len(progress_log) > 0
        pcts = [p for _, p, _ in progress_log]
        assert pcts[0] >= 0
        assert pcts[-1] == 100.0  # should reach full completion

    finally:
        # Clean up temp video
        try:
            os.unlink(video_path)
        except OSError:
            pass


# ====================================================================
# Task 10 tests — color analysis + face detection
# ====================================================================


def test_color_analysis():
    """Create a solid red image, run _analyze_colors, verify dominant_hue is warm."""
    from backend.modules.visual_analyzer import VisualAnalyzer

    analyzer = VisualAnalyzer()

    # Create a solid red (HSV hue ~0) temp image
    import cv2
    red_img = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)  # BGR red
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, red_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    try:
        result = analyzer._analyze_colors([tmp.name])
        assert isinstance(result, dict)
        assert "dominant_hue" in result
        # Red should map to warm hue
        assert result["dominant_hue"] in ("暖色调", "warm"), \
            f"Expected warm hue for solid red, got {result['dominant_hue']}"
        assert "saturation" in result
        assert "description" in result
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_face_detection_on_blank():
    """Run _detect_faces on the synthetic test video, verify it returns a list."""
    from backend.modules.visual_analyzer import VisualAnalyzer

    video_path = _make_test_video()
    try:
        analyzer = VisualAnalyzer()
        # Minimal shots list for face detection
        shots = [
            {"index": 0, "start_time": 0.0, "end_time": 2.0, "duration": 2.0,
             "thumbnail_path": None, "is_representative": False},
            {"index": 1, "start_time": 2.0, "end_time": 4.0, "duration": 2.0,
             "thumbnail_path": None, "is_representative": False},
        ]
        result = analyzer._detect_faces(video_path, shots)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        # On a blank color video, faces should be 0 (empty list or all face_count=0)
        if len(result) > 0:
            for entry in result:
                assert "shot_index" in entry
                assert "face_count" in entry
                assert entry["face_count"] == 0
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


# ====================================================================
# Task 11 tests — motion analysis + text detection
# ====================================================================


def test_motion_analysis():
    """Run _analyze_motion on a synthetic video, verify it returns a string
    with expected Chinese motion-classification labels."""
    from backend.modules.visual_analyzer import VisualAnalyzer

    video_path = _make_test_video()
    try:
        analyzer = VisualAnalyzer()
        shots = [
            {"index": 0, "start_time": 0.0, "end_time": 2.0, "duration": 2.0,
             "thumbnail_path": None, "is_representative": False},
            {"index": 1, "start_time": 2.0, "end_time": 4.0, "duration": 2.0,
             "thumbnail_path": None, "is_representative": False},
        ]
        result = analyzer._analyze_motion(video_path, shots)

        # Must return a string
        assert isinstance(result, str), f"Expected str, got {type(result)}"

        # The synthetic video is two static solid-color blocks, so motion
        # should be classified as "静态".  The summary string must contain
        # at least one of the known Chinese labels.
        has_label = (
            "静态" in result
            or "轻微运动" in result
            or "剧烈运动" in result
            or "未知" in result
        )
        assert has_label, f"Motion summary missing expected label: {result!r}"
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


def test_text_detection():
    """Run _detect_text on a synthetic video (no text), verify it returns
    a list of dicts with expected keys (shot_index, has_text, bbox, timestamp)."""
    from backend.modules.visual_analyzer import VisualAnalyzer

    video_path = _make_test_video()
    try:
        analyzer = VisualAnalyzer()
        shots = [
            {"index": 0, "start_time": 0.0, "end_time": 2.0, "duration": 2.0,
             "thumbnail_path": None, "is_representative": False},
            {"index": 1, "start_time": 2.0, "end_time": 4.0, "duration": 2.0,
             "thumbnail_path": None, "is_representative": False},
        ]
        result = analyzer._detect_text(video_path, shots)

        # Must return a list
        assert isinstance(result, list), f"Expected list, got {type(result)}"

        # The synthetic video has no text, so all entries should have has_text=False
        # and bbox should be None
        assert len(result) > 0, "Expected at least one result entry per shot"

        for entry in result:
            assert "shot_index" in entry, f"Missing shot_index: {entry}"
            assert "has_text" in entry, f"Missing has_text: {entry}"
            assert "bbox" in entry, f"Missing bbox: {entry}"
            assert "timestamp" in entry, f"Missing timestamp: {entry}"
            # No text on solid-color frames
            assert entry["has_text"] is False, \
                f"Expected has_text=False on blank video, got {entry['has_text']}"
            assert entry["bbox"] is None or entry["bbox"] == [], \
                f"Expected bbox=None/[] on blank video, got {entry['bbox']}"
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass
