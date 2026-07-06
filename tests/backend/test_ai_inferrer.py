"""Tests for AIInferrer — multi-modal LLM inference (Tasks 15+16).

Covers:
- module_name property
- Graceful degradation when no API key is configured
- Correct parsing of a valid LLM JSON response
- Retry logic on transient failures
"""
from __future__ import annotations
import pytest
import base64
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from backend.models.schemas import TechMetadata


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_test_tech_meta(**overrides) -> TechMetadata:
    """Return a minimal TechMetadata for testing."""
    defaults = dict(
        container_format="mp4",
        video_codec="H.264",
        video_profile="High",
        resolution_width=1920,
        resolution_height=1080,
        frame_rate=30.0,
        total_bitrate_bps=8_000_000,
        video_bitrate_bps=6_500_000,
        audio_codec="AAC",
        audio_sample_rate_hz=44100,
        audio_bitrate_bps=128_000,
        gop_structure="GOP=60",
        color_space="BT.709",
        hdr_info="SDR",
        duration=32.0,
        file_size_bytes=15_200_000,
        platform_fingerprint=None,
    )
    defaults.update(overrides)
    return TechMetadata(**defaults)


def _make_test_jpeg() -> bytes:
    """Create a minimal valid JPEG image in memory (1x1 red pixel)."""
    import struct
    # Minimal valid JPEG: SOI + APP0 + DQT + SOF + DHT + SOS + image data + EOI
    # Using a well-known tiny JPEG
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09"
        b"\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
        b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff"
        b"\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04"
        b"\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q"
        b"\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\x09\x0a\x16"
        b"\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
        b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99"
        b"\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
        b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
        b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1"
        b"\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00"
        b"\x00?\x00\xd2\xcf \x11\x00\x03\x11\x01\x02\x11\x01\x03\x11\x01"
        b"\xff\xd9"
    )


def _write_test_keyframes(temp_dir: str, count: int = 3) -> list[str]:
    """Write *count* tiny JPEG files into *temp_dir* and return their paths."""
    paths: list[str] = []
    for i in range(count):
        p = os.path.join(temp_dir, f"keyframe_{i:04d}.jpg")
        with open(p, "wb") as f:
            f.write(_make_test_jpeg())
        paths.append(p)
    return paths


def _valid_llm_json_response() -> dict:
    """Return the expected parsed dict from a valid LLM response."""
    return {
        "inferred_tool": "可灵1.6",
        "inferred_tool_confidence": 0.85,
        "inferred_prompt": "赛博朋克城市夜景，霓虹灯，雨夜，未来感",
        "inferred_prompt_confidence": 0.7,
        "style_tags": ["赛博朋克", "电影感", "暖色调"],
        "inferred_workflow": "文生视频(可灵) → Topaz超分 → AE调色",
        "inferred_workflow_confidence": 0.65,
        "imitation_suggestions": ["建议使用可灵1.6 文生视频", "Topaz Video AI 2x超分"],
        "model_recommendations": ["ComfyUI + AnimateDiff 替代方案"],
        "overall_confidence": pytest.approx((0.85 + 0.7 + 0.65) / 3, rel=0.01),
    }


def _valid_llm_json_string() -> str:
    """Return a valid JSON response string (as the LLM would return)."""
    d = {
        "inferred_tool": "可灵1.6",
        "inferred_tool_confidence": 0.85,
        "inferred_prompt": "赛博朋克城市夜景，霓虹灯，雨夜，未来感",
        "inferred_prompt_confidence": 0.7,
        "style_tags": ["赛博朋克", "电影感", "暖色调"],
        "inferred_workflow": "文生视频(可灵) → Topaz超分 → AE调色",
        "inferred_workflow_confidence": 0.65,
        "imitation_suggestions": ["建议使用可灵1.6 文生视频", "Topaz Video AI 2x超分"],
        "model_recommendations": ["ComfyUI + AnimateDiff 替代方案"],
    }
    return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ai_inferrer_module_name():
    """Verify module_name returns "ai"."""
    from backend.modules.ai_inferrer import AIInferrer
    assert AIInferrer().module_name == "ai"


@pytest.mark.asyncio
async def test_extract_no_api_key_returns_empty():
    """When no API key is configured, extract should return empty result gracefully."""
    from backend.modules.ai_inferrer import AIInferrer

    # Ensure settings has no API key
    with patch("backend.modules.ai_inferrer.settings") as mock_settings:
        mock_settings.llm_api_key = ""
        mock_settings.llm_provider = "claude"

        inferrer = AIInferrer()
        meta = _make_test_tech_meta()
        result = await inferrer.extract("/fake/video.mp4", meta)

        # All fields should be None/empty/default
        assert result["inferred_tool"] is None
        assert result["inferred_tool_confidence"] == 0.0
        assert result["inferred_prompt"] is None
        assert result["inferred_prompt_confidence"] == 0.0
        assert result["style_tags"] == []
        assert result["inferred_workflow"] is None
        assert result["inferred_workflow_confidence"] == 0.0
        assert result["imitation_suggestions"] == []
        assert result["model_recommendations"] == []
        assert result["overall_confidence"] == 0.0


@pytest.mark.asyncio
async def test_extract_mocked_llm_response():
    """Mock httpx to return a valid JSON response and verify correct parsing."""
    from backend.modules.ai_inferrer import AIInferrer

    with tempfile.TemporaryDirectory() as tmpdir:
        keyframe_paths = _write_test_keyframes(tmpdir, count=3)

        # Build a mock httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": _valid_llm_json_string()}],
        }

        # Mock the httpx.AsyncClient context manager
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.modules.ai_inferrer.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("backend.modules.ai_inferrer.settings") as mock_settings:
                mock_settings.llm_api_key = "test-api-key"
                mock_settings.llm_provider = "claude"

                inferrer = AIInferrer()
                meta = _make_test_tech_meta()
                result = await inferrer.extract(
                    "/fake/video.mp4", meta,
                    keyframe_paths=keyframe_paths,
                    audio_text="这是一段测试音频文本",
                )

                # Verify parsed correctly
                assert result["inferred_tool"] == "可灵1.6"
                assert result["inferred_tool_confidence"] == 0.85
                assert result["inferred_prompt"] == "赛博朋克城市夜景，霓虹灯，雨夜，未来感"
                assert result["inferred_prompt_confidence"] == 0.7
                assert result["style_tags"] == ["赛博朋克", "电影感", "暖色调"]
                assert result["inferred_workflow"] == "文生视频(可灵) → Topaz超分 → AE调色"
                assert result["inferred_workflow_confidence"] == 0.65
                assert result["imitation_suggestions"] == [
                    "建议使用可灵1.6 文生视频", "Topaz Video AI 2x超分"
                ]
                assert result["model_recommendations"] == ["ComfyUI + AnimateDiff 替代方案"]
                expected_overall = (0.85 + 0.7 + 0.65) / 3
                assert result["overall_confidence"] == pytest.approx(expected_overall, rel=0.01)

                # Verify the API was called
                mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_extract_llm_retry_on_failure():
    """Mock 2 HTTP failures then a success — verify retry logic works."""
    from backend.modules.ai_inferrer import AIInferrer

    with tempfile.TemporaryDirectory() as tmpdir:
        keyframe_paths = _write_test_keyframes(tmpdir, count=2)

        # Response for the successful call
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "content": [{"text": _valid_llm_json_string()}],
        }

        # Build a mock client whose post() fails twice then succeeds
        mock_client = MagicMock()
        # First two calls raise an HTTP error, third succeeds
        mock_client.post = AsyncMock(side_effect=[
            Exception("Connection timeout"),
            Exception("Server error 500"),
            success_response,
        ])

        with patch("backend.modules.ai_inferrer.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("backend.modules.ai_inferrer.settings") as mock_settings:
                mock_settings.llm_api_key = "test-api-key"
                mock_settings.llm_provider = "claude"

                inferrer = AIInferrer()
                meta = _make_test_tech_meta()
                result = await inferrer.extract(
                    "/fake/video.mp4", meta,
                    keyframe_paths=keyframe_paths,
                )

                # Should have succeeded on the third attempt
                assert result["inferred_tool"] == "可灵1.6"
                assert result["inferred_tool_confidence"] == 0.85

                # Verify 3 attempts were made
                assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_extract_all_retries_exhausted():
    """When all 3 retries fail, return empty result (graceful degradation)."""
    from backend.modules.ai_inferrer import AIInferrer

    with tempfile.TemporaryDirectory() as tmpdir:
        keyframe_paths = _write_test_keyframes(tmpdir, count=2)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=Exception("Always fails"))

        with patch("backend.modules.ai_inferrer.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("backend.modules.ai_inferrer.settings") as mock_settings:
                mock_settings.llm_api_key = "test-api-key"
                mock_settings.llm_provider = "claude"

                inferrer = AIInferrer()
                meta = _make_test_tech_meta()
                result = await inferrer.extract(
                    "/fake/video.mp4", meta,
                    keyframe_paths=keyframe_paths,
                )

                # All retries exhausted → graceful degradation
                assert result["inferred_tool"] is None
                assert result["inferred_tool_confidence"] == 0.0
                assert result["overall_confidence"] == 0.0
                assert result["style_tags"] == []

                # 3 attempts were made
                assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_extract_invalid_json_fallback():
    """When LLM returns invalid JSON, fall back gracefully."""
    from backend.modules.ai_inferrer import AIInferrer

    with tempfile.TemporaryDirectory() as tmpdir:
        keyframe_paths = _write_test_keyframes(tmpdir, count=2)

        # Response with garbage text (not JSON)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "这是一段无效的回复，没有JSON格式"}],
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.modules.ai_inferrer.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("backend.modules.ai_inferrer.settings") as mock_settings:
                mock_settings.llm_api_key = "test-api-key"
                mock_settings.llm_provider = "claude"

                inferrer = AIInferrer()
                meta = _make_test_tech_meta()
                result = await inferrer.extract(
                    "/fake/video.mp4", meta,
                    keyframe_paths=keyframe_paths,
                )

                # Should return empty result on JSON parse failure
                assert result["inferred_tool"] is None
                assert result["inferred_tool_confidence"] == 0.0
                assert result["overall_confidence"] == 0.0
                assert result["style_tags"] == []


@pytest.mark.asyncio
async def test_extract_openai_provider():
    """Mock httpx for OpenAI provider — verify the correct endpoint is called."""
    from backend.modules.ai_inferrer import AIInferrer

    with tempfile.TemporaryDirectory() as tmpdir:
        keyframe_paths = _write_test_keyframes(tmpdir, count=1)

        # OpenAI-style response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": _valid_llm_json_string()},
            }],
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.modules.ai_inferrer.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("backend.modules.ai_inferrer.settings") as mock_settings:
                mock_settings.llm_api_key = "test-openai-key"
                mock_settings.llm_provider = "openai"

                inferrer = AIInferrer()
                meta = _make_test_tech_meta()
                result = await inferrer.extract(
                    "/fake/video.mp4", meta,
                    keyframe_paths=keyframe_paths,
                )

                # Verify it parsed correctly
                assert result["inferred_tool"] == "可灵1.6"
                assert result["inferred_tool_confidence"] == 0.85

                # Verify it called the OpenAI endpoint
                call_args = mock_client.post.call_args
                url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
                assert "openai" in url.lower()


@pytest.mark.asyncio
async def test_progress_reporting():
    """Verify that progress callbacks are invoked with expected percentages."""
    from backend.modules.ai_inferrer import AIInferrer

    with tempfile.TemporaryDirectory() as tmpdir:
        keyframe_paths = _write_test_keyframes(tmpdir, count=2)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": _valid_llm_json_string()}],
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.modules.ai_inferrer.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("backend.modules.ai_inferrer.settings") as mock_settings:
                mock_settings.llm_api_key = "test-api-key"
                mock_settings.llm_provider = "claude"

                progress_log: list[tuple] = []

                def _progress(module: str, pct: float, msg: str) -> None:
                    progress_log.append((module, pct, msg))

                inferrer = AIInferrer()
                meta = _make_test_tech_meta()
                result = await inferrer.extract(
                    "/fake/video.mp4", meta,
                    progress_cb=_progress,
                    keyframe_paths=keyframe_paths,
                )

                assert len(progress_log) > 0
                pcts = [p for _, p, _ in progress_log]
                assert pcts[0] == 0.0
                assert pcts[-1] == 100.0
                # Should have at least the major milestones
                assert any(p == 20.0 for p in pcts) or any(19 < p < 21 for p in pcts), \
                    f"Missing 20% progress step (loading keyframes): {pcts}"
                assert any(p == 50.0 for p in pcts) or any(49 < p < 51 for p in pcts), \
                    f"Missing 50% progress step (calling LLM): {pcts}"
                assert any(p == 90.0 for p in pcts) or any(89 < p < 91 for p in pcts), \
                    f"Missing 90% progress step (parsing response): {pcts}"


@pytest.mark.asyncio
async def test_extract_empty_keyframes():
    """When no keyframe paths are provided, extract returns empty result."""
    from backend.modules.ai_inferrer import AIInferrer

    with patch("backend.modules.ai_inferrer.settings") as mock_settings:
        mock_settings.llm_api_key = "test-api-key"
        mock_settings.llm_provider = "claude"

        inferrer = AIInferrer()
        meta = _make_test_tech_meta()
        result = await inferrer.extract(
            "/fake/video.mp4", meta,
            keyframe_paths=[],
        )

        assert result["inferred_tool"] is None
        assert result["overall_confidence"] == 0.0
