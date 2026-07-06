"""Tests for ExportService (Tasks 20-22)."""
import json
import pytest
from datetime import datetime, timezone

from backend.models.schemas import (
    TechMetadata,
    VisualAnalysis,
    AudioAnalysis,
    AIInference,
    SourceRecoveryHit,
    AnalysisResult,
    ShotItem,
    ColorSummary,
)
from backend.services.export_service import ExportService


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_tech_metadata() -> TechMetadata:
    return TechMetadata(
        container_format="mp4",
        video_codec="H.264",
        video_profile="High@L4",
        resolution_width=1920,
        resolution_height=1080,
        frame_rate=30.0,
        total_bitrate_bps=8_000_000,
        video_bitrate_bps=6_500_000,
        audio_codec="AAC-LC",
        audio_sample_rate_hz=44100,
        audio_bitrate_bps=128_000,
        gop_structure="GOP=60",
        color_space="BT.709",
        hdr_info="SDR",
        duration=32.0,
        file_size_bytes=15_200_000,
        platform_fingerprint="Suspected Bilibili re-encode",
    )


def _make_minimal_result(**overrides) -> AnalysisResult:
    """Build a complete AnalysisResult for export testing."""
    kwargs = dict(
        file_hash="abc123def456",
        file_path="/videos/test.mp4",
        analyzed_at=datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc),
        tech_metadata=_make_tech_metadata(),
        visual_analysis=VisualAnalysis(
            shots=[
                ShotItem(index=0, start_time=0.0, end_time=3.5, duration=3.5),
                ShotItem(index=1, start_time=3.5, end_time=7.0, duration=3.5),
            ],
            shot_count=2,
            avg_shot_duration=3.5,
            transitions=["hard cut"],
            color_summary=ColorSummary(
                dominant_hue="warm",
                saturation="high",
                description="cyan-orange",
            ),
            motion_summary="slight motion",
        ),
        audio_analysis=AudioAnalysis(
            full_text="你好世界，这是一段测试语音。",
            text_segments=[
                {"text": "你好世界", "start": 0.0, "end": 2.0},
                {"text": "这是一段测试语音", "start": 2.0, "end": 5.5},
            ],
            speech_rate=4.2,
            speech_emotion="calm",
            bgm_title="Summer Breeze",
            bgm_artist="Cool Artist",
            bgm_style_tags=["lo-fi", "chill"],
            bgm_emotion="relaxed",
            bgm_bpm=85,
        ),
        ai_inference=AIInference(
            inferred_tool="Suspected Kling 1.6 + Topaz enhancement",
            inferred_tool_confidence=0.85,
            inferred_prompt="A futuristic city at night with neon lights",
            inferred_prompt_confidence=0.72,
            style_tags=["cyberpunk", "neon", "night"],
            inferred_workflow="text-to-video -> image-to-video refine",
            inferred_workflow_confidence=0.68,
            imitation_suggestions=["Use Kling 1.6 for similar results"],
            model_recommendations=["Kling 1.6", "Topaz Video AI"],
            overall_confidence=0.70,
        ),
        source_recovery=SourceRecoveryHit(
            status="complete_match",
            source_url="https://civitai.com/images/1234567",
            similarity=94.5,
            hit_keyframes=3,
            total_keyframes_sent=5,
            workflow_json='{"nodes":[{"id":1,"type":"KSampler"}]}',
            prompt="Cyberpunk city night scene, 8K, hyper-detailed",
            seed=1234567890,
            sampler="dpmpp_2m",
            steps=28,
            cfg_scale=7.0,
            model_name="dreamshaper_8.safetensors",
            confidence_score=0.94,
            source_trust="civitai",
        ),
        module_status={
            "tech": "completed",
            "visual": "completed",
            "audio": "completed",
            "ai": "completed",
            "source_recovery": "completed",
        },
    )
    kwargs.update(overrides)
    return AnalysisResult(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_export_json_returns_valid_json(self):
        result = _make_minimal_result()
        output = ExportService.export_json(result)
        data = json.loads(output)
        assert data["file_hash"] == "abc123def456"
        assert data["schema_version"] == "1.3.0"
        # export timestamp should be present
        assert "exported_at" in data
        # source_recovery and ai_inference should be sibling nodes
        assert "source_recovery" in data
        assert "ai_inference" in data
        assert data["source_recovery"]["status"] == "complete_match"
        assert data["ai_inference"]["inferred_tool"] is not None


class TestExportMarkdown:
    def test_export_markdown_contains_key_sections(self):
        result = _make_minimal_result()
        md = ExportService.export_markdown(result)
        assert "# AI 视频分析报告" in md
        assert "## 基本信息" in md
        assert "## 技术元数据" in md
        assert "## 镜头分析" in md
        assert "## 音频分析" in md
        assert "## AI 推断" in md
        assert "## 原始来源与生成参数" in md
        assert "/videos/test.mp4" in md
        assert "abc123def456" in md  # file_hash (full or truncated)
        assert "Kling" in md
        assert "Cyberpunk" in md
        # Table format
        assert "| 字段 | 值 |" in md


class TestExportComfyUIWorkflow:
    def test_with_hit_returns_workflow_json(self):
        result = _make_minimal_result()
        output = ExportService.export_comfyui_workflow(result)
        assert output is not None
        assert "KSampler" in output
        data = json.loads(output)
        assert "nodes" in data

    def test_without_hit_returns_none(self):
        result = _make_minimal_result(source_recovery=None)
        output = ExportService.export_comfyui_workflow(result)
        assert output is None


class TestExportComfyUIPrompt:
    def test_uses_source_recovery_prompt_when_available(self):
        result = _make_minimal_result()
        output = ExportService.export_comfyui_prompt(result)
        assert "(原始)" in output
        assert "Cyberpunk city night scene" in output
        # Should also include model recommendations
        assert "dreamshaper_8" in output

    def test_falls_back_to_ai_inference_prompt(self):
        result = _make_minimal_result(source_recovery=None)
        output = ExportService.export_comfyui_prompt(result)
        assert "AI 推断提示词" in output
        assert "A futuristic city at night with neon lights" in output
        assert "Kling 1.6" in output


class TestExportSrt:
    def test_export_srt_with_segments(self):
        result = _make_minimal_result()
        srt = ExportService.export_srt(result)
        assert "1" in srt
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "你好世界" in srt
        assert "2" in srt
        assert "这是一段测试语音" in srt

    def test_export_srt_without_segments_returns_empty(self):
        result = _make_minimal_result(
            audio_analysis=AudioAnalysis(full_text="", text_segments=[]),
        )
        srt = ExportService.export_srt(result)
        assert srt == ""


class TestExportAll:
    def test_export_all_returns_all_keys(self):
        result = _make_minimal_result()
        exports = ExportService.export_all(result)
        assert set(exports.keys()) == {
            "json", "markdown", "comfyui_workflow", "comfyui_prompt", "srt",
        }
        assert isinstance(exports["json"], str)
        assert isinstance(exports["markdown"], str)
        assert isinstance(exports["comfyui_workflow"], str)
        assert isinstance(exports["comfyui_prompt"], str)
        assert isinstance(exports["srt"], str)
