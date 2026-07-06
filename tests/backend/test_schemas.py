"""Tests for Pydantic data models (Task 2)."""
import pytest
from backend.models.schemas import (
    TechMetadata,
    VisualAnalysis,
    AudioAnalysis,
    AIInference,
    SourceRecoveryHit,
    AnalysisResult,
    ModuleStatus,
)


def test_tech_metadata_parses_ffmpeg_output():
    data = {
        "container_format": "mp4",
        "video_codec": "H.264",
        "video_profile": "High@L4",
        "resolution_width": 1920,
        "resolution_height": 1080,
        "frame_rate": 30.0,
        "total_bitrate_bps": 8_000_000,
        "video_bitrate_bps": 6_500_000,
        "audio_codec": "AAC-LC",
        "audio_sample_rate_hz": 44100,
        "audio_bitrate_bps": 128_000,
        "gop_structure": "GOP=60",
        "color_space": "BT.709",
        "hdr_info": "SDR",
        "duration": 32.0,
        "file_size_bytes": 15_200_000,
        "platform_fingerprint": "Suspected Bilibili re-encode",
    }
    tm = TechMetadata(**data)
    assert tm.resolution_width == 1920
    assert tm.duration == 32.0


def test_source_recovery_hit_has_mandatory_fields():
    hit = SourceRecoveryHit(
        status="complete_match",
        source_url="https://civitai.com/images/1234567",
        similarity=94.5,
        hit_keyframes=2,
        total_keyframes_sent=2,
        workflow_json='{"nodes": []}',
        prompt="Cyberpunk city night scene",
        seed=1234567890,
        sampler="dpmpp_2m",
        steps=28,
        cfg_scale=7.0,
        model_name="dreamshaper_8.safetensors",
        confidence_score=0.94,
    )
    assert hit.status == "complete_match"
    assert hit.confidence_score == 0.94


def test_analysis_result_combines_all_modules():
    result = AnalysisResult(
        file_hash="abc123",
        file_path="/videos/test.mp4",
        tech_metadata=TechMetadata(
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
            platform_fingerprint=None,
        ),
        visual_analysis=None,
        audio_analysis=None,
        ai_inference=None,
        source_recovery=None,
        module_status={
            "tech": "completed",
            "visual": "skipped",
            "audio": "skipped",
            "ai": "skipped",
            "source_recovery": "skipped",
        },
    )
    assert result.tech_metadata is not None
    assert result.module_status["tech"] == "completed"
