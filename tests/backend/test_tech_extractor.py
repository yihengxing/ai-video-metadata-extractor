import pytest
from unittest.mock import patch, AsyncMock
from backend.modules.tech_extractor import TechExtractor
from backend.models.schemas import TechMetadata


@pytest.mark.asyncio
async def test_tech_extractor_returns_metadata():
    mock_probe = {
        "container_format": "mp4",
        "video_codec": "H.264",
        "video_profile": "High",
        "resolution_width": 1920,
        "resolution_height": 1080,
        "frame_rate": 30.0,
        "total_bitrate_bps": 8_000_000,
        "video_bitrate_bps": 6_500_000,
        "audio_codec": "AAC",
        "audio_sample_rate_hz": 44100,
        "audio_bitrate_bps": 128_000,
        "gop_structure": "GOP=60",
        "color_space": "BT.709",
        "hdr_info": "SDR",
        "duration": 32.0,
        "file_size_bytes": 15_200_000,
        "platform_fingerprint": "疑似B站二压（GOP对齐平台标准）",
    }

    with patch("backend.modules.tech_extractor.FFmpegService") as mock_svc:
        mock_svc.probe = AsyncMock(return_value={"raw": "probe"})
        mock_svc.parse.return_value = mock_probe
        mock_svc.is_installed.return_value = True

        extractor = TechExtractor()
        # Pass a minimal dummy TechMetadata (required by base signature)
        dummy_meta = TechMetadata(**mock_probe)
        result = await extractor.extract("/fake/video.mp4", dummy_meta)

    assert result["container_format"] == "mp4"
    assert result["resolution_width"] == 1920
    assert result["frame_rate"] == 30.0


def test_tech_extractor_module_name():
    assert TechExtractor().module_name == "tech"
