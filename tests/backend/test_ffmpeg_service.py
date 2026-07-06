import json
import pytest
from unittest.mock import patch, AsyncMock
from backend.services.ffmpeg_service import parse_ffprobe_output, FFmpegService


SAMPLE_FFPROBE = {
    "format": {
        "filename": "test.mp4",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "32.0",
        "size": "15200000",
        "bit_rate": "8000000"
    },
    "streams": [
        {
            "index": 0,
            "codec_name": "h264",
            "codec_long_name": "H.264 / AVC / MPEG-4 AVC",
            "profile": "High",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30/1",
            "bit_rate": "6500000",
            "color_space": "bt709",
            "color_transfer": "bt709",
            "pix_fmt": "yuv420p"
        },
        {
            "index": 1,
            "codec_name": "aac",
            "codec_long_name": "AAC (Advanced Audio Coding)",
            "sample_rate": "44100",
            "bit_rate": "128000",
            "channels": 2
        }
    ]
}


def test_parse_ffprobe_output_extracts_container():
    result = parse_ffprobe_output(SAMPLE_FFPROBE)
    assert result["container_format"] == "mp4"
    assert result["video_codec"] == "H.264"
    assert result["video_profile"] == "High"
    assert result["resolution_width"] == 1920
    assert result["resolution_height"] == 1080
    assert result["frame_rate"] == 30.0
    assert result["total_bitrate_bps"] == 8_000_000
    assert result["video_bitrate_bps"] == 6_500_000
    assert result["audio_codec"] == "AAC-LC"
    assert result["audio_sample_rate_hz"] == 44100
    assert result["audio_bitrate_bps"] == 128_000
    assert result["color_space"] == "BT.709"
    assert result["hdr_info"] == "SDR"
    assert result["duration"] == 32.0
    assert result["file_size_bytes"] == 15_200_000
