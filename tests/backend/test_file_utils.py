import os
import tempfile
from backend.utils.file_utils import compute_sha256, validate_video_file, SUPPORTED_EXTENSIONS


def test_compute_sha256_returns_64_char_hex():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"hello world")
        tmp = f.name
    try:
        h = compute_sha256(tmp)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
        # Same content = same hash
        assert compute_sha256(tmp) == h
    finally:
        os.unlink(tmp)


def test_validate_video_file_accepts_mp4():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(b"fake mp4 content")
        tmp = f.name
    try:
        ok, msg = validate_video_file(tmp)
        assert ok is True
        assert msg == ""
    finally:
        os.unlink(tmp)


def test_validate_video_file_rejects_unsupported():
    ok, msg = validate_video_file("/nonexistent/file.txt")
    assert ok is False
    assert "不支持的格式" in msg or "不存在" in msg
