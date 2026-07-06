"""Tests for AudioAnalyzer — faster-whisper speech-to-text (Task 12)."""
from __future__ import annotations
import pytest
import os
import sys
import tempfile
import wave
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from backend.models.schemas import TechMetadata


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_wav_file(path: str, duration_secs: float = 2.0,
                   sample_rate: int = 16000) -> str:
    """Create a minimal mono 16-bit PCM WAV file at *path*."""
    import struct
    n_frames = int(sample_rate * duration_secs)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write a 440 Hz sine wave
        for i in range(n_frames):
            value = int(16000 * 0.5 * (  # half amplitude
                __import__("math").sin(2.0 * 3.14159 * 440.0 * i / sample_rate)
            ))
            wf.writeframes(struct.pack("<h", max(-32768, min(32767, value))))
    return path


def _make_test_tech_meta(**overrides) -> TechMetadata:
    """Return a minimal TechMetadata for testing."""
    defaults = dict(
        container_format="mp4",
        video_codec="H.264",
        video_profile="High",
        resolution_width=320,
        resolution_height=240,
        frame_rate=30.0,
        total_bitrate_bps=1_000_000,
        video_bitrate_bps=800_000,
        audio_codec="AAC",
        audio_sample_rate_hz=44100,
        audio_bitrate_bps=128_000,
        gop_structure="GOP=30",
        color_space="BT.709",
        hdr_info="SDR",
        duration=3.0,
        file_size_bytes=500_000,
        platform_fingerprint=None,
    )
    defaults.update(overrides)
    return TechMetadata(**defaults)


def _install_mock_faster_whisper():
    """Create and inject a mock ``faster_whisper`` module into sys.modules
    so that ``from faster_whisper import WhisperModel`` succeeds during tests.

    Returns (mock_module, MockWhisperModel) for the caller to configure.
    """
    mock_module = MagicMock()
    MockWhisperModel = MagicMock()
    mock_module.WhisperModel = MockWhisperModel
    sys.modules["faster_whisper"] = mock_module
    return mock_module, MockWhisperModel


def _remove_mock_faster_whisper():
    """Clean up the mock module so later imports are unaffected."""
    sys.modules.pop("faster_whisper", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_audio_analyzer_module_name():
    """Verify the module_name property returns "audio"."""
    from backend.modules.audio_analyzer import AudioAnalyzer
    analyzer = AudioAnalyzer()
    assert analyzer.module_name == "audio"


# ===================================================================
# Audio extraction (Step 1 — FFmpeg subprocess)
# ===================================================================

@pytest.mark.asyncio
async def test_extract_audio_extraction():
    """Mock the FFmpeg subprocess and verify _extract_audio constructs the
    correct command and creates a non-empty WAV file."""
    from backend.modules.audio_analyzer import AudioAnalyzer

    # Prepare a real WAV file — our mock "FFmpeg" will copy it to the output
    src_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    src_wav.close()
    _make_wav_file(src_wav.name, duration_secs=1.0)

    dest_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    dest_wav.close()

    try:
        async def _fake_subprocess(*args, **kwargs):
            # Simulate FFmpeg: copy the source WAV to the destination
            import shutil
            # The output path is the second-to-last arg (before "-y")
            cmd = args[0] if args else []
            # Find output path — it's the argument before "-y"
            output_path = cmd[-2] if len(cmd) >= 2 and cmd[-1] == "-y" else dest_wav.name
            shutil.copy(src_wav.name, output_path)

            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_subprocess), \
             patch("backend.modules.audio_analyzer._find_ffmpeg", return_value="ffmpeg"):
            analyzer = AudioAnalyzer()
            result_path = await analyzer._extract_audio("/fake/video.mp4", dest_wav.name)
            assert result_path == dest_wav.name
            assert os.path.exists(result_path), "WAV file should exist"
            assert os.path.getsize(result_path) > 0, "WAV file should be non-empty"
    finally:
        for p in (src_wav.name, dest_wav.name):
            try:
                os.unlink(p)
            except OSError:
                pass


@pytest.mark.asyncio
async def test_extract_audio_ffmpeg_error():
    """When FFmpeg returns non-zero, _extract_audio should raise RuntimeError."""
    from backend.modules.audio_analyzer import AudioAnalyzer

    async def _fake_subprocess(*args, **kwargs):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"FFmpeg error"))
        return mock_proc

    dest_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    dest_wav.close()
    try:
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_subprocess), \
             patch("backend.modules.audio_analyzer._find_ffmpeg", return_value="ffmpeg"):
            analyzer = AudioAnalyzer()
            with pytest.raises(RuntimeError, match="FFmpeg"):
                await analyzer._extract_audio("/fake/video.mp4", dest_wav.name)
    finally:
        try:
            os.unlink(dest_wav.name)
        except OSError:
            pass


# ===================================================================
# No-audio-track handling
# ===================================================================

def test_has_audio_track_true():
    """_has_audio_track returns True when audio codec and sample rate are valid."""
    from backend.modules.audio_analyzer import AudioAnalyzer
    meta = _make_test_tech_meta(audio_codec="AAC", audio_sample_rate_hz=44100)
    assert AudioAnalyzer._has_audio_track(meta) is True


def test_has_audio_track_false_none_codec():
    """_has_audio_track returns False when audio_codec is 'none'."""
    from backend.modules.audio_analyzer import AudioAnalyzer
    meta = _make_test_tech_meta(audio_codec="none", audio_sample_rate_hz=0)
    assert AudioAnalyzer._has_audio_track(meta) is False


def test_has_audio_track_false_empty():
    """_has_audio_track returns False when audio fields are empty."""
    from backend.modules.audio_analyzer import AudioAnalyzer
    meta = _make_test_tech_meta(audio_codec="", audio_sample_rate_hz=0)
    assert AudioAnalyzer._has_audio_track(meta) is False


@pytest.mark.asyncio
async def test_extract_no_audio_track():
    """When tech_meta has no audio, extract returns empty results immediately."""
    from backend.modules.audio_analyzer import AudioAnalyzer
    analyzer = AudioAnalyzer()
    meta = _make_test_tech_meta(audio_codec="none", audio_sample_rate_hz=0)
    result = await analyzer.extract("/fake/noaudio.mp4", meta)
    assert result["full_text"] == ""
    assert result["text_segments"] == []
    assert result["speech_rate"] == 0.0


# ===================================================================
# Speech-to-text (Step 2 — faster-whisper)
# ===================================================================

@pytest.mark.asyncio
async def test_transcribe_returns_segments():
    """Mock faster_whisper and verify that _transcribe returns correct segments."""
    _install_mock_faster_whisper()
    try:
        # Force-reload the module so _WHISPER_AVAILABLE becomes True
        if "backend.modules.audio_analyzer" in sys.modules:
            del sys.modules["backend.modules.audio_analyzer"]
        from backend.modules.audio_analyzer import AudioAnalyzer

        # Build mock segment objects with .text, .start, .end
        class MockSegment:
            def __init__(self, text, start, end):
                self.text = text
                self.start = start
                self.end = end

        mock_segments = [
            MockSegment("你好世界", 0.0, 1.5),
            MockSegment("这是测试", 1.5, 3.0),
        ]
        mock_info = MagicMock()

        from faster_whisper import WhisperModel as MockWhisper
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (mock_segments, mock_info)
        MockWhisper.return_value = mock_model

        analyzer = AudioAnalyzer()
        full_text, segs = analyzer._transcribe("/fake/audio.wav")

        assert full_text == "你好世界这是测试"
        assert len(segs) == 2
        assert segs[0] == {"text": "你好世界", "start": 0.0, "end": 1.5}
        assert segs[1] == {"text": "这是测试", "start": 1.5, "end": 3.0}

        MockWhisper.assert_called_once_with("medium", device="cpu", compute_type="int8")
    finally:
        _remove_mock_faster_whisper()
        # Clean module cache so later tests get a fresh import
        sys.modules.pop("backend.modules.audio_analyzer", None)


@pytest.mark.asyncio
async def test_transcribe_model_cached():
    """WhisperModel should be created only once (lazy-load + cache)."""
    _install_mock_faster_whisper()
    try:
        if "backend.modules.audio_analyzer" in sys.modules:
            del sys.modules["backend.modules.audio_analyzer"]
        from backend.modules.audio_analyzer import AudioAnalyzer

        from faster_whisper import WhisperModel as MockWhisper
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        MockWhisper.return_value = mock_model

        analyzer = AudioAnalyzer()
        analyzer._transcribe("/fake/a.wav")
        analyzer._transcribe("/fake/b.wav")

        # Model should have been constructed only once
        assert MockWhisper.call_count == 1
    finally:
        _remove_mock_faster_whisper()
        sys.modules.pop("backend.modules.audio_analyzer", None)


# ===================================================================
# Full extract() pipeline with mocked dependencies
# ===================================================================

@pytest.mark.asyncio
async def test_extract_with_mocked_whisper():
    """Full extract() pipeline with mocked FFmpeg + Whisper + BGM/Speech — verify return dict."""
    _install_mock_faster_whisper()
    try:
        if "backend.modules.audio_analyzer" in sys.modules:
            del sys.modules["backend.modules.audio_analyzer"]
        from backend.modules.audio_analyzer import AudioAnalyzer

        # Prepare a real WAV file for our mock FFmpeg to "produce"
        src_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        src_wav.close()
        _make_wav_file(src_wav.name, duration_secs=2.0)

        try:
            # --- Mock FFmpeg subprocess ---
            async def _fake_ffmpeg(*args, **kwargs):
                import shutil
                cmd = args[0] if args else []
                # Output path is the argument before "-y"
                output_path = cmd[-2] if len(cmd) >= 2 and cmd[-1] == "-y" else ""
                if output_path and os.path.exists(src_wav.name):
                    shutil.copy(src_wav.name, output_path)
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                return mock_proc

            # --- Mock Whisper segments ---
            class MockSegment:
                def __init__(self, text, start, end):
                    self.text = text
                    self.start = start
                    self.end = end

            mock_segments = [
                MockSegment("这是第一段", 0.0, 2.0),
                MockSegment("这是第二段", 2.0, 4.0),
            ]

            from faster_whisper import WhisperModel as MockWhisper
            mock_model = MagicMock()
            mock_model.transcribe.return_value = (mock_segments, MagicMock())
            MockWhisper.return_value = mock_model

            with patch("asyncio.create_subprocess_exec", side_effect=_fake_ffmpeg), \
                 patch("backend.modules.audio_analyzer._find_ffmpeg", return_value="ffmpeg"):
                analyzer = AudioAnalyzer()

                # Mock Task 13 methods to avoid network/librosa dependencies
                analyzer._identify_bgm = AsyncMock(return_value={
                    "title": "Test Song",
                    "artist": "Test Artist",
                    "style_tags": ["pop", "electronic"],
                })
                analyzer._analyze_bgm_features = MagicMock(return_value={
                    "bpm": 128,
                    "emotion": "轻快电子",
                })
                analyzer._classify_speech_emotion = MagicMock(return_value="激昂")

                meta = _make_test_tech_meta()

                progress_log: list[tuple] = []

                def _progress(module: str, pct: float, msg: str) -> None:
                    progress_log.append((module, pct, msg))

                result = await analyzer.extract("/fake/video.mp4", meta,
                                                progress_cb=_progress)

                # --- Verify all required keys ---
                for key in ("full_text", "text_segments", "speech_rate",
                            "speech_emotion", "bgm_title", "bgm_artist",
                            "bgm_style_tags", "bgm_emotion", "bgm_bpm",
                            "sound_events", "voice_to_bg_ratio", "audio_structure"):
                    assert key in result, f"Missing key: {key}"

                # --- Verify populated fields (Task 12) ---
                assert isinstance(result["full_text"], str)
                assert len(result["full_text"]) > 0
                assert len(result["text_segments"]) == 2
                assert result["speech_rate"] > 0

                # --- Task 13 fields populated by mocked methods ---
                assert result["speech_emotion"] == "激昂"
                assert result["bgm_title"] == "Test Song"
                assert result["bgm_artist"] == "Test Artist"
                assert result["bgm_style_tags"] == ["pop", "electronic"]
                assert result["bgm_emotion"] == "轻快电子"
                assert result["bgm_bpm"] == 128

                # --- Placeholder fields (Task 14) ---
                assert result["sound_events"] == []
                assert result["voice_to_bg_ratio"] is None
                assert result["audio_structure"] is None

                # --- Progress reporting ---
                assert len(progress_log) > 0
                pcts = [p for _, p, _ in progress_log]
                assert pcts[0] == 0.0
                assert pcts[-1] == 100.0
        finally:
            try:
                os.unlink(src_wav.name)
            except OSError:
                pass
    finally:
        _remove_mock_faster_whisper()
        sys.modules.pop("backend.modules.audio_analyzer", None)


# ===================================================================
# Speech rate
# ===================================================================

@pytest.mark.asyncio
async def test_speech_rate_calculation():
    """Verify speech_rate = total chars / total speech duration."""
    from backend.modules.audio_analyzer import AudioAnalyzer

    analyzer = AudioAnalyzer()
    segments = [
        {"text": "你好世界", "start": 0.0, "end": 2.0},
        {"text": "测试文本", "start": 2.0, "end": 4.0},
    ]
    full_text = "你好世界测试文本"  # 8 chars
    rate = analyzer._calc_speech_rate(full_text, segments)
    assert rate == pytest.approx(2.0)  # 8 chars / 4.0 seconds

    # Empty input
    assert analyzer._calc_speech_rate("", []) == 0.0

    # Zero-duration guard
    assert analyzer._calc_speech_rate("test", [{"text": "test", "start": 0.0, "end": 0.0}]) == 0.0


# ===================================================================
# Task 13: BGM features & speech emotion
# ===================================================================

def _make_silent_wav(path: str, duration_secs: float = 2.0,
                     sample_rate: int = 16000) -> str:
    """Create a silent (all zeros) mono 16-bit PCM WAV file at *path*."""
    import struct
    n_frames = int(sample_rate * duration_secs)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for _ in range(n_frames):
            wf.writeframes(struct.pack("<h", 0))
    return path


def test_bgm_features_on_silence():
    """Run _analyze_bgm_features on a silent WAV and verify return dict keys."""
    from backend.modules.audio_analyzer import AudioAnalyzer, _LIBROSA_AVAILABLE

    # Create a silent WAV file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _make_silent_wav(tmp.name, duration_secs=2.0)

    try:
        analyzer = AudioAnalyzer()
        result = analyzer._analyze_bgm_features(tmp.name)

        # Always check that the result dict has the expected keys
        assert isinstance(result, dict), "Result should be a dict"
        assert "bpm" in result, "Missing key: bpm"
        assert "emotion" in result, "Missing key: emotion"

        if _LIBROSA_AVAILABLE:
            # librosa is available — result should have computed values
            # (silent audio may still produce a BPM estimate, possibly None)
            assert result["bpm"] is None or isinstance(result["bpm"], int)
            assert result["emotion"] is not None, \
                "emotion should be a string when librosa is available"
        else:
            # Graceful degradation — both values are None
            assert result["bpm"] is None
            assert result["emotion"] is None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def test_speech_emotion_returns_valid_category():
    """Verify _classify_speech_emotion returns one of the 5 valid categories."""
    from backend.modules.audio_analyzer import AudioAnalyzer, _LIBROSA_AVAILABLE

    # Create a regular (sine-wave) WAV to test classification
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _make_wav_file(tmp.name, duration_secs=2.0)

    valid_categories = {"平静", "激昂", "悲伤", "幽默", "中性", ""}

    try:
        analyzer = AudioAnalyzer()
        emotion = analyzer._classify_speech_emotion(tmp.name)

        assert isinstance(emotion, str), "Emotion should be a string"
        assert emotion in valid_categories, \
            f"Got '{emotion}', expected one of {valid_categories}"

        if _LIBROSA_AVAILABLE:
            # When librosa is available, should return a non-empty category
            assert emotion in {"平静", "激昂", "悲伤", "幽默", "中性"}, \
                f"Expected valid emotion category, got '{emotion}'"
        else:
            # Graceful degradation — returns empty string
            assert emotion == "", \
                f"Expected empty string when librosa unavailable, got '{emotion}'"
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_bgm_identify_no_config():
    """_identify_bgm returns None values when no API keys are configured."""
    from backend.modules.audio_analyzer import AudioAnalyzer

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _make_silent_wav(tmp.name, duration_secs=1.0)

    try:
        # Mock httpx to avoid real network calls
        with patch("httpx.AsyncClient", side_effect=Exception("no network in test")):
            analyzer = AudioAnalyzer()
            result = await analyzer._identify_bgm(tmp.name)

            assert isinstance(result, dict)
            assert "title" in result
            assert "artist" in result
            assert "style_tags" in result
            # No API keys configured → should return None values
            assert result["title"] is None
            assert result["artist"] is None
            assert result["style_tags"] == []
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_bgm_features_no_librosa_graceful():
    """_analyze_bgm_features returns None when librosa is not available."""
    # Remove librosa from sys.modules to simulate unavailability
    librosa_in_modules = "librosa" in sys.modules
    saved_librosa = sys.modules.pop("librosa", None)

    # Force-reload the module to pick up _LIBROSA_AVAILABLE = False
    saved_audio = sys.modules.pop("backend.modules.audio_analyzer", None)
    try:
        from backend.modules.audio_analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        result = analyzer._analyze_bgm_features("/nonexistent/audio.wav")

        assert result == {"bpm": None, "emotion": None}
    finally:
        # Restore module state
        if saved_audio is not None:
            sys.modules["backend.modules.audio_analyzer"] = saved_audio
        if librosa_in_modules and saved_librosa is not None:
            sys.modules["librosa"] = saved_librosa


@pytest.mark.asyncio
async def test_speech_emotion_no_librosa_graceful():
    """_classify_speech_emotion returns empty string when librosa is not available."""
    librosa_in_modules = "librosa" in sys.modules
    saved_librosa = sys.modules.pop("librosa", None)

    saved_audio = sys.modules.pop("backend.modules.audio_analyzer", None)
    try:
        from backend.modules.audio_analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        emotion = analyzer._classify_speech_emotion("/nonexistent/audio.wav")

        assert emotion == ""
    finally:
        if saved_audio is not None:
            sys.modules["backend.modules.audio_analyzer"] = saved_audio
        if librosa_in_modules and saved_librosa is not None:
            sys.modules["librosa"] = saved_librosa


# ===================================================================
# Graceful degradation
# ===================================================================

@pytest.mark.asyncio
async def test_graceful_degradation_no_whisper():
    """When _WHISPER_AVAILABLE is False, extract returns empty results
    (after audio extraction succeeds)."""
    # Ensure faster_whisper is NOT in sys.modules so the module sets
    # _WHISPER_AVAILABLE = False at import time
    _remove_mock_faster_whisper()
    if "backend.modules.audio_analyzer" in sys.modules:
        del sys.modules["backend.modules.audio_analyzer"]

    from backend.modules.audio_analyzer import AudioAnalyzer

    async def _fake_ffmpeg(*args, **kwargs):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_ffmpeg), \
         patch("backend.modules.audio_analyzer._find_ffmpeg", return_value="ffmpeg"):
        analyzer = AudioAnalyzer()
        meta = _make_test_tech_meta()
        result = await analyzer.extract("/fake/video.mp4", meta)
        assert result["full_text"] == ""
        assert result["text_segments"] == []
        assert result["speech_rate"] == 0.0


def test_module_registers_in_init():
    """Verify AudioAnalyzer can be imported from backend.modules."""
    from backend.modules.audio_analyzer import AudioAnalyzer
    assert AudioAnalyzer is not None


def test_empty_result_keys():
    """_empty_result() should return all required AudioAnalysis keys."""
    from backend.modules.audio_analyzer import AudioAnalyzer
    result = AudioAnalyzer._empty_result()
    expected_keys = {
        "full_text", "text_segments", "speech_rate",
        "speech_emotion", "bgm_title", "bgm_artist",
        "bgm_style_tags", "bgm_emotion", "bgm_bpm",
        "sound_events", "voice_to_bg_ratio", "audio_structure",
    }
    assert set(result.keys()) == expected_keys
