"""Audio Analyzer — faster-whisper speech-to-text (Task 12).

Extracts the audio track from a video via FFmpeg, then transcribes it
using faster-whisper with word-level timestamps.  Falls back gracefully
when dependencies are missing or the video has no audio stream.
"""
from __future__ import annotations
import asyncio
import logging
import os
import shutil
import tempfile
from typing import Optional

from backend.modules.base import Extractor, ProgressCallback
from backend.models.schemas import TechMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import guard for faster-whisper (the model is ~1.5 GB)
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel  # noqa: F401
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False


# ---------------------------------------------------------------------------
# FFmpeg path resolution (mirrors ffmpeg_service.py)
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    """Locate the ffmpeg executable."""
    bundled = os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "ffmpeg.exe")
    if os.path.exists(bundled):
        return bundled
    bundled_nix = os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "ffmpeg")
    if os.path.exists(bundled_nix):
        return bundled_nix
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise RuntimeError("FFmpeg 未找到，请安装 FFmpeg")


# ---------------------------------------------------------------------------
# AudioAnalyzer
# ---------------------------------------------------------------------------

class AudioAnalyzer(Extractor):
    """Speech-to-text analysis via faster-whisper.

    The WhisperModel is lazy-loaded on the first call and cached for the
    lifetime of the instance so that subsequent videos reuse the model.
    """

    def __init__(self) -> None:
        self._whisper_model: Optional[object] = None

    # ------------------------------------------------------------------
    # Extractor interface
    # ------------------------------------------------------------------

    @property
    def module_name(self) -> str:
        return "audio"

    async def extract(
        self,
        file_path: str,
        tech_meta: TechMetadata,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> dict:
        """Run the full audio extraction + transcription pipeline.

        Returns a dict matching :class:`AudioAnalysis`.
        """
        if progress_cb:
            progress_cb("audio", 0.0, "开始音频分析...")

        # --- Check for audio stream -----------------------------------
        if not self._has_audio_track(tech_meta):
            logger.info("Video has no audio track — returning empty result")
            if progress_cb:
                progress_cb("audio", 100.0, "无音轨，跳过")
            return self._empty_result()

        # --- Step 1: extract audio WAV --------------------------------
        if progress_cb:
            progress_cb("audio", 5.0, "提取音轨...")
        wav_path = await self._extract_audio(file_path)
        if progress_cb:
            progress_cb("audio", 20.0, "音轨提取完成")

        try:
            # --- Guard: faster-whisper available? ---------------------
            if not _WHISPER_AVAILABLE:
                logger.warning(
                    "faster-whisper 未安装，跳过语音转文字。"
                    " 请执行: pip install faster-whisper"
                )
                if progress_cb:
                    progress_cb("audio", 100.0, "faster-whisper 未安装，跳过转写")
                return self._empty_result()

            # --- Step 2: transcribe -----------------------------------
            if progress_cb:
                progress_cb("audio", 30.0, "加载 Whisper 模型...")
            full_text, text_segments = await asyncio.to_thread(
                self._transcribe, wav_path
            )
            if progress_cb:
                progress_cb("audio", 80.0, "转写完成")

            # --- Step 3: speech rate ----------------------------------
            speech_rate = self._calc_speech_rate(full_text, text_segments)

            if progress_cb:
                progress_cb("audio", 100.0, "音频分析完成")

            return {
                "full_text": full_text,
                "text_segments": text_segments,
                "speech_rate": round(speech_rate, 2),
                # Placeholder fields — filled by Tasks 13 & 14
                "speech_emotion": "",
                "bgm_title": None,
                "bgm_artist": None,
                "bgm_style_tags": [],
                "bgm_emotion": None,
                "bgm_bpm": None,
                "sound_events": [],
                "voice_to_bg_ratio": None,
                "audio_structure": None,
            }
        finally:
            # Clean up the temporary WAV
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Audio extraction (Step 1)
    # ------------------------------------------------------------------

    async def _extract_audio(
        self, video_path: str, output_wav: Optional[str] = None
    ) -> str:
        """Extract mono 16 kHz PCM WAV from *video_path* using FFmpeg.

        If *output_wav* is not given a temporary file is created.
        Returns the path to the WAV file (caller is responsible for cleanup).
        """
        if output_wav is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            output_wav = tmp.name

        ffmpeg_path = _find_ffmpeg()
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-vn",               # drop video
            "-acodec", "pcm_s16le",
            "-ar", "16000",      # 16 kHz sample rate
            "-ac", "1",          # mono
            output_wav,
            "-y",                # overwrite
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Clean up partial file
            try:
                os.unlink(output_wav)
            except OSError:
                pass
            raise RuntimeError(
                f"FFmpeg 音频提取失败 (code={proc.returncode}): {stderr.decode()}"
            )

        return output_wav

    # ------------------------------------------------------------------
    # Speech-to-text (Step 2)
    # ------------------------------------------------------------------

    def _transcribe(self, wav_path: str) -> tuple[str, list[dict]]:
        """Run faster-whisper on *wav_path*.

        Returns ``(full_text, segments)`` where *segments* is a list of
        ``{"text": str, "start": float, "end": float}`` dicts.

        The WhisperModel is created once and cached on ``self._whisper_model``.
        """
        if self._whisper_model is None:
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel(
                "medium", device="cpu", compute_type="int8"
            )

        segments, _info = self._whisper_model.transcribe(
            wav_path, language="zh", beam_size=5
        )

        full_text_parts: list[str] = []
        seg_list: list[dict] = []

        for seg in segments:
            full_text_parts.append(seg.text)
            seg_list.append({
                "text": seg.text,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
            })

        return "".join(full_text_parts), seg_list

    # ------------------------------------------------------------------
    # Speech rate (Step 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_speech_rate(full_text: str, segments: list[dict]) -> float:
        """Characters per second of actual speech time."""
        if not segments:
            return 0.0
        total_duration = sum(
            max(seg["end"] - seg["start"], 0.0) for seg in segments
        )
        if total_duration <= 0.0:
            return 0.0
        return len(full_text) / total_duration

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_audio_track(tech_meta: TechMetadata) -> bool:
        """Return True if *tech_meta* indicates an audio stream is present."""
        # A valid audio codec and non-zero sample rate means audio exists
        return bool(
            tech_meta.audio_codec
            and tech_meta.audio_codec.lower() not in ("none", "unknown", "")
            and tech_meta.audio_sample_rate_hz > 0
        )

    @staticmethod
    def _empty_result() -> dict:
        """Return a result dict with all placeholder/default values."""
        return {
            "full_text": "",
            "text_segments": [],
            "speech_rate": 0.0,
            "speech_emotion": "",
            "bgm_title": None,
            "bgm_artist": None,
            "bgm_style_tags": [],
            "bgm_emotion": None,
            "bgm_bpm": None,
            "sound_events": [],
            "voice_to_bg_ratio": None,
            "audio_structure": None,
        }
