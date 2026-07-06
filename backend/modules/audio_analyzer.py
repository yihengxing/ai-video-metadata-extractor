"""Audio Analyzer — faster-whisper speech-to-text + BGM recognition (Task 12 + 13).

Extracts the audio track from a video via FFmpeg, transcribes it
using faster-whisper with word-level timestamps, identifies background
music (ACRCloud/AudD), and classifies speech emotion via librosa.
Falls back gracefully when dependencies are missing or the video has
no audio stream.
"""
from __future__ import annotations
import asyncio
import base64
import hashlib
import logging
import os
import shutil
import tempfile
import time
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
# Lazy-import guard for librosa (used by BGM features + speech emotion)
# ---------------------------------------------------------------------------
try:
    import librosa  # noqa: F401
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


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
                progress_cb("audio", 70.0, "转写完成")

            # --- Step 3: speech rate ----------------------------------
            speech_rate = self._calc_speech_rate(full_text, text_segments)

            # --- Step 4: BGM identification (API) --------------------
            if progress_cb:
                progress_cb("audio", 80.0, "识别背景音乐...")
            bgm_info = await self._identify_bgm(wav_path)

            # --- Step 5: BGM features (BPM + emotion) ----------------
            if progress_cb:
                progress_cb("audio", 88.0, "分析BGM特征...")
            bgm_features = await asyncio.to_thread(
                self._analyze_bgm_features, wav_path
            )

            # --- Step 6: speech emotion classification ---------------
            if progress_cb:
                progress_cb("audio", 95.0, "分析语音情感...")
            speech_emotion = await asyncio.to_thread(
                self._classify_speech_emotion, wav_path
            )

            if progress_cb:
                progress_cb("audio", 100.0, "音频分析完成")

            return {
                "full_text": full_text,
                "text_segments": text_segments,
                "speech_rate": round(speech_rate, 2),
                # Task 13: BGM recognition + audio features
                "speech_emotion": speech_emotion,
                "bgm_title": bgm_info.get("title"),
                "bgm_artist": bgm_info.get("artist"),
                "bgm_style_tags": bgm_info.get("style_tags", []),
                "bgm_emotion": bgm_features.get("emotion"),
                "bgm_bpm": bgm_features.get("bpm"),
                # Placeholder fields — filled by Task 14
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
    # BGM fingerprint recognition (Step 4)
    # ------------------------------------------------------------------

    async def _identify_bgm(self, wav_path: str) -> dict:
        """Identify background music via ACRCloud (primary) or AudD (fallback).

        Returns ``{"title": str|None, "artist": str|None, "style_tags": list[str]}``.
        If both APIs fail or are not configured, returns None values.
        """
        try:
            from backend.config import settings

            acrcloud_key = settings.get("acrcloud_key", "")
            acrcloud_secret = settings.get("acrcloud_secret", "")

            # Try ACRCloud first if credentials are configured
            if acrcloud_key and acrcloud_secret:
                try:
                    result = await self._identify_acrcloud(
                        wav_path, acrcloud_key, acrcloud_secret
                    )
                    if result["title"] is not None:
                        return result
                except Exception as exc:
                    logger.warning("ACRCloud identification failed: %s", exc)

            # Fallback to AudD
            try:
                result = await self._identify_audd(wav_path)
                if result["title"] is not None:
                    return result
            except Exception as exc:
                logger.warning("AudD identification failed: %s", exc)

            logger.warning("BGM identification: both ACRCloud and AudD failed or not configured")
            return {"title": None, "artist": None, "style_tags": []}
        except Exception as exc:
            logger.warning("BGM identification failed: %s", exc)
            return {"title": None, "artist": None, "style_tags": []}

    @staticmethod
    async def _identify_acrcloud(
        wav_path: str, key: str, secret: str
    ) -> dict:
        """Identify BGM using the ACRCloud fingerprinting API."""
        import httpx

        http_method = "POST"
        http_uri = "/v1/identify"
        data_type = "audio"
        signature_version = "1"
        timestamp = str(int(time.time()))

        string_to_sign = (
            http_method + "\n"
            + http_uri + "\n"
            + key + "\n"
            + data_type + "\n"
            + signature_version + "\n"
            + timestamp
        )
        sign = hashlib.sha1(
            string_to_sign.encode()
        ).digest()
        signature = base64.b64encode(sign).decode()

        with open(wav_path, "rb") as fh:
            audio_bytes = fh.read()

        files = {
            "sample": (os.path.basename(wav_path), audio_bytes, "audio/wav"),
        }
        data = {
            "access_key": key,
            "sample_bytes": str(len(audio_bytes)),
            "timestamp": timestamp,
            "signature": signature,
            "data_type": data_type,
            "signature_version": signature_version,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://identify-eu-west-1.acrcloud.com/v1/identify",
                data=data,
                files=files,
            )
            response.raise_for_status()
            result = response.json()

        if result.get("status", {}).get("msg") == "Success":
            music_list = result.get("metadata", {}).get("music", [])
            if music_list:
                music = music_list[0]
                style_tags: list[str] = []
                for genre in music.get("genres", []):
                    if "name" in genre:
                        style_tags.append(genre["name"])
                # Extract first artist name
                artists = music.get("artists", [])
                artist_name = artists[0].get("name") if artists else None
                return {
                    "title": music.get("title"),
                    "artist": artist_name,
                    "style_tags": style_tags,
                }

        return {"title": None, "artist": None, "style_tags": []}

    @staticmethod
    async def _identify_audd(wav_path: str) -> dict:
        """Identify BGM using the AudD API (fallback)."""
        import httpx

        with open(wav_path, "rb") as fh:
            audio_bytes = fh.read()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.audd.io/",
                data={
                    "api_token": "test",
                    "return": "spotify,apple_music",
                },
                files={"file": audio_bytes},
            )
            response.raise_for_status()
            result = response.json()

        if result.get("status") == "success" and result.get("result"):
            r = result["result"]
            style_tags: list[str] = []
            # Try to extract genres from Spotify metadata
            spotify_data = r.get("spotify") or {}
            album = spotify_data.get("album") or {}
            if isinstance(album, dict) and album.get("genres"):
                style_tags = album["genres"]
            return {
                "title": r.get("title"),
                "artist": r.get("artist"),
                "style_tags": style_tags,
            }

        return {"title": None, "artist": None, "style_tags": []}

    # ------------------------------------------------------------------
    # BPM detection + BGM emotion (Step 5)
    # ------------------------------------------------------------------

    def _analyze_bgm_features(self, wav_path: str) -> dict:
        """Detect BPM and classify BGM emotion using librosa spectral features.

        Returns ``{"bpm": int|None, "emotion": str|None}``.
        """
        if not _LIBROSA_AVAILABLE:
            logger.warning(
                "librosa 未安装，跳过 BPM/情感分析。"
                " 请执行: pip install librosa"
            )
            return {"bpm": None, "emotion": None}

        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(wav_path, sr=None, duration=30.0)

            # --- BPM detection ---
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = int(round(float(tempo))) if tempo > 0 else None

            # --- Spectral features ---
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            rms = float(np.mean(librosa.feature.rms(y=y)))

            # --- Heuristic emotion classification ---
            if tempo > 120 and spectral_centroid > 2000 and rms > 0.1:
                emotion = "轻快电子"
            elif tempo < 80 and spectral_bandwidth < 1500 and rms < 0.1:
                emotion = "舒缓钢琴"
            elif rms > 0.2 and spectral_bandwidth > 2000:
                emotion = "史诗管弦"
            else:
                emotion = "未知"

            return {"bpm": bpm, "emotion": emotion}
        except Exception as exc:
            logger.warning("BGM feature analysis failed: %s", exc)
            return {"bpm": None, "emotion": None}

    # ------------------------------------------------------------------
    # Speech emotion classification (Step 6)
    # ------------------------------------------------------------------

    def _classify_speech_emotion(self, wav_path: str) -> str:
        """Classify speech emotion using librosa spectral features.

        Returns one of: ``"平静"`` / ``"激昂"`` / ``"悲伤"`` / ``"幽默"`` / ``"中性"``.
        """
        if not _LIBROSA_AVAILABLE:
            logger.warning(
                "librosa 未安装，跳过语音情感分类。"
                " 请执行: pip install librosa"
            )
            return ""

        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(wav_path, sr=None, duration=30.0)

            # --- MFCC features ---
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_std = np.std(mfcc, axis=1)

            # Pitch variance proxy (MFCC coeff 0 standard deviation)
            pitch_variance = float(mfcc_std[0])

            # --- Spectral centroid ---
            spectral_centroid = float(np.mean(
                librosa.feature.spectral_centroid(y=y, sr=sr)
            ))

            # --- Zero-crossing rate ---
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

            # --- RMS energy ---
            rms = float(np.mean(librosa.feature.rms(y=y)))

            # --- Heuristic emotion classifier ---
            if spectral_centroid > 2000 and pitch_variance > 30 and zcr > 0.1:
                emotion = "激昂"
            elif zcr > 0.15:
                emotion = "悲伤"
            elif pitch_variance < 20 and spectral_centroid < 1500 and zcr < 0.08:
                emotion = "平静"
            elif pitch_variance > 25 and zcr > 0.1:
                emotion = "幽默"
            else:
                emotion = "中性"

            return emotion
        except Exception as exc:
            logger.warning("Speech emotion classification failed: %s", exc)
            return ""

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
