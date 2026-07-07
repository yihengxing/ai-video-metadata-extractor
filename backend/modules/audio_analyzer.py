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

        # --- Load audio once for all librosa-dependent analysis -------
        y_audio = None
        sr_audio = None
        if _LIBROSA_AVAILABLE:
            try:
                import librosa
                y_audio, sr_audio = librosa.load(
                    wav_path, sr=None, duration=120.0
                )
            except Exception as exc:
                logger.warning("Failed to load audio once: %s", exc)

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
                self._analyze_bgm_features, wav_path, y_audio, sr_audio
            )

            # --- Step 6: speech emotion classification ---------------
            if progress_cb:
                progress_cb("audio", 95.0, "分析语音情感...")
            speech_emotion = await asyncio.to_thread(
                self._classify_speech_emotion, wav_path, y_audio, sr_audio
            )

            # --- Step 7: sound event detection (Task 14) --------------
            if progress_cb:
                progress_cb("audio", 96.0, "检测音效事件...")
            sound_events = await asyncio.to_thread(
                self._detect_sound_events, wav_path, y_audio, sr_audio
            )

            # --- Step 8: loudness / voice-to-background ratio ---------
            if progress_cb:
                progress_cb("audio", 98.0, "分析响度与人声比例...")
            voice_to_bg_ratio = await asyncio.to_thread(
                self._analyze_loudness, wav_path, text_segments, y_audio, sr_audio
            )

            # --- Step 9: audio structure segmentation -----------------
            audio_structure = await asyncio.to_thread(
                self._segment_structure, wav_path,
                tech_meta.duration if tech_meta.duration else 0.0,
                y_audio, sr_audio,
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
                # Task 14: Sound effects + loudness + structure
                "sound_events": sound_events,
                "voice_to_bg_ratio": voice_to_bg_ratio,
                "audio_structure": audio_structure,
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

    def _analyze_bgm_features(self, wav_path: str, y=None, sr=None) -> dict:
        """Detect BPM and classify BGM emotion using librosa spectral features.

        Parameters
        ----------
        wav_path : str
            Path to the WAV file (used as fallback when *y*/*sr* are None).
        y : np.ndarray or None
            Pre-loaded audio time-series (avoids a redundant load).
        sr : int or None
            Sample rate corresponding to *y*.

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

            if y is None or sr is None:
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

    def _classify_speech_emotion(self, wav_path: str, y=None, sr=None) -> str:
        """Classify speech emotion using librosa spectral features.

        Parameters
        ----------
        wav_path : str
            Path to the WAV file (used as fallback when *y*/*sr* are None).
        y : np.ndarray or None
            Pre-loaded audio time-series (avoids a redundant load).
        sr : int or None
            Sample rate corresponding to *y*.

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

            if y is None or sr is None:
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
    # Sound event detection (Step 7 — Task 14)
    # ------------------------------------------------------------------

    def _detect_sound_events(self, wav_path: str, y=None, sr=None) -> list[str]:
        """Detect and classify audio events using librosa onset detection.

        Uses spectral features to classify each detected onset into:
          - "转场音效" (transition SFX): short burst + high frequency
          - "UI点击音" (UI click): repeated pattern
          - "爆炸/环境音" (explosion/ambient): low-frequency rumble

        Parameters
        ----------
        wav_path : str
            Path to the WAV file (used as fallback when *y*/*sr* are None).
        y : np.ndarray or None
            Pre-loaded audio time-series (avoids a redundant load).
        sr : int or None
            Sample rate corresponding to *y*.

        Returns a list of detected event type strings (may be empty).
        If librosa is unavailable, returns an empty list.
        """
        if not _LIBROSA_AVAILABLE:
            logger.warning("librosa 未安装，跳过音效检测。")
            return []

        try:
            import librosa
            import numpy as np

            if y is None or sr is None:
                y, sr = librosa.load(wav_path, sr=None, duration=30.0)

            # Detect onsets (event boundaries)
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, units="frames",
                backtrack=True,
            )

            if len(onset_frames) == 0:
                return []

            # Convert onset frames to time
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            # Compute spectral features for classification
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = float(np.mean(spectral_centroids))

            # Classify each onset
            events: list[str] = []
            # Track inter-onset intervals for repeated-pattern detection
            intervals = np.diff(onset_times)

            for i, onset_time in enumerate(onset_times):
                # Determine the onset type by analyzing the window around the onset
                frame = onset_frames[i]
                window = min(2048, len(y) - frame)

                if window <= 0:
                    continue

                segment = y[frame:frame + window]

                # Compute local RMS energy
                local_rms = float(np.sqrt(np.mean(segment ** 2)))

                # Compute local spectral centroid
                if len(segment) >= 2048:
                    stft = np.abs(librosa.stft(segment, n_fft=2048))
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                    local_centroid = float(np.average(freqs, weights=np.mean(stft, axis=1)))
                else:
                    local_centroid = spectral_centroid_mean

                # Check for repeated pattern (uniform intervals)
                is_repeated = False
                if i < len(intervals):
                    # If this interval is similar to the next one (within 20ms tolerance)
                    interval = intervals[i]
                    if interval < 1.0:  # short interval = possibly repeated
                        is_repeated = True
                if i > 0:
                    interval = intervals[i - 1]
                    if interval < 1.0:
                        is_repeated = True

                # Classify based on characteristics
                if is_repeated and local_centroid > 1500:
                    events.append("UI点击音")
                elif local_rms > 0.15 and local_centroid > 2500:
                    events.append("转场音效")
                elif local_centroid < 800 and local_rms > 0.1:
                    events.append("爆炸/环境音")

            return events
        except Exception as exc:
            logger.warning("Sound event detection failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Voice-to-background ratio / loudness (Step 8 — Task 14)
    # ------------------------------------------------------------------

    def _analyze_loudness(
        self, wav_path: str, text_segments: list[dict],
        y=None, sr=None,
    ) -> Optional[str]:
        """Analyze loudness and voice-to-background ratio using librosa.

        Computes RMS energy for speech segments (from *text_segments* timestamps)
        vs. non-speech segments, then returns a descriptive ratio string with
        an approximate LUFS estimate when possible.

        Parameters
        ----------
        wav_path : str
            Path to the WAV file (used as fallback when *y*/*sr* are None).
        text_segments : list[dict]
            Speech segments with ``{"start": float, "end": float}``.
        y : np.ndarray or None
            Pre-loaded audio time-series (avoids a redundant load).
        sr : int or None
            Sample rate corresponding to *y*.

        Returns a string like ``"人声占主导 / -14 LUFS"`` or ``None`` if
        librosa is unavailable.
        """
        if not _LIBROSA_AVAILABLE:
            logger.warning("librosa 未安装，跳过响度分析。")
            return None

        try:
            import librosa
            import numpy as np

            if y is None or sr is None:
                y, sr = librosa.load(wav_path, sr=None, duration=60.0)
            total_samples = len(y)
            total_duration = total_samples / sr

            if total_duration <= 0 or total_samples == 0:
                return None

            # --- Mark speech frames from text_segments ---
            speech_mask = np.zeros(total_samples, dtype=bool)
            for seg in text_segments:
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                start_sample = max(0, start_sample)
                end_sample = min(total_samples, end_sample)
                if end_sample > start_sample:
                    speech_mask[start_sample:end_sample] = True

            # --- Compute RMS for speech and non-speech ---
            speech_rms: Optional[float] = None
            nonspeech_rms: Optional[float] = None

            if np.any(speech_mask):
                speech_rms = float(
                    np.sqrt(np.mean(y[speech_mask] ** 2))
                )
            if np.any(~speech_mask):
                nonspeech_rms = float(
                    np.sqrt(np.mean(y[~speech_mask] ** 2))
                )

            # --- Approximate LUFS from RMS (rough conversion) ---
            # dB FS = 20 * log10(rms / full_scale)
            # For 16-bit audio, full_scale = 32768 → normalized RMS = rms (already float [-1,1])
            # LUFS ≈ dB FS integrated, roughly: dB FS - 0.691 for speech-like signals
            def _rms_to_approx_lufs(rms_val: float) -> float:
                if rms_val <= 0:
                    return -70.0  # essentially silent
                db_fs = 20.0 * np.log10(rms_val)
                # Approximate integrated LUFS (speech-type signal correction)
                return float(db_fs - 0.691)

            # --- Build the result string ---
            if speech_rms is not None and nonspeech_rms is not None and nonspeech_rms > 0:
                ratio = speech_rms / nonspeech_rms
                if ratio > 3.0:
                    desc = "人声占主导"
                elif ratio > 1.5:
                    desc = "人声略高于背景"
                elif ratio > 0.5:
                    desc = "人声与背景均衡"
                else:
                    desc = "背景音占主导"

                # Use the overall RMS for the LUFS estimate
                overall_rms = float(np.sqrt(np.mean(y ** 2)))
                approx_lufs = _rms_to_approx_lufs(overall_rms)
                lufs_str = f"{approx_lufs:.0f} LUFS"

                return f"{desc} / {lufs_str}"
            elif speech_rms is not None:
                # Only speech — no significant background
                approx_lufs = _rms_to_approx_lufs(speech_rms)
                return f"人声占主导 / {approx_lufs:.0f} LUFS"
            elif nonspeech_rms is not None:
                # No speech at all
                approx_lufs = _rms_to_approx_lufs(nonspeech_rms)
                return f"纯背景音 / {approx_lufs:.0f} LUFS"
            else:
                return None
        except Exception as exc:
            logger.warning("Loudness analysis failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Audio structure segmentation (Step 9 — Task 14)
    # ------------------------------------------------------------------

    def _segment_structure(
        self, wav_path: str, duration: float,
        y=None, sr=None,
    ) -> Optional[str]:
        """Segment audio into structural sections using spectral flux.

        Detects major changes in spectral features to divide audio into:
        intro, main sections, transitions, and outro.

        Parameters
        ----------
        wav_path : str
            Path to the WAV file (used as fallback when *y*/*sr* are None).
        duration : float
            Total duration of the audio in seconds.
        y : np.ndarray or None
            Pre-loaded audio time-series (avoids a redundant load).
        sr : int or None
            Sample rate corresponding to *y*.

        Returns a string like ``"前奏(0-3s) → 主段1(3-12s) → ..."``.
        For audio shorter than 5 seconds, a simple description is returned.
        If librosa is unavailable, returns ``None``.
        """
        if not _LIBROSA_AVAILABLE:
            logger.warning("librosa 未安装，跳过音频结构分割。")
            return None

        try:
            import librosa
            import numpy as np

            # Short audio — trivial structure
            if duration < 5.0:
                return f"简短视频音频 ({duration:.1f}s)"

            if y is None or sr is None:
                y, sr = librosa.load(wav_path, sr=None, duration=min(duration, 120.0))

            # --- Compute spectral flux (frame-to-frame change) ---
            # Use mel spectrogram for better perceptual relevance
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

            # Spectral flux: squared difference between consecutive frames
            spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
            flux_mean = float(np.mean(spectral_flux))
            flux_std = float(np.std(spectral_flux))

            # --- Find peaks in spectral flux (major changes) ---
            # A change point is a frame where flux exceeds mean + 1.5 * std
            threshold = flux_mean + 1.5 * flux_std

            if threshold <= 0 or flux_std <= 0:
                # Uniform audio — single segment
                if duration < 10.0:
                    return f"单段音频 ({duration:.1f}s)"
                else:
                    return f"连续音频 ({duration:.1f}s)"

            change_frames = np.where(spectral_flux > threshold)[0]

            if len(change_frames) == 0:
                if duration < 10.0:
                    return f"单段音频 ({duration:.1f}s)"
                else:
                    return f"连续音频 ({duration:.1f}s)"

            # Convert frame indices to times
            hop_length = 512  # librosa default for melspectrogram
            change_times = librosa.frames_to_time(
                change_frames, sr=sr, hop_length=hop_length
            )

            # Merge nearby changes (within 1 second)
            merged_times: list[float] = []
            for t in change_times:
                if not merged_times or t - merged_times[-1] > 1.0:
                    merged_times.append(float(t))
                else:
                    # Merge: keep the average
                    merged_times[-1] = (merged_times[-1] + float(t)) / 2.0

            # --- Build section labels ---
            if len(merged_times) == 0:
                return f"单段音频 ({duration:.1f}s)"

            sections: list[str] = []
            boundaries = [0.0] + merged_times + [duration]

            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                seg_duration = end - start

                if seg_duration < 0.3:
                    continue  # skip very short segments

                if i == 0:
                    label = f"前奏({start:.0f}-{end:.0f}s)"
                elif i == len(boundaries) - 2:
                    label = f"结尾({start:.0f}-{end:.0f}s)"
                else:
                    # Alternate between main sections and transitions
                    if i % 2 == 0:
                        seg_idx = i // 2
                        label = f"主段{seg_idx}({start:.0f}-{end:.0f}s)"
                    else:
                        seg_idx = (i + 1) // 2
                        label = f"过渡{seg_idx}({start:.0f}-{end:.0f}s)"

                sections.append(label)

            if not sections:
                return f"单段音频 ({duration:.1f}s)"

            return " → ".join(sections)
        except Exception as exc:
            logger.warning("Audio structure segmentation failed: %s", exc)
            return None

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
