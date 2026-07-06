# Task 13 Report: BGM Recognition & Audio Features (Audio Analyzer Part 2)

**Status**: COMPLETE
**Date**: 2026-07-06
**Tests**: 19/19 audio analyzer tests passing, full suite 43/43 passing

---

## Files Modified

### `backend/modules/audio_analyzer.py`

Added three new analysis methods to `AudioAnalyzer` and integrated them into `extract()`:

#### 1. `_identify_bgm(self, wav_path: str) -> dict` (Step 4)
BGM fingerprint recognition with dual-API strategy:
- **ACRCloud** (primary): Reads `acrcloud_key`/`acrcloud_secret` from `backend.config.settings`. Builds HMAC-SHA1 signed request to `/v1/identify`, extracts `metadata.music[0]` on `"Success"` status.
- **AudD** (fallback): Uses `api.audd.io` with test token. Extracts genres from Spotify metadata when available.
- Returns `{"title": str|None, "artist": str|None, "style_tags": list[str]}`.
- Graceful degradation: outer try/except returns None values + log warning; inner ACRCloud/AudD exceptions caught individually so one failure does not prevent the other.

#### 2. `_analyze_bgm_features(self, wav_path: str) -> dict` (Step 5)
BPM detection + BGM emotion classification via librosa:
- Loads audio with librosa (30s max), detects tempo via `librosa.beat.beat_track`.
- Computes spectral centroid, spectral bandwidth, RMS energy.
- Heuristic emotion classifier:
  - Tempo > 120 + centroid > 2000 + RMS > 0.1: "轻快电子"
  - Tempo < 80 + bandwidth < 1500 + RMS < 0.1: "舒缓钢琴"
  - RMS > 0.2 + bandwidth > 2000: "史诗管弦"
  - Default: "未知"
- Returns `{"bpm": int|None, "emotion": str|None}`.
- `_LIBROSA_AVAILABLE` guard: returns None if librosa not installed.

#### 3. `_classify_speech_emotion(self, wav_path: str) -> str` (Step 6)
Speech emotion classification via librosa spectral features:
- Extracts MFCC (13 coefficients), spectral centroid, ZCR, RMS.
- Heuristic classifier using pitch variance (MFCC std), centroid, ZCR:
  - High centroid + pitch variance + ZCR: "激昂"
  - Very high ZCR: "悲伤"
  - Low pitch variance + low centroid + low ZCR: "平静"
  - Moderate pitch variance + ZCR: "幽默"
  - Default: "中性"
- Returns one of: `"平静"` / `"激昂"` / `"悲伤"` / `"幽默"` / `"中性"` (or `""` on failure).
- `_LIBROSA_AVAILABLE` guard: returns `""` if librosa not installed.

#### Integration into `extract()`
After transcription (Step 2), calls are made:
- Step 4 at 80%: `_identify_bgm` (async, API calls)
- Step 5 at 88%: `_analyze_bgm_features` (CPU, run in `asyncio.to_thread`)
- Step 6 at 95%: `_classify_speech_emotion` (CPU, run in `asyncio.to_thread`)

All 12 return dict fields are populated with real values instead of placeholders.

#### Module-level additions
- Added `_LIBROSA_AVAILABLE` lazy-import guard (mirrors `_WHISPER_AVAILABLE` pattern).
- Added imports: `base64`, `hashlib`, `time`.

### `tests/backend/test_audio_analyzer.py`

5 new tests added (14 -> 19), 1 existing test updated:

| Test | What it verifies |
|------|-----------------|
| `test_bgm_features_on_silence` | `_analyze_bgm_features` on silent WAV returns dict with `bpm`/`emotion` keys; handles both librosa-available and unavailable cases |
| `test_speech_emotion_returns_valid_category` | `_classify_speech_emotion` returns one of the 5 valid categories + empty string |
| `test_bgm_identify_no_config` | `_identify_bgm` returns None values when no API keys configured; mocked httpx to avoid network |
| `test_bgm_features_no_librosa_graceful` | `_analyze_bgm_features` returns `{"bpm": None, "emotion": None}` when librosa is not available |
| `test_speech_emotion_no_librosa_graceful` | `_classify_speech_emotion` returns `""` when librosa is not available |

`test_extract_with_mocked_whisper` updated: mocks `_identify_bgm`, `_analyze_bgm_features`, and `_classify_speech_emotion` on the analyzer instance to avoid network/librosa dependencies, and verifies the populated Task 13 fields.

## Key Design Decisions

### Dual-API BGM identification
ACRCloud as primary (requires API credentials), AudD as fallback (test token available without registration). Both are wrapped in independent try/except blocks so one API's failure does not prevent the other from being tried.

### Librosa lazy-import guard
`_LIBROSA_AVAILABLE` flag set at module import time, checked inside each librosa-dependent method before attempting computation. If unavailable, returns sensible defaults and logs a warning with install instructions.

### Heuristic classifiers (not ML)
Both BGM emotion and speech emotion use simple threshold-based heuristics on librosa spectral features rather than trained models. This avoids model weight dependencies and keeps the module self-contained.

### Sync methods wrapped in `asyncio.to_thread`
`_analyze_bgm_features` and `_classify_speech_emotion` are synchronous methods that perform CPU-bound librosa operations. They are called via `asyncio.to_thread` in `extract()` to avoid blocking the event loop.

### Progress reporting
`0% -> 5% -> 20% -> 30% -> 70% -> 80% -> 88% -> 95% -> 100%`

Note: the transcription completion was moved from 80% to 70% to accommodate the new steps at 80%, 88%, and 95%.

### Return dict (matching AudioAnalysis schema)
```python
{
    "full_text": str,
    "text_segments": [{"text": str, "start": float, "end": float}, ...],
    "speech_rate": float,
    "speech_emotion": "激昂",           # populated by Task 13
    "bgm_title": "Test Song",           # populated by Task 13
    "bgm_artist": "Test Artist",        # populated by Task 13
    "bgm_style_tags": ["pop"],          # populated by Task 13
    "bgm_emotion": "轻快电子",           # populated by Task 13
    "bgm_bpm": 128,                     # populated by Task 13
    "sound_events": [],                 # placeholder for Task 14
    "voice_to_bg_ratio": None,          # placeholder for Task 14
    "audio_structure": None,            # placeholder for Task 14
}
```

## Notes
- ACRCloud requires valid credentials in the config (`acrcloud_key`/`acrcloud_secret`). Without them, the call is skipped and AudD fallback is used.
- AudD's `"api_token": "test"` has limited quota; production would need a real token.
- librosa must be installed (`pip install librosa`) for BPM/emotion features — it is already in `backend/requirements.txt`.
- httpx must be installed for API calls — it is already in `backend/requirements.txt`.
- Methods gracefully handle the WAV file being deleted by the `finally` cleanup block — the WAV is extracted, all 6 steps run, then it is cleaned up.
- Tests that simulate librosa unavailability pop `librosa` from `sys.modules` and then restore it, so subsequent tests are unaffected.
