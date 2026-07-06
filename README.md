# AI Video Metadata Extractor

AI短视频元数据逆向提取工具 -- reverse-engineer the generation parameters of
AI-generated short videos by extracting technical metadata, visual features,
audio characteristics, and performing source recovery via reverse image search.

## Architecture

```
ai-video-metadata-extractor/
├── backend/              # Python FastAPI backend
│   ├── main.py           # FastAPI application entry point
│   ├── orchestrator.py   # Multi-module pipeline orchestrator
│   ├── config.py         # Runtime configuration (API keys, settings)
│   ├── modules/          # Analysis modules (DAG)
│   │   ├── base.py           # Abstract Extractor / Matcher interfaces
│   │   ├── tech_extractor.py # FFprobe-based technical metadata
│   │   ├── visual_analyzer.py# Scene detection, CLIP, YOLO, motion
│   │   ├── audio_analyzer.py # Whisper STT, BGM recognition, loudness
│   │   ├── ai_inferrer.py    # LLM-based tool/prompt inference
│   │   └── source_recovery.py# SauceNAO + Civitai source matching
│   ├── models/schemas.py # Pydantic data models
│   ├── services/         # Shared services
│   │   ├── cache_service.py  # SHA-256 file-hash JSON cache
│   │   ├── export_service.py # JSON, Markdown, SRT, ComfyUI export
│   │   └── ffmpeg_service.py # FFmpeg/FFprobe probe & parse
│   └── utils/            # File I/O, keyframe preprocessing
├── src/                  # React frontend (TypeScript)
│   ├── components/       # UI panels: Results, Compare, Export
│   ├── store/            # Zustand state management
│   ├── services/         # API client, WebSocket hooks
│   └── types/            # TypeScript type definitions
├── electron/             # Electron main process (Python manager)
├── tests/                # pytest backend tests
├── electron-builder.yml  # Electron packaging config
└── package.json          # Frontend / Electron dependencies
```

## Analysis Pipeline

```
Validate + Hash -> Tech -> (Visual || Audio) -> (AI || Source Recovery) -> Aggregate -> Cache
```

### Modules

| Module | Description |
|--------|-------------|
| **Tech** | Container format, codecs, resolution, framerate, bitrates, GOP, color space, HDR, platform fingerprint |
| **Visual** | Shot detection (PySceneDetect), keyframe grid, CLIP style classification, YOLO object detection, face detection, color histogram, motion analysis (optical flow), text region detection |
| **Audio** | Speech-to-text (Whisper), speech emotion, BGM fingerprint recognition, BPM detection, sound event detection, loudness analysis, audio structure segmentation |
| **AI Inference** | LLM-based AI tool identification, prompt reversal, workflow inference, model recommendations |
| **Source Recovery** | SauceNAO reverse image search, Civitai API metadata/ComfyUI workflow recovery, comfyworkflows.com scraper |

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- FFmpeg / FFprobe (in PATH or in `./ffmpeg/`)

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Required Python packages:
- fastapi, uvicorn, httpx
- pydantic
- cryptography (config encryption)
- opencv-python
- scenedetect (PySceneDetect)
- faster-whisper
- librosa, soundfile
- torch, open-clip-torch (optional -- CLIP classification)
- ultralytics (optional -- YOLO object detection)

### Frontend

```bash
npm install
```

### Configuration

Sensitive API keys are encrypted on disk using Fernet (symmetric encryption).
Keys are managed via the Electron settings panel or REST API:

```json
{
  "llm_api_key": "",
  "llm_provider": "claude",
  "saucenao_api_key": "",
  "acrcloud_key": "",
  "acrcloud_secret": "",
  "source_recovery_consent": false
}
```

## Running

### Development (backend + frontend separately)

```bash
# Terminal 1: Backend
cd backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
npm run dev
```

### Packaged (Electron)

```bash
npm run build && npm run electron:build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Start video analysis |
| `GET` | `/analyze/{hash}/status` | Query analysis status |
| `GET` | `/cache` | List cached analyses |
| `GET` | `/cache/{hash}` | Get cached result |
| `DELETE` | `/cache/{hash}` | Delete cached result |
| `GET` | `/export/{hash}` | Export result (format: json, markdown, srt, comfyui_workflow, comfyui_prompt) |
| `WS` | `/ws/{hash}` | WebSocket progress stream |

## Testing

```bash
cd backend
python -m pytest ../tests/ -v
```

## Export Formats

- **JSON** -- Full analysis result as pretty-printed JSON
- **Markdown** -- Chinese-language analysis report
- **SRT** -- Subtitles from speech-to-text segments
- **ComfyUI Workflow** -- Recovered original workflow JSON
- **ComfyUI Prompt** -- Prompt text formatted for ComfyUI consumption
