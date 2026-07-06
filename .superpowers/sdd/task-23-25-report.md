# Tasks 23-25 Report: Electron Shell (Main Process + Preload + Python Manager + Vite Config)

**Date:** 2026-07-06
**Status:** Implemented

## Files Created

### Electron Main Process
- **`electron/main.ts`** — Electron main process entry point
  - Creates `BrowserWindow` (1400x900, min 1100x700, dark theme #141414)
  - Loads Vite dev server in development (`VITE_DEV_SERVER_URL` env var) or built files in production
  - Application menu with File (Open Video, Compare, Settings), Edit, View, Help
  - IPC handlers: `get-backend-url`, `dialog:openFile`
  - Launches `PythonManager` on app ready, stops on quit

### Preload Script
- **`electron/preload.ts`** — Context bridge exposing safe API to renderer
  - `getBackendUrl()` — fetches the dynamic backend port from main process
  - `onMenuCompare`, `onMenuSettings`, `onMenuOpenFile` — menu event listeners
  - `openFileDialog()` — native file open dialog for video files
  - Cleanup listeners for each menu event

### Python Manager
- **`electron/python-manager.ts`** — Python subprocess lifecycle management
  - **Python discovery:** checks `python-embedded/python.exe` (embedded), `python-embedded/bin/python3` (macOS/Linux), falls back to `python3`
  - **Spawn:** `python -m uvicorn backend.main:app --host 127.0.0.1 --port 0` with `PYTHONUNBUFFERED=1`
  - **Port detection:** parses uvicorn's "Uvicorn running on http://127.0.0.1:{port}" from stdout/stderr with 30s timeout
  - **Health check:** polls `GET /health` every 500ms after a 1s initial delay, with 30s total timeout
  - **Graceful shutdown:** Windows uses `taskkill /f /t`, Unix uses `SIGTERM` with 5s fallback to `SIGKILL`

### Vite Configuration
- **`vite.config.ts`** — Vite config for React + Electron
  - React plugin enabled
  - Output directory: `dist`
  - Path alias: `@` maps to `src/`
  - Dev server on port 5173

### React Entry Points
- **`index.html`** — Root HTML with `<div id="root">` and module script entry
- **`src/main.tsx`** — React root render with Ant Design dark theme ConfigProvider
- **`src/App.tsx`** — Minimal placeholder app showing title and "connecting" status

### Type Declarations
- **`src/types/electron.d.ts`** — `ElectronAPI` interface and global `Window` augmentation
  - Includes `getBackendUrl`, menu listener registration/cleanup, and `openFileDialog`

## Key Design Decisions

1. **Dynamic port allocation:** Using `--port 0` lets uvicorn choose any free port, avoiding port conflicts
2. **Port parsing from stderr:** Uvicorn prints its "running on..." message to stderr; the manager reads both stdout and stderr
3. **Health check polling:** Ensures the backend is fully initialized before Electron loads the UI
4. **Cross-platform process kill:** Windows uses `taskkill` (SIGTERM not supported), Unix uses POSIX signals
5. **Context isolation:** renderer has no direct Node.js access; all IPC goes through the preload bridge
