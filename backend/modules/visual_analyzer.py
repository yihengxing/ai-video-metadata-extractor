"""Visual analyzer — scene detection & keyframe extraction (Task 9).

Integrates PySceneDetect for shot boundary detection and OpenCV for frame
extraction.  Tasks 10-11 will extend this module with CLIP/YOLO and
motion/text detection.
"""
from __future__ import annotations
import os
import hashlib
from typing import Optional

from backend.modules.base import Extractor, ProgressCallback
from backend.models.schemas import TechMetadata
from backend.utils.keyframe_utils import rank_representative_frames

# ---------------------------------------------------------------------------
# Dependency checks (import-time validation so we fail early)
# ---------------------------------------------------------------------------
try:
    import cv2  # noqa: F401
except ImportError:
    raise RuntimeError(
        "OpenCV (cv2) 未安装。请执行: pip install opencv-python"
    )

try:
    from scenedetect import detect, ContentDetector  # noqa: F401
except ImportError:
    raise RuntimeError(
        "PySceneDetect 未安装。请执行: pip install scenedetect"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_KEYFRAME_GRID_COLS = 4
_KEYFRAME_GRID_ROWS = 4
_KEYFRAME_GRID_COUNT = _KEYFRAME_GRID_COLS * _KEYFRAME_GRID_ROWS  # 16
_REPRESENTATIVE_TOP_N = 5
_THUMB_BASE = os.path.expanduser("~/.ai-video-analyzer/thumbnails")


class VisualAnalyzer(Extractor):
    """Scene detection and keyframe extraction using PySceneDetect + OpenCV."""

    @property
    def module_name(self) -> str:
        return "visual"

    # ------------------------------------------------------------------
    async def extract(
        self,
        file_path: str,
        tech_meta: TechMetadata,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> dict:
        """Run full visual extraction pipeline.

        Pipeline:
        1. PySceneDetect shot detection
        2. Per-shot thumbnail extraction
        3. Uniform keyframe grid across whole video
        4. Rank representative frames (top 5)
        """
        # -- 0 % ---------------------------------------------------------
        _report(progress_cb, 0.0, "开始场景检测...")

        # Prepare thumbnail directory
        file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        thumb_dir = os.path.join(_THUMB_BASE, file_hash)
        os.makedirs(thumb_dir, exist_ok=True)

        video_duration = tech_meta.duration

        # -- 30 % : scene detection --------------------------------------
        _report(progress_cb, 5.0, "运行 PySceneDetect 镜头检测...")
        scene_list = _detect_scenes(file_path)
        _report(progress_cb, 30.0, f"检测到 {len(scene_list)} 个镜头")

        # -- 60 % : extract per-shot thumbnails --------------------------
        shot_count = len(scene_list)
        shots: list[dict] = []
        all_thumbnail_paths: list[str] = []

        for idx, (start_tc, end_tc) in enumerate(scene_list):
            start_sec = start_tc.seconds
            end_sec = end_tc.seconds
            duration = end_sec - start_sec

            thumb_path = os.path.join(thumb_dir, f"shot_{idx:04d}.jpg")
            _extract_frame_at_time(file_path, start_sec, thumb_path)

            shots.append({
                "index": idx,
                "start_time": start_sec,
                "end_time": end_sec,
                "duration": duration,
                "thumbnail_path": thumb_path,
                "is_representative": False,
            })
            all_thumbnail_paths.append(thumb_path)

            # Report per-shot progress within the 30-60 % band
            if shot_count > 0:
                sub_pct = 30.0 + 30.0 * ((idx + 1) / shot_count)
                _report(progress_cb, sub_pct, f"提取镜头 {idx + 1}/{shot_count} 缩略图...")

        # -- 80 % : keyframe grid (uniform samples across whole video) ---
        _report(progress_cb, 65.0, "生成关键帧网格...")
        grid_paths = _build_keyframe_grid(file_path, video_duration, thumb_dir)

        # -- 90 % : rank representative frames ---------------------------
        _report(progress_cb, 85.0, "排序代表性帧...")
        representative = rank_representative_frames(
            all_thumbnail_paths, top_n=_REPRESENTATIVE_TOP_N
        )

        # Mark representative shots
        rep_set = set(representative)
        for shot in shots:
            if shot["thumbnail_path"] in rep_set:
                shot["is_representative"] = True

        # -- Compute derived values --------------------------------------
        avg_shot_dur = (video_duration / shot_count) if shot_count > 0 else 0.0
        # For now, all transitions are hard cuts (task 11 will classify)
        transitions = ["硬切"] * max(0, shot_count - 1)

        # -- 100 % -------------------------------------------------------
        _report(progress_cb, 100.0, "视觉分析完成")

        return {
            "shots": shots,
            "shot_count": shot_count,
            "avg_shot_duration": round(avg_shot_dur, 2),
            "transitions": transitions,
            "keyframe_grid_paths": grid_paths,
            "representative_frames": representative,
            "color_summary": None,          # Task 10
            "text_regions": [],             # Task 11
            "face_detections": [],          # Task 10
            "object_detections": [],        # Task 10
            "motion_summary": None,         # Task 11
        }


# ====================================================================
# Internal helpers
# ====================================================================

def _report(cb: Optional[ProgressCallback], pct: float, msg: str) -> None:
    """Fire progress callback if provided."""
    if cb:
        cb("visual", pct, msg)


def _detect_scenes(video_path: str) -> list:
    """Run PySceneDetect ContentDetector and return scene list."""
    return detect(video_path, ContentDetector())


def _extract_frame_at_time(
    video_path: str, time_sec: float, out_path: str
) -> None:
    """Extract a single frame at *time_sec* and save as JPEG."""
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        ret, frame = cap.read()
        if not ret:
            # Fallback: seek to beginning and read first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    finally:
        cap.release()


def _build_keyframe_grid(
    video_path: str, duration: float, thumb_dir: str
) -> list[str]:
    """Sample frames uniformly across the whole video for the keyframe grid.

    Returns a list of saved thumbnail paths.
    """
    if duration <= 0:
        return []

    grid_count = min(_KEYFRAME_GRID_COUNT, max(4, int(duration * 4)))
    paths: list[str] = []

    cap = cv2.VideoCapture(video_path)
    try:
        for i in range(grid_count):
            t = duration * (i + 0.5) / grid_count  # sample at midpoints
            out_path = os.path.join(thumb_dir, f"grid_{i:04d}.jpg")
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                paths.append(out_path)
    finally:
        cap.release()

    return paths
