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

# Sentinel for caching lazy-load failures (Task 11 review fix)
_LOAD_FAILED = object()


class VisualAnalyzer(Extractor):
    """Scene detection and keyframe extraction using PySceneDetect + OpenCV.

    Task 10 adds CLIP classification, YOLO object detection, OpenCV face
    detection, and HSV color histogram analysis — all with lazy loading
    and graceful degradation.
    """

    def __init__(self) -> None:
        super().__init__()
        # Lazy-loaded ML models (Task 10)
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._yolo_model = None
        self._face_cascade = None

    @property
    def module_name(self) -> str:
        return "visual"

    # ------------------------------------------------------------------
    # Lazy-loading helpers (Task 10)
    # ------------------------------------------------------------------

    def _get_clip(self):
        """Lazy-load OpenCLIP model. Returns (model, preprocess, tokenizer).

        Load failures are cached via _LOAD_FAILED sentinel so retries don't
        hammer missing dependencies (Task 11 review fix).
        """
        if self._clip_model is None:
            try:
                import open_clip
                import torch
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self._clip_model = model
                self._clip_preprocess = preprocess
                self._clip_tokenizer = tokenizer
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "OpenCLIP 未安装，风格分类不可用: %s", e
                )
                self._clip_model = _LOAD_FAILED
                self._clip_preprocess = _LOAD_FAILED
                self._clip_tokenizer = _LOAD_FAILED
        if self._clip_model is _LOAD_FAILED:
            return None, None, None
        return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    def _get_yolo(self):
        """Lazy-load YOLOv8 model. Returns YOLO instance or None.

        Load failures are cached via _LOAD_FAILED sentinel so retries don't
        hammer missing dependencies (Task 11 review fix).
        """
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO("yolov8n.pt")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "YOLO 模型加载失败: %s", e
                )
                self._yolo_model = _LOAD_FAILED
        return self._yolo_model if self._yolo_model is not _LOAD_FAILED else None

    def _get_face_cascade(self):
        """Lazy-load OpenCV Haar cascade for face detection.

        Load failures are cached via _LOAD_FAILED sentinel so retries don't
        hammer missing resources (Task 11 review fix).
        """
        if self._face_cascade is None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "OpenCV 人脸检测级联加载失败: %s", e
                )
                self._face_cascade = _LOAD_FAILED
        return self._face_cascade if self._face_cascade is not _LOAD_FAILED else None

    # ------------------------------------------------------------------
    # Task 10 analysis methods
    # ------------------------------------------------------------------

    def _analyze_colors(self, image_paths: list[str]) -> dict:
        """Compute HSV histogram per keyframe and classify dominant hue.

        Returns a ColorSummary-compatible dict with keys:
            dominant_hue, saturation_level, description
        """
        import logging
        logger = logging.getLogger(__name__)

        if not image_paths:
            logger.warning("_analyze_colors: 无关键帧输入")
            return {
                "dominant_hue": "中性",
                "saturation_level": "unknown",
                "description": "无图像数据",
            }

        total_warm = 0.0
        total_cool = 0.0
        total_neutral = 0.0
        total_saturation = 0.0
        valid_count = 0

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0].flatten()
            s = hsv[:, :, 1].flatten()

            # Classify hue ranges: warm (0-30, 150-179), cool (75-135), neutral (rest)
            warm_mask = (h < 30) | (h > 150)
            cool_mask = (h >= 75) & (h <= 135)

            total_warm += float(warm_mask.sum())
            total_cool += float(cool_mask.sum())
            total_neutral += float((~warm_mask & ~cool_mask).sum())
            total_saturation += float(s.mean())
            valid_count += 1

        if valid_count == 0:
            return {
                "dominant_hue": "中性",
                "saturation_level": "unknown",
                "description": "无法读取图像",
            }

        avg_warm = total_warm / valid_count
        avg_cool = total_cool / valid_count
        avg_neutral = total_neutral / valid_count
        avg_sat = total_saturation / valid_count

        # Determine dominant hue
        if avg_warm > avg_cool and avg_warm > avg_neutral:
            dominant_hue = "暖色调"
        elif avg_cool > avg_warm and avg_cool > avg_neutral:
            dominant_hue = "冷色调"
        else:
            dominant_hue = "中性"

        # Classify saturation level
        if avg_sat > 128:
            saturation_level = "高饱和度"
        elif avg_sat > 64:
            saturation_level = "中等饱和度"
        else:
            saturation_level = "低饱和度"

        # Build description
        hue_desc = {"暖色调": "暖色系", "冷色调": "冷色系", "中性": "中性色系"}[dominant_hue]
        description = f"{hue_desc}·{saturation_level}"

        return {
            "dominant_hue": dominant_hue,
            "saturation_level": saturation_level,
            "description": description,
        }

    def _detect_faces(self, file_path: str, shots: list[dict]) -> list[dict]:
        """Detect faces at shot midpoints using OpenCV Haar cascade.

        Returns list of {shot_index, face_count, bbox_list}.
        """
        import logging
        logger = logging.getLogger(__name__)

        cascade = self._get_face_cascade()
        if cascade is None:
            logger.warning("_detect_faces: 级联分类器不可用，跳过人脸检测")
            return []

        cap = cv2.VideoCapture(file_path)
        results: list[dict] = []

        try:
            for shot in shots:
                mid_time = (shot["start_time"] + shot["end_time"]) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    results.append({
                        "shot_index": shot["index"],
                        "face_count": 0,
                        "bbox_list": [],
                    })
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                bbox_list = [
                    {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                    for (x, y, w, h) in faces
                ]
                results.append({
                    "shot_index": shot["index"],
                    "face_count": len(bbox_list),
                    "bbox_list": bbox_list,
                })
        finally:
            cap.release()

        return results

    def _detect_objects(self, file_path: str, shots: list[dict]) -> list[dict]:
        """Detect objects at shot midpoints using YOLOv8.

        Returns list of {class_name, confidence, shot_index, timestamp}.
        """
        import logging
        logger = logging.getLogger(__name__)

        yolo = self._get_yolo()
        if yolo is None:
            logger.warning("_detect_objects: YOLO 模型不可用，跳过物体检测")
            return []

        results: list[dict] = []
        cap = cv2.VideoCapture(file_path)

        try:
            for shot in shots:
                mid_time = (shot["start_time"] + shot["end_time"]) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    continue

                # YOLO inference
                yolo_results = yolo(frame, verbose=False)
                for r in yolo_results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        class_name = r.names.get(cls_id, str(cls_id))
                        confidence = float(box.conf[0].item())
                        results.append({
                            "class_name": class_name,
                            "confidence": round(confidence, 4),
                            "shot_index": shot["index"],
                            "timestamp": round(mid_time, 2),
                        })
        finally:
            cap.release()

        return results

    def _classify_style(self, image_paths: list[str]) -> dict:
        """Zero-shot CLIP classification of visual style tags.

        Returns {"dominant_style": str, "style_scores": dict, "color_mood": str}.
        """
        import logging
        import numpy as np
        logger = logging.getLogger(__name__)

        model, preprocess, tokenizer = self._get_clip()
        if model is None:
            logger.warning("_classify_style: CLIP 模型不可用，跳过风格分类")
            return {
                "dominant_style": "未知",
                "style_scores": {},
                "color_mood": "未知",
            }

        # Chinese style tags matching PRD v1.2 requirements
        style_categories = [
            "赛博朋克",
            "电影感",
            "复古",
            "水墨风",
            "动漫风",
            "写实",
            "高饱和度",
            "低饱和度",
            "暖色调",
            "冷色调",
            "暗色调",
            "明亮",
        ]

        # Encode text prompts
        try:
            import torch
            text_tokens = tokenizer(style_categories)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.warning("_classify_style: 文本编码失败: %s", e)
            return {
                "dominant_style": "未知",
                "style_scores": {},
                "color_mood": "未知",
            }

        # Classify each keyframe and accumulate scores
        accumulated: dict[str, float] = {cat: 0.0 for cat in style_categories}
        valid_count = 0

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            # BGR to RGB for CLIP
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                from PIL import Image
                pil_img = Image.fromarray(img_rgb)
                img_tensor = preprocess(pil_img).unsqueeze(0)
                with torch.no_grad():
                    image_features = model.encode_image(img_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T).squeeze(0)
                    probs = torch.softmax(similarity, dim=0)
                for cat, prob in zip(style_categories, probs.tolist()):
                    accumulated[cat] += prob
                valid_count += 1
            except Exception as e:
                logger.warning("_classify_style: 帧分类失败 %s: %s", path, e)
                continue

        if valid_count == 0:
            return {
                "dominant_style": "未知",
                "style_scores": {},
                "color_mood": "未知",
            }

        # Average scores across frames
        style_scores = {cat: round(v / valid_count, 4) for cat, v in accumulated.items()}
        dominant_style = max(style_scores, key=lambda k: style_scores[k])

        # Derive color mood from style scores
        warm_score = style_scores.get("暖色调", 0) + style_scores.get("高饱和度", 0) * 0.5
        cool_score = style_scores.get("冷色调", 0) + style_scores.get("低饱和度", 0) * 0.5
        dark_score = style_scores.get("暗色调", 0)
        bright_score = style_scores.get("明亮", 0)

        if dark_score > bright_score and dark_score > 0.3:
            color_mood = "暗色调"
        elif bright_score > dark_score and bright_score > 0.3:
            color_mood = "明亮"
        elif warm_score > cool_score:
            color_mood = "暖色调"
        elif cool_score > warm_score:
            color_mood = "冷色调"
        else:
            color_mood = "中性"

        return {
            "dominant_style": dominant_style,
            "style_scores": style_scores,
            "color_mood": color_mood,
        }

    # ------------------------------------------------------------------
    # Task 11 analysis methods
    # ------------------------------------------------------------------

    def _analyze_motion(self, file_path: str, shots: list[dict]) -> str:
        """Compute optical-flow motion analysis at ~1 fps across the video.

        Samples frames at roughly 1 fps.  For each consecutive pair, the
        average displacement magnitude of tracked feature points is computed
        via sparse optical flow (cv2.calcOpticalFlowPyrLK).  Segments are
        classified as:

        * 静态       — avg motion < 0.5 px
        * 轻微运动    — avg motion 0.5 – 2.0 px
        * 剧烈运动    — avg motion > 2.0 px

        Returns a summary string such as
        ``"静态为主 (60%), 轻微运动 (30%), 剧烈运动 (10%)"``.
        """
        import logging
        logger = logging.getLogger(__name__)

        _STATIC = "静态"
        _SLIGHT = "轻微运动"
        _VIGOROUS = "剧烈运动"
        _THRESHOLD_LOW = 0.5
        _THRESHOLD_HIGH = 2.0

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if duration <= 0 or fps <= 0:
            cap.release()
            logger.warning("_analyze_motion: 无法获取视频时长或帧率")
            return "未知"

        # Sample at roughly 1 fps — read every *fps* frames
        sample_interval = max(1, int(fps))
        classifications: list[str] = []

        prev_gray = None
        frame_idx = 0
        read_count = 0

        try:
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                read_count += 1

                if prev_gray is not None:
                    # Detect feature points in previous frame
                    prev_pts = cv2.goodFeaturesToTrack(
                        prev_gray,
                        maxCorners=200,
                        qualityLevel=0.01,
                        minDistance=10,
                    )
                    if prev_pts is not None:
                        # Track into current frame
                        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray, gray, prev_pts, None
                        )
                        if curr_pts is not None and status is not None:
                            # Only keep successfully tracked points
                            status = status.reshape(-1).astype(bool)
                            if status.sum() > 0:
                                displacements = (
                                    prev_pts[status] - curr_pts[status]
                                )
                                avg_motion = float(
                                    (displacements ** 2).sum(axis=1).mean() ** 0.5
                                )
                            else:
                                avg_motion = 0.0
                        else:
                            avg_motion = 0.0
                    else:
                        avg_motion = 0.0

                    # Classify this segment
                    if avg_motion < _THRESHOLD_LOW:
                        classifications.append(_STATIC)
                    elif avg_motion < _THRESHOLD_HIGH:
                        classifications.append(_SLIGHT)
                    else:
                        classifications.append(_VIGOROUS)

                prev_gray = gray
                frame_idx += sample_interval

        finally:
            cap.release()

        total_segments = len(classifications)
        if total_segments == 0:
            logger.warning("_analyze_motion: 无运动段可分析")
            return "静态为主 (因视频过短)"

        static_cnt = classifications.count(_STATIC)
        slight_cnt = classifications.count(_SLIGHT)
        vigorous_cnt = classifications.count(_VIGOROUS)

        static_pct = round(100.0 * static_cnt / total_segments)
        slight_pct = round(100.0 * slight_cnt / total_segments)
        vigorous_pct = round(100.0 * vigorous_cnt / total_segments)

        # Build weighted summary — list parts with >0% contribution
        parts: list[str] = []
        if static_pct > 0:
            parts.append(f"{_STATIC} ({static_pct}%)")
        if slight_pct > 0:
            parts.append(f"{_SLIGHT} ({slight_pct}%)")
        if vigorous_pct > 0:
            parts.append(f"{_VIGOROUS} ({vigorous_pct}%)")

        if not parts:
            return "静态为主 (100%)"

        return "，".join(parts)

    def _detect_text(self, file_path: str, shots: list[dict]) -> list[dict]:
        """Detect text regions in shot keyframes using contour-based heuristics.

        For each shot's keyframe (sampled at the shot midpoint), this method
        applies a simple contour-based text candidate detection:
        1. Convert to grayscale and apply adaptive thresholding.
        2. Find external contours.
        3. Filter by aspect ratio and area to identify text-like regions.

        Returns a list of dicts with keys:
            shot_index, has_text, bbox (or None), timestamp
        """
        import logging
        logger = logging.getLogger(__name__)

        cap = cv2.VideoCapture(file_path)
        results: list[dict] = []

        try:
            for shot in shots:
                mid_time = (shot["start_time"] + shot["end_time"]) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000.0)
                ret, frame = cap.read()

                entry: dict = {
                    "shot_index": shot["index"],
                    "has_text": False,
                    "bbox": None,
                    "timestamp": round(mid_time, 2),
                }

                if not ret:
                    results.append(entry)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Adaptive threshold to highlight potential text regions
                thresh = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11, 2,
                )
                # Morphological close to merge nearby text components
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(
                    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                text_bboxes: list[dict] = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    aspect_ratio = w / h if h > 0 else 0

                    # Filter: area between 100 and 10000, aspect ratio typical
                    # of characters/words (0.1 – 10)
                    if 100 < area < 10000 and 0.1 < aspect_ratio < 10:
                        text_bboxes.append({
                            "x": int(x),
                            "y": int(y),
                            "w": int(w),
                            "h": int(h),
                        })

                if text_bboxes:
                    entry["has_text"] = True
                    entry["bbox"] = text_bboxes

                results.append(entry)

        finally:
            cap.release()

        return results

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

        # -- Task 10: Color analysis (92 %) ------------------------------
        _report(progress_cb, 92.0, "分析色彩直方图...")
        color_summary = self._analyze_colors(all_thumbnail_paths)

        # -- Task 10: Face detection (93 %) ------------------------------
        _report(progress_cb, 93.0, "检测人脸...")
        face_detections = self._detect_faces(file_path, shots)

        # -- Task 10: Object detection (94 %) ----------------------------
        _report(progress_cb, 94.0, "检测物体...")
        object_detections = self._detect_objects(file_path, shots)

        # -- Task 10: CLIP style classification (95 %) -------------------
        _report(progress_cb, 95.0, "CLIP 风格分类...")
        style_tags = self._classify_style(all_thumbnail_paths)

        # -- Task 11: Motion analysis (97 %) -----------------------------
        _report(progress_cb, 97.0, "运动分析（光流法）...")
        motion_summary = self._analyze_motion(file_path, shots)

        # -- Task 11: Text detection (99 %) ------------------------------
        _report(progress_cb, 99.0, "检测文字区域...")
        text_regions = self._detect_text(file_path, shots)

        # -- 100 % -------------------------------------------------------
        _report(progress_cb, 100.0, "视觉分析完成")

        return {
            "shots": shots,
            "shot_count": shot_count,
            "avg_shot_duration": round(avg_shot_dur, 2),
            "transitions": transitions,
            "keyframe_grid_paths": grid_paths,
            "representative_frames": representative,
            "color_summary": color_summary,          # Task 10
            "text_regions": text_regions,            # Task 11
            "face_detections": face_detections,      # Task 10
            "object_detections": object_detections,  # Task 10
            "motion_summary": motion_summary,        # Task 11
            "style_tags": style_tags,                # Task 10 (new field)
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
