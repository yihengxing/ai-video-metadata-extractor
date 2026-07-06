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
        """Lazy-load OpenCLIP model. Returns (model, preprocess, tokenizer)."""
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
            except ImportError as e:
                import logging
                logging.getLogger(__name__).warning(
                    "OpenCLIP 未安装，风格分类不可用: %s", e
                )
                return None, None, None
        return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    def _get_yolo(self):
        """Lazy-load YOLOv8 ONNX model. Returns YOLO instance or None."""
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO("yolov8n.pt")
            except ImportError as e:
                import logging
                logging.getLogger(__name__).warning(
                    "ultralytics 未安装，YOLO 物体检测不可用: %s", e
                )
                return None
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "YOLO 模型加载失败: %s", e
                )
                return None
        return self._yolo_model

    def _get_face_cascade(self):
        """Lazy-load OpenCV Haar cascade for face detection."""
        if self._face_cascade is None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "OpenCV 人脸检测级联加载失败: %s", e
                )
                return None
        return self._face_cascade

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

        # -- Task 10: Face detection (94 %) ------------------------------
        _report(progress_cb, 94.0, "检测人脸...")
        face_detections = self._detect_faces(file_path, shots)

        # -- Task 10: Object detection (96 %) ----------------------------
        _report(progress_cb, 96.0, "检测物体...")
        object_detections = self._detect_objects(file_path, shots)

        # -- Task 10: CLIP style classification (98 %) -------------------
        _report(progress_cb, 98.0, "CLIP 风格分类...")
        style_tags = self._classify_style(all_thumbnail_paths)

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
            "text_regions": [],                      # Task 11
            "face_detections": face_detections,      # Task 10
            "object_detections": object_detections,  # Task 10
            "motion_summary": None,                  # Task 11
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
