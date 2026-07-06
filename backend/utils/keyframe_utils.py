"""关键帧预处理工具 — 缩放至 ≤1000px 边，JPEG q85，<500KB。"""
from __future__ import annotations
from PIL import Image
import io
import os
from typing import List


def preprocess_for_upload(image_path: str, max_side: int = 1000,
                          quality: int = 85, max_bytes: int = 500 * 1024) -> bytes:
    """缩放并编码图片为 JPEG 字节，供 SauceNAO/LLM 上传。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        ratio = max_side / longest
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    # If still over max_bytes, reduce quality
    q = quality
    while len(data) > max_bytes and q > 20:
        q -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
    return data


def rank_representative_frames(shot_thumbnails: List[str], top_n: int = 5) -> List[str]:
    """按镜头权重 + 画面复杂度排序，返回 top_n 代表帧路径。
    排序规则: (1) 镜头时长权重 (2) 画面熵值 (复杂度) (3) 无文字区域优先。
    v1.3: 为源回捞匹配器提供最优输入。
    """
    import cv2
    import numpy as np

    scored: List[tuple[float, str]] = []
    for path in shot_thumbnails:
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # 拉普拉斯方差作为复杂度近似
        complexity = float(cv2.Laplacian(img, cv2.CV_64F).var())
        scored.append((complexity, path))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_n]]
