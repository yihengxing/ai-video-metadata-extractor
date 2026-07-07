"""文件校验与哈希工具。"""
import hashlib
import os
from typing import Tuple

from backend.config import SUPPORTED_EXTENSIONS

SUPPORTED_EXTENSIONS_STR = "、".join(sorted(SUPPORTED_EXTENSIONS))


def compute_sha256(file_path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """计算文件 SHA-256 哈希（8MB 分块）。"""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """校验视频文件：存在性 + 扩展名支持。返回 (is_valid, error_message)。"""
    if not os.path.isfile(file_path):
        return False, f"文件不存在: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"不支持的格式 {ext}，支持: {SUPPORTED_EXTENSIONS_STR}"

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "文件为空"

    if file_size > 2 * 1024 * 1024 * 1024:  # 2GB
        return True, "文件超过 2GB，分析可能较慢"

    return True, ""
