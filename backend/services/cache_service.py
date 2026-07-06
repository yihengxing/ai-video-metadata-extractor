"""SHA-256 文件哈希缓存服务 — 避免重复分析同一视频文件。"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


DEFAULT_CACHE_DIR = Path.home() / ".ai-video-analyzer" / "cache"


class CacheService:
    """基于文件系统的 JSON 缓存。
    路径: {cache_dir}/{hash[:2]}/{hash}.json
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    def _cache_path(self, file_hash: str) -> Path:
        prefix = file_hash[:2]
        return self._root / prefix / f"{file_hash}.json"

    def exists(self, file_hash: str) -> bool:
        return self._cache_path(file_hash).exists()

    def get(self, file_hash: str) -> Optional[dict]:
        path = self._cache_path(file_hash)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, file_hash: str, data: dict) -> None:
        path = self._cache_path(file_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 确保缓存数据包含时间戳
        data.setdefault("cached_at", datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, file_hash: str) -> None:
        path = self._cache_path(file_hash)
        if path.exists():
            path.unlink()

    def list_all(self) -> list[dict]:
        """List all cached entries with metadata."""
        items: list[dict] = []
        if not self._root.exists():
            return items
        for prefix_dir in self._root.iterdir():
            if not prefix_dir.is_dir():
                continue
            for cache_file in prefix_dir.glob("*.json"):
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    items.append({
                        "file_hash": data.get("file_hash", cache_file.stem),
                        "file_path": data.get("file_path", ""),
                        "cached_at": data.get("cached_at", ""),
                    })
                except (json.JSONDecodeError, OSError):
                    continue
        return items
