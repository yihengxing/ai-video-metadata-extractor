"""分析模块抽象基类 — Extractor（提取器）与 Matcher（匹配器，v1.3）。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, Optional
from backend.models.schemas import TechMetadata, SourceRecoveryHit


# 进度回调签名: (module_name: str, progress_pct: float, message: str) -> None
ProgressCallback = Callable[[str, float, str], None]


class Extractor(ABC):
    """分析提取器基类。每个分析模块（技术/视觉/音频/AI推断）实现此接口。"""

    @abstractmethod
    async def extract(
        self,
        file_path: str,
        tech_meta: TechMetadata,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> dict:
        """执行提取，返回模块结果 dict。"""
        ...

    @property
    @abstractmethod
    def module_name(self) -> str:
        """返回模块唯一标识: "tech" | "visual" | "audio" | "ai"."""
        ...


class Matcher(ABC):
    """源回捞匹配器接口（v1.3 新增）。可插拔设计：v1.0 实装 SauceNAO 路由匹配器，
    后期可新增本地 CLIP 索引匹配器作为另一实现。"""

    @abstractmethod
    async def match(
        self,
        keyframe_paths: list[str],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> list[SourceRecoveryHit]:
        """对关键帧列表执行反向搜索，返回命中列表。"""
        ...

    @property
    @abstractmethod
    def matcher_name(self) -> str:
        """返回匹配器唯一标识。"""
        ...
