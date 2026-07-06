"""技术元数据提取器 — FFmpeg 探针，§4.1 全部字段。"""
from __future__ import annotations
from typing import Optional
from backend.modules.base import Extractor, ProgressCallback
from backend.models.schemas import TechMetadata
from backend.services.ffmpeg_service import FFmpegService


class TechExtractor(Extractor):
    """技术提取器：编码参数、分辨率、帧率、码率、GOP、色彩空间、平台指纹。"""

    @property
    def module_name(self) -> str:
        return "tech"

    async def extract(
        self,
        file_path: str,
        tech_meta: TechMetadata,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> dict:
        if progress_cb:
            progress_cb("tech", 0.0, "开始技术提取...")

        if not FFmpegService.is_installed():
            raise RuntimeError(
                "FFmpeg 未安装。请安装 FFmpeg 或将 ffprobe 放入应用目录。"
            )

        if progress_cb:
            progress_cb("tech", 30.0, "运行 ffprobe 探针...")

        raw_probe = await FFmpegService.probe(file_path)

        if progress_cb:
            progress_cb("tech", 70.0, "解析探针数据...")

        parsed = FFmpegService.parse(raw_probe)

        if progress_cb:
            progress_cb("tech", 100.0, "技术提取完成")

        return parsed
