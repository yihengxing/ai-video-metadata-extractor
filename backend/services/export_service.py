"""Export service — converts AnalysisResult into all supported output formats."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional

from backend.models.schemas import AnalysisResult


class ExportService:
    """Stateless exporter: every method takes an AnalysisResult and returns a string (or None)."""

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    @staticmethod
    def export_json(result: AnalysisResult) -> str:
        """Serialize the full AnalysisResult to pretty-printed JSON."""
        data = json.loads(result.model_dump_json())
        data["exported_at"] = datetime.now(timezone.utc).isoformat()
        return json.dumps(data, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    @staticmethod
    def export_markdown(result: AnalysisResult) -> str:
        """Generate a Chinese-language Markdown report."""
        lines: list[str] = []

        # -- Title --
        lines.append("# AI 视频分析报告")
        lines.append("")

        # -- Basic info --
        lines.append("## 基本信息")
        lines.append("")
        lines.append(f"- 文件: {result.file_path}")
        lines.append(f"- 哈希: {result.file_hash[:16]}...")
        lines.append(f"- 分析时间: {result.analyzed_at.isoformat()}")
        lines.append("")

        # -- Tech metadata --
        lines.append("## 技术元数据")
        lines.append("")
        lines.append("| 字段 | 值 |")
        lines.append("|------|-----|")
        tm = result.tech_metadata
        tech_fields = [
            ("容器格式", tm.container_format),
            ("视频编码", tm.video_codec),
            ("视频 Profile", tm.video_profile),
            ("分辨率", f"{tm.resolution_width}x{tm.resolution_height}"),
            ("帧率", f"{tm.frame_rate} fps"),
            ("总比特率", f"{tm.total_bitrate_bps} bps"),
            ("视频比特率", f"{tm.video_bitrate_bps} bps"),
            ("音频编码", tm.audio_codec),
            ("音频采样率", f"{tm.audio_sample_rate_hz} Hz"),
            ("音频比特率", f"{tm.audio_bitrate_bps} bps"),
            ("GOP 结构", tm.gop_structure),
            ("色彩空间", tm.color_space),
            ("HDR 信息", tm.hdr_info),
            ("时长", f"{tm.duration} 秒"),
            ("文件大小", f"{tm.file_size_bytes} 字节"),
        ]
        if tm.platform_fingerprint:
            tech_fields.append(("平台指纹", tm.platform_fingerprint))
        for label, value in tech_fields:
            lines.append(f"| {label} | {value} |")
        lines.append("")

        # -- Visual analysis --
        if result.visual_analysis:
            va = result.visual_analysis
            lines.append("## 镜头分析")
            lines.append("")
            lines.append(f"- 镜头数量: {va.shot_count}")
            lines.append(f"- 平均时长: {va.avg_shot_duration}s")
            if va.color_summary:
                lines.append(f"- 画面色调: {va.color_summary.description}")
                lines.append(f"- 主色调: {va.color_summary.dominant_hue}")
                lines.append(f"- 饱和度: {va.color_summary.saturation}")
            if va.motion_summary:
                lines.append(f"- 运动总结: {va.motion_summary}")
            if va.transitions:
                lines.append(f"- 转场类型: {', '.join(va.transitions)}")
            lines.append("")

        # -- Audio analysis --
        if result.audio_analysis:
            aa = result.audio_analysis
            lines.append("## 音频分析")
            lines.append("")
            if aa.full_text:
                lines.append(f"- 配音全文: {aa.full_text}")
            if aa.speech_rate:
                lines.append(f"- 语速: {aa.speech_rate} 字符/秒")
            if aa.speech_emotion:
                lines.append(f"- 语音情绪: {aa.speech_emotion}")
            if aa.bgm_title:
                artist = f" - {aa.bgm_artist}" if aa.bgm_artist else ""
                lines.append(f"- BGM: {aa.bgm_title}{artist}")
            if aa.bgm_style_tags:
                lines.append(f"- BGM 风格: {', '.join(aa.bgm_style_tags)}")
            if aa.bgm_emotion:
                lines.append(f"- BGM 情绪: {aa.bgm_emotion}")
            if aa.bgm_bpm is not None:
                lines.append(f"- BGM BPM: {aa.bgm_bpm}")
            if aa.sound_events:
                lines.append(f"- 声音事件: {', '.join(aa.sound_events)}")
            if aa.voice_to_bg_ratio:
                lines.append(f"- 人声/背景比: {aa.voice_to_bg_ratio}")
            if aa.audio_structure:
                lines.append(f"- 音频结构: {aa.audio_structure}")
            lines.append("")

        # -- AI inference --
        if result.ai_inference:
            ai = result.ai_inference
            lines.append("## AI 推断")
            lines.append("")
            if ai.inferred_tool:
                lines.append(f"- 推测工具: {ai.inferred_tool}")
                lines.append(f"- 工具置信度: {ai.inferred_tool_confidence:.0%}")
            if ai.inferred_prompt:
                lines.append(f"- 推测提示词: {ai.inferred_prompt}")
                lines.append(f"- 提示词置信度: {ai.inferred_prompt_confidence:.0%}")
            if ai.style_tags:
                lines.append(f"- 风格标签: {', '.join(ai.style_tags)}")
            if ai.inferred_workflow:
                lines.append(f"- 推测工作流: {ai.inferred_workflow}")
                lines.append(f"- 工作流置信度: {ai.inferred_workflow_confidence:.0%}")
            if ai.model_recommendations:
                lines.append(f"- 模型推荐: {', '.join(ai.model_recommendations)}")
            if ai.imitation_suggestions:
                lines.append(f"- 模仿建议: {', '.join(ai.imitation_suggestions)}")
            lines.append(f"- 综合置信度: {ai.overall_confidence:.0%}")
            lines.append("")

        # -- Source recovery --
        if result.source_recovery and result.source_recovery.status in ("complete_match", "partial_match"):
            sr = result.source_recovery
            lines.append("## 原始来源与生成参数")
            lines.append("")
            if sr.source_url:
                lines.append(f"- 来源: {sr.source_url}")
            if sr.prompt:
                lines.append(f"- 原始 Prompt: {sr.prompt}")
            if sr.seed is not None:
                lines.append(f"- Seed: {sr.seed}")
            if sr.sampler:
                lines.append(f"- Sampler: {sr.sampler}")
            if sr.steps is not None:
                lines.append(f"- Steps: {sr.steps}")
            if sr.cfg_scale is not None:
                lines.append(f"- CFG Scale: {sr.cfg_scale}")
            if sr.model_name:
                lines.append(f"- 模型: {sr.model_name}")
            if sr.source_trust:
                lines.append(f"- 来源信任度: {sr.source_trust}")
            if sr.workflow_json:
                lines.append(f"- 工作流 JSON: 已恢复 (长度 {len(sr.workflow_json)} 字符)")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ComfyUI workflow
    # ------------------------------------------------------------------

    @staticmethod
    def export_comfyui_workflow(result: AnalysisResult) -> Optional[str]:
        """Return the original ComfyUI workflow JSON if recovered, else None."""
        if result.source_recovery and result.source_recovery.workflow_json:
            return result.source_recovery.workflow_json
        return None

    # ------------------------------------------------------------------
    # ComfyUI prompt
    # ------------------------------------------------------------------

    @staticmethod
    def export_comfyui_prompt(result: AnalysisResult) -> str:
        """Format prompt text for ComfyUI consumption."""
        lines: list[str] = []

        if result.source_recovery and result.source_recovery.prompt:
            lines.append("# 原始 Prompt (原始)")
            lines.append(result.source_recovery.prompt)
            if result.source_recovery.model_name:
                lines.append("")
                lines.append("# 原始模型")
                lines.append(result.source_recovery.model_name)
        elif result.ai_inference and result.ai_inference.inferred_prompt:
            lines.append("# AI 推断提示词")
            lines.append(result.ai_inference.inferred_prompt)
        else:
            lines.append("# (无可用提示词)")

        lines.append("")

        # Model recommendations
        recommendations: list[str] = []
        if result.source_recovery and result.source_recovery.model_name:
            recommendations.append(result.source_recovery.model_name)
        if result.ai_inference and result.ai_inference.model_recommendations:
            recommendations.extend(result.ai_inference.model_recommendations)
        if recommendations:
            lines.append("# 建议模型")
            for r in recommendations:
                lines.append(f"- {r}")

        # Confidence
        confidence_parts: list[str] = []
        if result.ai_inference and result.ai_inference.overall_confidence:
            confidence_parts.append(f"# 置信度: {result.ai_inference.overall_confidence:.0%}")
        if result.source_recovery and result.source_recovery.confidence_score:
            confidence_parts.append(f"# 来源置信度: {result.source_recovery.confidence_score:.0%}")
        if confidence_parts:
            lines.append("")
            for cp in confidence_parts:
                lines.append(cp)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SRT subtitles
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_to_srt_timestamp(seconds: float) -> str:
        """Convert float seconds to SRT timestamp format HH:MM:SS,mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def export_srt(result: AnalysisResult) -> str:
        """Convert Whisper text_segments to SRT subtitle format."""
        if not result.audio_analysis or not result.audio_analysis.text_segments:
            return ""

        blocks: list[str] = []
        for i, seg in enumerate(result.audio_analysis.text_segments, start=1):
            start = ExportService._seconds_to_srt_timestamp(seg["start"])
            end = ExportService._seconds_to_srt_timestamp(seg["end"])
            text = seg.get("text", "")
            blocks.append(f"{i}")
            blocks.append(f"{start} --> {end}")
            blocks.append(text)
            blocks.append("")

        return "\n".join(blocks)

    # ------------------------------------------------------------------
    # All formats
    # ------------------------------------------------------------------

    @staticmethod
    def export_all(result: AnalysisResult) -> dict[str, Optional[str]]:
        """Return all export formats in a single dict."""
        return {
            "json": ExportService.export_json(result),
            "markdown": ExportService.export_markdown(result),
            "comfyui_workflow": ExportService.export_comfyui_workflow(result),
            "comfyui_prompt": ExportService.export_comfyui_prompt(result),
            "srt": ExportService.export_srt(result),
        }
