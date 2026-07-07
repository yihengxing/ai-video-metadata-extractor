"""AI inference engine — multi-modal LLM client for video metadata extraction.

Combined Tasks 15+16: sends up to 5 keyframes + technical metadata + optional
audio text to a multimodal LLM (Claude/GPT-4V) and parses the structured JSON
response into an :class:`AIInference`-compatible dict.

Supports:
- Claude Messages API (Anthropic)
- OpenAI Chat Completions API (GPT-4V / GPT-4o)
- 3 retries with exponential backoff (1s, 2s, 4s)
- 60-second timeout
- Graceful degradation when API key is missing or all retries fail
"""
from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
from typing import Optional

import httpx

from backend.modules.base import Extractor, ProgressCallback
from backend.models.schemas import TechMetadata
from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_KEYFRAMES = 5
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds: 1, 2, 4
_REQUEST_TIMEOUT = 60.0  # seconds

_CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
_OPENAI_MODEL = "gpt-4o"

_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
_GEMINI_MODEL = "gemini-2.0-flash"

_QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
_QWEN_MODEL = "qwen-vl-max"

_THUMB_BASE = os.path.expanduser("~/.ai-video-analyzer/thumbnails")

_SYSTEM_PROMPT = (
    "你是一位专业的 AI 视频分析专家。根据提供的视频关键帧、技术元数据和音频文本，"
    "请分析并推断以下内容：\n"
    "1. 使用了哪个 AI 视频生成工具（如可灵、Sora、Runway 等）\n"
    "2. 可能的生成提示词\n"
    "3. 画面风格标签\n"
    "4. 可能的生成工作流\n"
    "5. 模仿建议和替代模型推荐\n\n"
    "对每个推断提供 0-1 的置信度。仅返回 JSON，不要其他文字。\n\n"
    '返回格式示例：\n'
    '{\n'
    '  "inferred_tool": "可灵1.6",\n'
    '  "inferred_tool_confidence": 0.85,\n'
    '  "inferred_prompt": "赛博朋克城市夜景，霓虹灯，雨夜...",\n'
    '  "inferred_prompt_confidence": 0.7,\n'
    '  "style_tags": ["赛博朋克", "电影感", "暖色调"],\n'
    '  "inferred_workflow": "文生视频(可灵) → Topaz超分 → AE调色",\n'
    '  "inferred_workflow_confidence": 0.65,\n'
    '  "imitation_suggestions": ["建议使用可灵1.6 文生视频", "Topaz Video AI 2x超分"],\n'
    '  "model_recommendations": ["ComfyUI + AnimateDiff 替代方案"]\n'
    '}'
)


# ---------------------------------------------------------------------------
# AIInferrer
# ---------------------------------------------------------------------------

class AIInferrer(Extractor):
    """Multi-modal LLM inference engine for AI video metadata extraction.

    Sends keyframes + technical metadata + optional audio text to a
    multi-modal LLM and parses the structured JSON inference result.
    """

    @property
    def module_name(self) -> str:
        return "ai"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(
        self,
        file_path: str,
        tech_meta: TechMetadata,
        progress_cb: Optional[ProgressCallback] = None,
        keyframe_paths: Optional[list[str]] = None,
        audio_text: str = "",
    ) -> dict:
        """Run AI inference from keyframes, tech metadata, and audio text.

        Parameters
        ----------
        file_path : str
            Video file path (used for hash-based keyframe discovery).
        tech_meta : TechMetadata
            Technical metadata from the video probe.
        progress_cb : ProgressCallback, optional
            Progress reporting callback (module, pct, message).
        keyframe_paths : list[str], optional
            Explicit list of keyframe image paths.  If not provided,
            auto-discovers from the thumbnail cache directory.
        audio_text : str, optional
            Transcribed audio text to include in the prompt context.

        Returns
        -------
        dict
            AIInference-compatible dictionary.  All fields are set to
            ``None`` / empty / 0.0 when inference cannot be performed
            (no API key, no keyframes, or all retries exhausted).
        """
        await _report(progress_cb, 0.0, "开始AI推断...")

        # --- Check API key --------------------------------------------------
        api_key = settings.llm_api_key
        if not api_key:
            logger.warning("未配置LLM API密钥，跳过AI推断")
            return self._empty_result()

        # --- Load keyframes ------------------------------------------------
        await _report(progress_cb, 10.0, "加载关键帧...")
        if keyframe_paths is None:
            keyframe_paths = self._find_keyframes(file_path)
        keyframe_paths = [p for p in keyframe_paths if os.path.exists(p)]
        keyframe_paths = keyframe_paths[:_MAX_KEYFRAMES]

        if not keyframe_paths:
            logger.warning("无可用的关键帧，跳过AI推断")
            await _report(progress_cb, 100.0, "AI推断跳过（无关键帧）")
            return self._empty_result()

        # --- Encode images as base64 ---------------------------------------
        await _report(progress_cb, 20.0, "编码关键帧为base64...")
        images_b64: list[str] = []
        for path in keyframe_paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                images_b64.append(base64.b64encode(data).decode("utf-8"))
            except Exception as exc:
                logger.warning("读取关键帧失败 %s: %s", path, exc)

        if not images_b64:
            logger.warning("所有关键帧编码失败，跳过AI推断")
            return self._empty_result()

        # --- Build prompt --------------------------------------------------
        user_text = self._build_user_prompt(tech_meta, audio_text)

        # --- Call LLM with retry -------------------------------------------
        await _report(progress_cb, 50.0, "调用多模态LLM推断...")
        response_text = await self._call_llm(images_b64, user_text)

        if response_text is None:
            logger.error("LLM调用失败（所有重试已用尽）")
            await _report(progress_cb, 100.0, "AI推断失败（LLM不可用）")
            return self._empty_result()

        # --- Parse response ------------------------------------------------
        await _report(progress_cb, 90.0, "解析LLM响应...")
        parsed = self._parse_response(response_text)

        # --- Compute overall confidence ------------------------------------
        confidences = [
            parsed.get("inferred_tool_confidence", 0.0),
            parsed.get("inferred_prompt_confidence", 0.0),
            parsed.get("inferred_workflow_confidence", 0.0),
        ]
        parsed["overall_confidence"] = round(sum(confidences) / 3.0, 4)

        await _report(progress_cb, 100.0, "AI推断完成")
        return parsed

    # ------------------------------------------------------------------
    # LLM calling with retry
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        images_b64: list[str],
        user_text: str,
    ) -> Optional[str]:
        """Call the configured LLM provider with retry.

        Returns the raw text content from the response, or ``None`` if
        all attempts are exhausted.
        """
        api_key = settings.llm_api_key
        provider = settings.llm_provider or "claude"

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                    if provider == "openai":
                        body, headers = self._build_openai_request(images_b64, user_text, api_key)
                        endpoint = _OPENAI_API_URL
                    elif provider == "gemini":
                        body, headers = self._build_gemini_request(images_b64, user_text, api_key)
                        endpoint = f"{_GEMINI_API_URL}?key={api_key}"
                    elif provider == "qwen":
                        body, headers = self._build_qwen_request(images_b64, user_text, api_key)
                        endpoint = _QWEN_API_URL
                    elif provider == "custom":
                        body, headers = self._build_custom_request(images_b64, user_text, api_key)
                        endpoint = settings.get("llm_custom_endpoint", "")
                        if not endpoint:
                            logger.error("自定义端点未配置")
                            return None
                    else:
                        # Default to Claude
                        body, headers = self._build_claude_request(images_b64, user_text, api_key)
                        endpoint = _CLAUDE_API_URL

                    resp = await client.post(endpoint, json=body, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = self._extract_text(provider, data)

                    logger.info("LLM调用成功（第%d次尝试）", attempt)
                    return raw_text

            except Exception as exc:
                logger.warning(
                    "LLM调用失败（第%d/%d次）: %s",
                    attempt, _MAX_RETRIES, exc,
                )
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

        return None

    # ------------------------------------------------------------------
    # Request builders (one per provider)
    # ------------------------------------------------------------------

    def _build_claude_request(
        self, images_b64: list[str], user_text: str, api_key: str
    ) -> tuple[dict, dict]:
        """Claude Messages API."""
        content: list[dict] = []
        for img_b64 in images_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
            })
        content.append({"type": "text", "text": user_text})

        body = {
            "model": _CLAUDE_MODEL,
            "max_tokens": 2048,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        return body, headers

    def _build_openai_request(
        self, images_b64: list[str], user_text: str, api_key: str
    ) -> tuple[dict, dict]:
        """OpenAI Chat Completions API (GPT-4V / GPT-4o)."""
        user_content: list[dict] = []
        for img_b64 in images_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"},
            })
        user_content.append({"type": "text", "text": user_text})

        body = {
            "model": _OPENAI_MODEL,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return body, headers

    def _build_gemini_request(
        self, images_b64: list[str], user_text: str, api_key: str
    ) -> tuple[dict, dict]:
        """Google Gemini API (Gemini 2.0 Flash — fast + free tier)."""
        parts: list[dict] = []
        for img_b64 in images_b64:
            parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": img_b64},
            })
        parts.append({"text": f"{_SYSTEM_PROMPT}\n\n{user_text}"})

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.4},
        }
        headers = {"Content-Type": "application/json"}
        return body, headers

    def _build_qwen_request(
        self, images_b64: list[str], user_text: str, api_key: str
    ) -> tuple[dict, dict]:
        """Qwen-VL via Alibaba DashScope (OpenAI-compatible endpoint).
        Best native Chinese multimodal model.
        """
        user_content: list[dict] = []
        for img_b64 in images_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        user_content.append({"type": "text", "text": user_text})

        body = {
            "model": _QWEN_MODEL,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return body, headers

    def _build_custom_request(
        self, images_b64: list[str], user_text: str, api_key: str
    ) -> tuple[dict, dict]:
        """Custom OpenAI-compatible endpoint (e.g. API proxy, self-hosted).
        Uses the same format as OpenAI but with configurable model name.
        """
        model = settings.get("llm_custom_model") or "gpt-4o"
        user_content: list[dict] = []
        for img_b64 in images_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"},
            })
        user_content.append({"type": "text", "text": user_text})

        body = {
            "model": model,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return body, headers

    # ------------------------------------------------------------------
    # Response text extraction (provider-specific)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(provider: str, data: dict) -> str:
        """Extract the text content from a provider-specific API response."""
        if provider == "gemini":
            # Gemini returns: candidates[0].content.parts[0].text
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return ""
        elif provider in ("openai", "qwen", "custom"):
            # OpenAI-compatible: choices[0].message.content
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                return ""
        else:
            # Claude: content[0].text
            try:
                return data["content"][0]["text"]
            except (KeyError, IndexError):
                return ""

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        tech_meta: TechMetadata,
        audio_text: str = "",
    ) -> str:
        """Build the user-facing prompt string with metadata and optional
        audio transcription."""
        lines = [
            "请分析以下视频并返回JSON：",
            "",
            "## 技术元数据",
            f"- 容器格式: {tech_meta.container_format}",
            f"- 视频编码: {tech_meta.video_codec} ({tech_meta.video_profile})",
            f"- 分辨率: {tech_meta.resolution_width}x{tech_meta.resolution_height}",
            f"- 帧率: {tech_meta.frame_rate} fps",
            f"- 视频码率: {tech_meta.video_bitrate_bps} bps",
            f"- 音频编码: {tech_meta.audio_codec}",
            f"- 采样率: {tech_meta.audio_sample_rate_hz} Hz",
            f"- 时长: {tech_meta.duration:.1f}秒",
            f"- 色彩空间: {tech_meta.color_space} / {tech_meta.hdr_info}",
            f"- GOP结构: {tech_meta.gop_structure}",
        ]
        if tech_meta.platform_fingerprint:
            lines.append(f"- 平台指纹: {tech_meta.platform_fingerprint}")

        if audio_text:
            lines.append("")
            lines.append("## 提取的音频文本")
            lines.append(audio_text)

        lines.append("")
        lines.append("请根据以上信息和上传的关键帧图像，返回JSON推断结果。")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw_text: str) -> dict:
        """Parse the LLM response text into an AIInference-compatible dict.

        Attempts to extract JSON from the raw text.  If parsing fails,
        returns an empty result.
        """
        # Try direct JSON parse first
        try:
            data = json.loads(raw_text)
            return self._normalize_fields(data)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block from markdown or raw text
        json_match = re.search(
            r'\{[^{}]*"inferred_tool"[^{}]*\}',
            raw_text, re.DOTALL,
        )
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._normalize_fields(data)
            except json.JSONDecodeError:
                pass

        # Broader attempt: find any JSON object
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "inferred_tool" in data:
                    return self._normalize_fields(data)
            except json.JSONDecodeError:
                pass

        logger.warning("无法从LLM响应中提取有效JSON: %s", raw_text[:200])
        return self._empty_result()

    def _normalize_fields(self, data: dict) -> dict:
        """Normalize and validate parsed fields against the AIInference schema."""
        result = {
            "inferred_tool": data.get("inferred_tool"),
            "inferred_tool_confidence": float(data.get("inferred_tool_confidence", 0.0)),
            "inferred_prompt": data.get("inferred_prompt"),
            "inferred_prompt_confidence": float(data.get("inferred_prompt_confidence", 0.0)),
            "style_tags": data.get("style_tags", []) or [],
            "inferred_workflow": data.get("inferred_workflow"),
            "inferred_workflow_confidence": float(data.get("inferred_workflow_confidence", 0.0)),
            "imitation_suggestions": data.get("imitation_suggestions", []) or [],
            "model_recommendations": data.get("model_recommendations", []) or [],
            "overall_confidence": 0.0,  # Computed later by caller
        }
        return result

    # ------------------------------------------------------------------
    # Keyframe auto-discovery
    # ------------------------------------------------------------------

    def _find_keyframes(self, file_path: str) -> list[str]:
        """Try to auto-discover keyframes from the thumbnail cache directory.

        Looks for grid_*.jpg and shot_*.jpg files under the per-video
        thumbnail subdirectory derived from the file path hash.
        """
        file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        thumb_dir = os.path.join(_THUMB_BASE, file_hash)
        if not os.path.isdir(thumb_dir):
            return []

        # Prefer representative frames (shot_*.jpg) over grid
        paths: list[str] = []
        for name in sorted(os.listdir(thumb_dir)):
            if name.lower().endswith(".jpg"):
                paths.append(os.path.join(thumb_dir, name))

        # Return up to _MAX_KEYFRAMES, prioritizing grid (uniform) first
        # then shots (which may have duplicates of same scene)
        grid_paths = [p for p in paths if "grid_" in os.path.basename(p)]
        shot_paths = [p for p in paths if "shot_" in os.path.basename(p)]
        combined = grid_paths + shot_paths
        return combined[:_MAX_KEYFRAMES]

    # ------------------------------------------------------------------
    # Empty / graceful-degradation result
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        """Return an empty AIInference-compatible dict for graceful degradation."""
        return {
            "inferred_tool": None,
            "inferred_tool_confidence": 0.0,
            "inferred_prompt": None,
            "inferred_prompt_confidence": 0.0,
            "style_tags": [],
            "inferred_workflow": None,
            "inferred_workflow_confidence": 0.0,
            "imitation_suggestions": [],
            "model_recommendations": [],
            "overall_confidence": 0.0,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _report(cb: Optional[ProgressCallback], pct: float, msg: str) -> None:
    """Fire progress callback if provided."""
    if cb:
        await cb("ai", pct, msg)
