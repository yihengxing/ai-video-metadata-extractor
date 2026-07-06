"""FFmpeg 命令行封装 — 探针提取、格式检查。"""
from __future__ import annotations
import asyncio
import json
import os
import shutil
from typing import Optional


# 平台转码指纹规则库（§4.1）
PLATFORM_FINGERPRINTS = [
    {"name": "B站", "hint": "GOP对齐平台标准", "gop_interval": 2.0, "codec": "h264", "profile": "High"},
    {"name": "抖音", "hint": "HEVC MainProfile", "codec": "hevc", "profile": "Main"},
]


def _find_ffprobe() -> str:
    """查找 ffprobe 可执行文件。"""
    # 优先使用打包的二进制
    bundled = os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "ffprobe.exe")
    if os.path.exists(bundled):
        return bundled
    bundled_nix = os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "ffprobe")
    if os.path.exists(bundled_nix):
        return bundled_nix
    # Fallback to PATH
    path = shutil.which("ffprobe")
    if path:
        return path
    raise RuntimeError("FFprobe 未找到，请安装 FFmpeg")


def check_ffprobe_installed() -> bool:
    try:
        _find_ffprobe()
        return True
    except RuntimeError:
        return False


async def probe(file_path: str) -> dict:
    """运行 ffprobe 获取完整探针数据，返回原始 JSON dict。"""
    ffprobe_path = _find_ffprobe()
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {stderr.decode()}")
    return json.loads(stdout.decode())


def parse_ffprobe_output(raw: dict) -> dict:
    """将 ffprobe JSON 输出解析为 §4.1 技术元数据字段。"""
    fmt = raw.get("format", {})
    streams = raw.get("streams", [])

    # 检测流类型: codec_type 字段不一定存在，通过特征字段回退
    def _is_video_stream(s: dict) -> bool:
        return s.get("codec_type") == "video" or "width" in s

    def _is_audio_stream(s: dict) -> bool:
        return s.get("codec_type") == "audio" or "sample_rate" in s

    video_stream = next((s for s in streams if _is_video_stream(s)), {})
    audio_stream = next((s for s in streams if _is_audio_stream(s)), {})

    # 容器格式: 从 format_name 列表中取最简标识
    # 按优先级选择: mp4 > mkv > webm > mov > avi > flv > ts > mxf
    _CONTAINER_PRIORITY = ["mp4", "mkv", "webm", "mov", "avi", "flv", "ts", "mxf"]
    format_names = [n.strip() for n in fmt.get("format_name", "").split(",")]
    container = "unknown"
    for preferred in _CONTAINER_PRIORITY:
        if preferred in format_names:
            container = preferred
            break
    if container == "unknown" and format_names:
        container = format_names[0]

    # 视频编码: 显示名称映射
    _VCODEC_DISPLAY = {
        "h264": "H.264",
        "hevc": "H.265/HEVC",
        "av1": "AV1",
        "vp9": "VP9",
        "mpeg4": "MPEG-4",
        "mpeg2video": "MPEG-2",
        "prores": "ProRes",
    }
    vcodec_raw = video_stream.get("codec_name", "unknown")
    vcodec_display = _VCODEC_DISPLAY.get(vcodec_raw, vcodec_raw.upper() if vcodec_raw in ("h264", "hevc", "av1") else vcodec_raw.capitalize())

    # 帧率解析 "30/1" → 30.0
    rfr = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = rfr.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    # 色彩与 HDR
    _COLOR_SPACE_MAP = {
        "bt709": "BT.709",
        "bt2020": "BT.2020",
        "bt470bg": "BT.601",
        "smpte170m": "SMPTE 170M",
    }
    color_space_raw = video_stream.get("color_space", "unknown")
    color_space = _COLOR_SPACE_MAP.get(color_space_raw, color_space_raw.upper() if color_space_raw != "unknown" else "BT.709")
    color_transfer = video_stream.get("color_transfer", "")
    hdr_info = "HDR" if color_transfer in ("smpte2084", "arib-std-b67") else "SDR"

    # 音频编码显示名称映射
    _ACODEC_DISPLAY = {
        "aac": "AAC-LC",
        "mp3": "MP3",
        "opus": "Opus",
        "vorbis": "Vorbis",
        "flac": "FLAC",
        "ac3": "AC-3",
        "eac3": "E-AC-3",
        "pcm_s16le": "PCM",
    }
    acodec_raw = audio_stream.get("codec_name", "unknown") or "unknown"
    acodec_display = _ACODEC_DISPLAY.get(acodec_raw, acodec_raw.upper())

    # 平台转码指纹
    fingerprint = _detect_platform_fingerprint(vcodec_raw, video_stream.get("profile", ""))

    # 文件大小: size 存在则使用，否则尝试从文件系统获取
    if "size" in fmt:
        file_size = int(fmt["size"])
    else:
        fname = fmt.get("filename", "")
        file_size = os.path.getsize(fname) if fname and os.path.exists(fname) else 0

    return {
        "container_format": container,
        "video_codec": vcodec_display,
        "video_profile": video_stream.get("profile", "unknown"),
        "resolution_width": video_stream.get("width", 0),
        "resolution_height": video_stream.get("height", 0),
        "frame_rate": round(fps, 2),
        "total_bitrate_bps": int(fmt.get("bit_rate", 0)),
        "video_bitrate_bps": int(video_stream.get("bit_rate", 0)),
        "audio_codec": acodec_display,
        "audio_sample_rate_hz": int(audio_stream.get("sample_rate", 0)),
        "audio_bitrate_bps": int(audio_stream.get("bit_rate", 0)),
        "gop_structure": _extract_gop_info(raw, video_stream),
        "color_space": color_space,
        "hdr_info": hdr_info,
        "duration": float(fmt.get("duration", 0)),
        "file_size_bytes": file_size,
        "platform_fingerprint": fingerprint,
    }


def _detect_platform_fingerprint(codec: str, profile: str) -> Optional[str]:
    """根据编码特征匹配平台转码指纹。"""
    codec_lower = codec.lower()
    for fp in PLATFORM_FINGERPRINTS:
        if fp["codec"] in codec_lower and fp["profile"].lower() in profile.lower():
            return f"疑似{fp['name']}二压（{fp['hint']}）"
    return None


def _extract_gop_info(raw: dict, video_stream: dict) -> str:
    """Extract GOP structure information.

    Attempts to estimate GOP size from duration and frame count when ffprobe
    does not provide explicit keyframe info.  Returns a note that full GOP
    analysis requires deeper frame-by-frame inspection.
    """
    duration = float(raw.get("format", {}).get("duration", 0))
    nb_frames = int(video_stream.get("nb_frames", 0)) or int(
        raw.get("format", {}).get("nb_frames", 0)
    )

    # If ffprobe provides has_b_frames, use it as a hint
    has_b_frames = video_stream.get("has_b_frames")

    if has_b_frames is not None:
        return f"GOP信息: has_b_frames={has_b_frames}（帧级别GOP分析需解码器支持）"

    if nb_frames > 0 and duration > 0:
        estimated_fps = nb_frames / duration
        # Assume typical GOP of 2 seconds worth of frames
        estimated_gop = max(1, int(estimated_fps * 2))
        return f"GOP信息不可用（估计约{estimated_gop}帧，需解码器精确分析）"

    return "GOP信息不可用（需解码器逐帧分析）"


class FFmpegService:
    """FFmpeg 服务外观。"""

    @staticmethod
    async def probe(file_path: str) -> dict:
        return await probe(file_path)

    @staticmethod
    def parse(raw: dict) -> dict:
        return parse_ffprobe_output(raw)

    @staticmethod
    def is_installed() -> bool:
        return check_ffprobe_installed()
