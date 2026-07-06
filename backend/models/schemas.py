"""Pydantic data models for all analysis results."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Literal
from pydantic import BaseModel, Field


class TechMetadata(BaseModel):
    """S4.1 Technical metadata read directly from FFmpeg probe."""

    container_format: str
    video_codec: str
    video_profile: str
    resolution_width: int
    resolution_height: int
    frame_rate: float
    total_bitrate_bps: int
    video_bitrate_bps: int
    audio_codec: str
    audio_sample_rate_hz: int
    audio_bitrate_bps: int
    gop_structure: str
    color_space: str
    hdr_info: str
    duration: float  # seconds
    file_size_bytes: int
    platform_fingerprint: Optional[str] = None


class ShotItem(BaseModel):
    """A single detected shot."""

    index: int
    start_time: float
    end_time: float
    duration: float
    thumbnail_path: Optional[str] = None
    is_representative: bool = False  # v1.3: representative frame marker


class ColorSummary(BaseModel):
    """Dominant color characteristics of the video."""

    dominant_hue: str  # "warm" / "cool" / "neutral"
    saturation: str  # "high" / "medium" / "low"
    description: str  # "cyan-orange" / "warm yellow" etc.


class VisualAnalysis(BaseModel):
    """S4.2 Visual analysis results."""

    shots: list[ShotItem] = Field(default_factory=list)
    shot_count: int = 0
    avg_shot_duration: float = 0.0
    transitions: list[str] = Field(default_factory=list)  # ["hard cut", "fade", ...]
    keyframe_grid_paths: list[str] = Field(default_factory=list)
    representative_frames: list[str] = Field(default_factory=list)  # v1.3: top-5 for source recovery sorting
    color_summary: Optional[ColorSummary] = None
    text_regions: list[dict] = Field(default_factory=list)
    face_detections: list[dict] = Field(default_factory=list)
    object_detections: list[dict] = Field(default_factory=list)
    motion_summary: Optional[str] = None  # "mostly static" / "slight motion" / "heavy motion"
    style_tags: list[str] = Field(default_factory=list)  # CLIP style classification results


class AudioAnalysis(BaseModel):
    """S4.3 Audio analysis results."""

    full_text: str = ""
    text_segments: list[dict] = Field(default_factory=list)  # [{text, start, end}, ...]
    speech_rate: float = 0.0  # characters/second
    speech_emotion: str = ""  # "calm" / "passionate" / ...
    bgm_title: Optional[str] = None
    bgm_artist: Optional[str] = None
    bgm_style_tags: list[str] = Field(default_factory=list)
    bgm_emotion: Optional[str] = None
    bgm_bpm: Optional[int] = None
    sound_events: list[str] = Field(default_factory=list)
    voice_to_bg_ratio: Optional[str] = None
    audio_structure: Optional[str] = None


class AIInference(BaseModel):
    """S4.4 AI inference results fallback when source recovery misses."""

    inferred_tool: Optional[str] = None  # "suspected Kling 1.6 + Topaz enhancement"
    inferred_tool_confidence: float = 0.0
    inferred_prompt: Optional[str] = None  # LLM-reversed prompt
    inferred_prompt_confidence: float = 0.0
    style_tags: list[str] = Field(default_factory=list)
    inferred_workflow: Optional[str] = None  # "text-to-video -> image-to-video refine -> Topaz upscale"
    inferred_workflow_confidence: float = 0.0
    imitation_suggestions: list[str] = Field(default_factory=list)
    model_recommendations: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.0


HitStatus = Literal["complete_match", "partial_match", "located_only", "miss"]


class SourceRecoveryHit(BaseModel):
    """S4.5 Source recovery hit result."""

    status: HitStatus
    source_url: Optional[str] = None
    similarity: float = 0.0
    hit_keyframes: int = 0
    total_keyframes_sent: int = 5
    workflow_json: Optional[str] = None  # original ComfyUI workflow API JSON
    prompt: Optional[str] = None
    seed: Optional[int] = None
    sampler: Optional[str] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    model_name: Optional[str] = None
    confidence_score: float = 0.0
    source_trust: Optional[str] = None  # "civitai" | "comfyworkflows" | "other"


ModuleStatusValue = Literal["pending", "running", "completed", "failed", "skipped"]

# Type alias for backward compatibility with tests and consumers
ModuleStatus = ModuleStatusValue


class AnalysisResult(BaseModel):
    """Aggregated result of a complete video analysis."""

    file_hash: str
    file_path: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.3.0"
    tech_metadata: TechMetadata
    visual_analysis: Optional[VisualAnalysis] = None
    audio_analysis: Optional[AudioAnalysis] = None
    ai_inference: Optional[AIInference] = None
    source_recovery: Optional[SourceRecoveryHit] = None
    module_status: dict[str, ModuleStatusValue] = Field(default_factory=dict)


class AnalysisProgress(BaseModel):
    """Progress update pushed via WebSocket."""

    file_hash: str
    module: str  # "tech" | "visual" | "audio" | "ai" | "source_recovery"
    status: ModuleStatusValue
    progress_pct: float = 0.0
    message: str = ""
    partial_result: Optional[dict] = None
