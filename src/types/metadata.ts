/** TypeScript types mirroring backend Pydantic models (backend/models/schemas.py). */

// ---- S4.1 Technical Metadata ----
export interface TechMetadata {
  container_format: string;
  video_codec: string;
  video_profile: string;
  resolution_width: number;
  resolution_height: number;
  frame_rate: number;
  total_bitrate_bps: number;
  video_bitrate_bps: number;
  audio_codec: string;
  audio_sample_rate_hz: number;
  audio_bitrate_bps: number;
  gop_structure: string;
  color_space: string;
  hdr_info: string;
  duration: number; // seconds
  file_size_bytes: number;
  platform_fingerprint: string | null;
}

// ---- S4.2 Visual Analysis ----
export interface ShotItem {
  index: number;
  start_time: number;
  end_time: number;
  duration: number;
  thumbnail_path: string | null;
  is_representative: boolean; // v1.3: representative frame marker
}

export interface ColorSummary {
  dominant_hue: string; // "warm" | "cool" | "neutral"
  saturation: string; // "high" | "medium" | "low"
  description: string; // "cyan-orange" | "warm yellow" etc.
}

export interface VisualAnalysis {
  shots: ShotItem[];
  shot_count: number;
  avg_shot_duration: number;
  transitions: string[]; // ["hard cut", "fade", ...]
  keyframe_grid_paths: string[];
  representative_frames: string[]; // v1.3: top-5 for source recovery sorting
  color_summary: ColorSummary | null;
  text_regions: Record<string, unknown>[];
  face_detections: Record<string, unknown>[];
  object_detections: Record<string, unknown>[];
  motion_summary: string | null; // "mostly static" | "slight motion" | "heavy motion"
}

// ---- S4.3 Audio Analysis ----
export interface AudioAnalysis {
  full_text: string;
  text_segments: Array<{ text: string; start: number; end: number }>;
  speech_rate: number; // characters/second
  speech_emotion: string; // "calm" | "passionate" | ...
  bgm_title: string | null;
  bgm_artist: string | null;
  bgm_style_tags: string[];
  bgm_emotion: string | null;
  bgm_bpm: number | null;
  sound_events: string[];
  voice_to_bg_ratio: string | null;
  audio_structure: string | null;
}

// ---- S4.4 AI Inference ----
export interface AIInference {
  inferred_tool: string | null; // "suspected Kling 1.6 + Topaz enhancement"
  inferred_tool_confidence: number;
  inferred_prompt: string | null; // LLM-reversed prompt
  inferred_prompt_confidence: number;
  style_tags: string[];
  inferred_workflow: string | null; // "text-to-video -> image-to-video refine -> Topaz upscale"
  inferred_workflow_confidence: number;
  imitation_suggestions: string[];
  model_recommendations: string[];
  overall_confidence: number;
}

// ---- S4.5 Source Recovery ----
export type HitStatus = "complete_match" | "partial_match" | "located_only" | "miss";

export interface SourceRecoveryHit {
  status: HitStatus;
  source_url: string | null;
  similarity: number;
  hit_keyframes: number;
  total_keyframes_sent: number;
  workflow_json: string | null; // original ComfyUI workflow API JSON
  prompt: string | null;
  seed: number | null;
  sampler: string | null;
  steps: number | null;
  cfg_scale: number | null;
  model_name: string | null;
  confidence_score: number;
  source_trust: string | null; // "civitai" | "comfyworkflows" | "other"
}

// ---- Aggregated Result ----
export type ModuleStatusValue = "pending" | "running" | "completed" | "failed" | "skipped";

export interface AnalysisResult {
  file_hash: string;
  file_path: string;
  analyzed_at: string; // ISO 8601 datetime
  schema_version: string;
  tech_metadata: TechMetadata;
  visual_analysis: VisualAnalysis | null;
  audio_analysis: AudioAnalysis | null;
  ai_inference: AIInference | null;
  source_recovery: SourceRecoveryHit | null;
  module_status: Record<string, ModuleStatusValue>;
}

// ---- WebSocket Progress Message ----
export interface AnalysisProgress {
  file_hash: string;
  module: string; // "tech" | "visual" | "audio" | "ai" | "source_recovery"
  status: ModuleStatusValue;
  progress_pct: number;
  message: string;
  partial_result: Record<string, unknown> | null;
}

// ---- Module Labels (UI display names) ----
export const MODULE_LABELS: Record<string, string> = {
  tech: "技术提取",
  visual: "视觉分析",
  audio: "音频分析",
  ai: "AI 推断",
  source_recovery: "源回捞",
};

export const MODULE_KEYS = ["tech", "visual", "audio", "ai", "source_recovery"] as const;
export type ModuleKey = (typeof MODULE_KEYS)[number];

// ---- LLM Provider ----
export const LLM_PROVIDERS = ["claude", "openai", "gemini", "qwen", "custom"] as const;
export type LLMProvider = (typeof LLM_PROVIDERS)[number];

export const LLM_PROVIDER_LABELS: Record<LLMProvider, string> = {
  claude: "Claude (Anthropic)",
  openai: "GPT-4V / GPT-4o (OpenAI)",
  gemini: "Gemini 2.0 Flash (Google)",
  qwen: "Qwen-VL 通义千问 (阿里百炼)",
  custom: "自定义端点 (OpenAI 兼容)",
};

// ---- App Settings ----
export interface AppSettings {
  llm_api_key: string;
  llm_provider: LLMProvider;
  llm_custom_endpoint: string;
  llm_custom_model: string;
  saucenao_api_key: string;
  acrcloud_key: string;
  acrcloud_secret: string;
  source_recovery_consent: boolean;
  theme: "dark" | "light";
}
