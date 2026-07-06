/**
 * CompareView — side-by-side comparison of two AnalysisResults.
 *
 * Opens as a large Ant Design Modal (90vw x 90vh). Each side shows the
 * video filename and collapsible sections (Tech, Visual, Audio, AI,
 * Source Recovery). Values that differ between the two videos are
 * highlighted with a yellow background.
 */
import React, { useMemo, useState } from "react";
import { Modal, Collapse, Button, Typography, Descriptions, Tag, Empty, Select, Space } from "antd";
import { CloseOutlined, FileOutlined, SwapOutlined } from "@ant-design/icons";
import { useAnalysisStore } from "../../store/analysisStore";
import { api } from "../../services/api";
import type { AnalysisResult, TechMetadata } from "../../types/metadata";
import { MODULE_LABELS } from "../../types/metadata";

const { Text, Title } = Typography;

// ---- Diff helpers ----

/**
 * Returns a background-color style when left !== right.
 * Otherwise returns an empty object.
 */
function diffStyle<T>(left: T, right: T): React.CSSProperties {
  if (left !== right) {
    return { backgroundColor: "rgba(250, 219, 20, 0.15)", padding: "2px 4px", borderRadius: 3 };
  }
  return {};
}

function diffStyleDeep<T>(left: T, right: T): React.CSSProperties {
  const leftStr = JSON.stringify(left);
  const rightStr = JSON.stringify(right);
  if (leftStr !== rightStr) {
    return { backgroundColor: "rgba(250, 219, 20, 0.15)", padding: "2px 4px", borderRadius: 3 };
  }
  return {};
}

// ---- Format helpers ----

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatBitrate(bps: number): string {
  if (bps >= 1_000_000) return `${(bps / 1_000_000).toFixed(1)} Mbps`;
  if (bps >= 1_000) return `${(bps / 1_000).toFixed(0)} kbps`;
  return `${bps} bps`;
}

function fileNameFromPath(filePath: string): string {
  const parts = filePath.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1];
}

// ---- Diff-aware TechMetadata table ----

function TechCompareTable({ left, right }: { left: TechMetadata; right: TechMetadata }) {
  const rows: Array<{ label: string; left: React.ReactNode; right: React.ReactNode }> = [
    {
      label: "封装格式",
      left: <Tag color="blue" style={{ margin: 0 }}>{left.container_format}</Tag>,
      right: <Tag color="blue" style={{ margin: 0 }}>{right.container_format}</Tag>,
    },
    {
      label: "时长",
      left: <span style={diffStyle(left.duration, right.duration)}>{formatDuration(left.duration)}</span>,
      right: <span style={diffStyle(right.duration, left.duration)}>{formatDuration(right.duration)}</span>,
    },
    {
      label: "视频编码",
      left: <span style={diffStyle(left.video_codec, right.video_codec)}>{left.video_codec}</span>,
      right: <span style={diffStyle(right.video_codec, left.video_codec)}>{right.video_codec}</span>,
    },
    {
      label: "编码配置",
      left: <span style={diffStyle(left.video_profile, right.video_profile)}>{left.video_profile || "-"}</span>,
      right: <span style={diffStyle(right.video_profile, left.video_profile)}>{right.video_profile || "-"}</span>,
    },
    {
      label: "分辨率",
      left: (
        <span style={diffStyle(`${left.resolution_width}x${left.resolution_height}`, `${right.resolution_width}x${right.resolution_height}`)}>
          {left.resolution_width} x {left.resolution_height}
        </span>
      ),
      right: (
        <span style={diffStyle(`${right.resolution_width}x${right.resolution_height}`, `${left.resolution_width}x${left.resolution_height}`)}>
          {right.resolution_width} x {right.resolution_height}
        </span>
      ),
    },
    {
      label: "帧率",
      left: <span style={diffStyle(left.frame_rate, right.frame_rate)}>{left.frame_rate} fps</span>,
      right: <span style={diffStyle(right.frame_rate, left.frame_rate)}>{right.frame_rate} fps</span>,
    },
    {
      label: "总比特率",
      left: <span style={diffStyle(left.total_bitrate_bps, right.total_bitrate_bps)}>{formatBitrate(left.total_bitrate_bps)}</span>,
      right: <span style={diffStyle(right.total_bitrate_bps, left.total_bitrate_bps)}>{formatBitrate(right.total_bitrate_bps)}</span>,
    },
    {
      label: "视频比特率",
      left: <span style={diffStyle(left.video_bitrate_bps, right.video_bitrate_bps)}>{formatBitrate(left.video_bitrate_bps)}</span>,
      right: <span style={diffStyle(right.video_bitrate_bps, left.video_bitrate_bps)}>{formatBitrate(right.video_bitrate_bps)}</span>,
    },
    {
      label: "音频编码",
      left: <span style={diffStyle(left.audio_codec, right.audio_codec)}>{left.audio_codec}</span>,
      right: <span style={diffStyle(right.audio_codec, left.audio_codec)}>{right.audio_codec}</span>,
    },
    {
      label: "音频采样率",
      left: <span style={diffStyle(left.audio_sample_rate_hz, right.audio_sample_rate_hz)}>{left.audio_sample_rate_hz} Hz</span>,
      right: <span style={diffStyle(right.audio_sample_rate_hz, left.audio_sample_rate_hz)}>{right.audio_sample_rate_hz} Hz</span>,
    },
    {
      label: "音频比特率",
      left: <span style={diffStyle(left.audio_bitrate_bps, right.audio_bitrate_bps)}>{formatBitrate(left.audio_bitrate_bps)}</span>,
      right: <span style={diffStyle(right.audio_bitrate_bps, left.audio_bitrate_bps)}>{formatBitrate(right.audio_bitrate_bps)}</span>,
    },
    {
      label: "GOP 结构",
      left: <span style={diffStyle(left.gop_structure, right.gop_structure)}>{left.gop_structure || "-"}</span>,
      right: <span style={diffStyle(right.gop_structure, left.gop_structure)}>{right.gop_structure || "-"}</span>,
    },
    {
      label: "色彩空间",
      left: <span style={diffStyle(left.color_space, right.color_space)}>{left.color_space || "-"}</span>,
      right: <span style={diffStyle(right.color_space, left.color_space)}>{right.color_space || "-"}</span>,
    },
    {
      label: "HDR 信息",
      left: <span style={diffStyle(left.hdr_info, right.hdr_info)}>{left.hdr_info || "-"}</span>,
      right: <span style={diffStyle(right.hdr_info, left.hdr_info)}>{right.hdr_info || "-"}</span>,
    },
    {
      label: "文件大小",
      left: <span style={diffStyle(left.file_size_bytes, right.file_size_bytes)}>{formatFileSize(left.file_size_bytes)}</span>,
      right: <span style={diffStyle(right.file_size_bytes, left.file_size_bytes)}>{formatFileSize(right.file_size_bytes)}</span>,
    },
    {
      label: "平台指纹",
      left: (
        <span style={diffStyle(left.platform_fingerprint, right.platform_fingerprint)}>
          {left.platform_fingerprint ? (
            <Tag color="orange" style={{ margin: 0 }}>{left.platform_fingerprint}</Tag>
          ) : (
            <Text style={{ fontSize: 12, color: "#888" }}>未检测到</Text>
          )}
        </span>
      ),
      right: (
        <span style={diffStyle(right.platform_fingerprint, left.platform_fingerprint)}>
          {right.platform_fingerprint ? (
            <Tag color="orange" style={{ margin: 0 }}>{right.platform_fingerprint}</Tag>
          ) : (
            <Text style={{ fontSize: 12, color: "#888" }}>未检测到</Text>
          )}
        </span>
      ),
    },
  ];

  return (
    <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
      <colgroup>
        <col style={{ width: "30%" }} />
        <col style={{ width: "35%" }} />
        <col style={{ width: "35%" }} />
      </colgroup>
      <thead>
        <tr style={{ borderBottom: "1px solid #303030" }}>
          <th style={{ color: "#aaa", fontWeight: 500, padding: "4px 8px", textAlign: "left" }}>字段</th>
          <th style={{ color: "#aaa", fontWeight: 500, padding: "4px 8px", textAlign: "left" }}>视频 A</th>
          <th style={{ color: "#aaa", fontWeight: 500, padding: "4px 8px", textAlign: "left" }}>视频 B</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label} style={{ borderBottom: "1px solid #1f1f1f" }}>
            <td style={{ color: "#aaa", padding: "6px 8px" }}>{row.label}</td>
            <td style={{ color: "#e8e8e8", padding: "6px 8px" }}>{row.left}</td>
            <td style={{ color: "#e8e8e8", padding: "6px 8px" }}>{row.right}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---- Visual analysis compare ----

function VisualCompareTable({
  left,
  right,
}: {
  left: AnalysisResult["visual_analysis"];
  right: AnalysisResult["visual_analysis"];
}) {
  if (!left && !right) {
    return <Text style={{ color: "#888", fontSize: 12 }}>双方均无视觉分析数据</Text>;
  }

  const leftShots = left?.shot_count ?? 0;
  const rightShots = right?.shot_count ?? 0;
  const leftAvg = left?.avg_shot_duration ?? 0;
  const rightAvg = right?.avg_shot_duration ?? 0;
  const leftMotion = left?.motion_summary ?? null;
  const rightMotion = right?.motion_summary ?? null;
  const leftColor = left?.color_summary?.description ?? null;
  const rightColor = right?.color_summary?.description ?? null;

  return (
    <Descriptions
      size="small"
      column={1}
      colon={false}
      labelStyle={{ color: "#aaa", fontSize: 12, paddingRight: 8 }}
      contentStyle={{ color: "#e8e8e8", fontSize: 12 }}
    >
      <Descriptions.Item label="镜头数">
        <span style={{ ...diffStyle(leftShots, rightShots), marginRight: 4 }}>
          A: {leftShots}
        </span>
        <span style={diffStyle(rightShots, leftShots)}>B: {rightShots}</span>
      </Descriptions.Item>
      <Descriptions.Item label="平均镜头时长">
        <span style={{ ...diffStyle(leftAvg, rightAvg), marginRight: 4 }}>
          A: {leftAvg.toFixed(1)}s
        </span>
        <span style={diffStyle(rightAvg, leftAvg)}>B: {rightAvg.toFixed(1)}s</span>
      </Descriptions.Item>
      <Descriptions.Item label="色彩基调">
        <span style={{ ...diffStyle(leftColor, rightColor), marginRight: 4 }}>
          A: {leftColor ?? "无"}
        </span>
        <span style={diffStyle(rightColor, leftColor)}>B: {rightColor ?? "无"}</span>
      </Descriptions.Item>
      <Descriptions.Item label="运动幅度">
        <span style={{ ...diffStyle(leftMotion, rightMotion), marginRight: 4 }}>
          A: {leftMotion ?? "无"}
        </span>
        <span style={diffStyle(rightMotion, leftMotion)}>B: {rightMotion ?? "无"}</span>
      </Descriptions.Item>
      <Descriptions.Item label="转场类型">
        <span style={{ ...diffStyleDeep(left?.transitions, right?.transitions), marginRight: 4 }}>
          A: {left?.transitions?.join(", ") ?? "无"}
        </span>
        <span style={diffStyleDeep(right?.transitions, left?.transitions)}>
          B: {right?.transitions?.join(", ") ?? "无"}
        </span>
      </Descriptions.Item>
    </Descriptions>
  );
}

// ---- Audio analysis compare ----

function AudioCompareTable({
  left,
  right,
}: {
  left: AnalysisResult["audio_analysis"];
  right: AnalysisResult["audio_analysis"];
}) {
  if (!left && !right) {
    return <Text style={{ color: "#888", fontSize: 12 }}>双方均无音频分析数据</Text>;
  }

  const leftRate = left?.speech_rate ?? 0;
  const rightRate = right?.speech_rate ?? 0;
  const leftEmotion = left?.speech_emotion ?? null;
  const rightEmotion = right?.speech_emotion ?? null;
  const leftBgmEmotion = left?.bgm_emotion ?? null;
  const rightBgmEmotion = right?.bgm_emotion ?? null;
  const leftBpm = left?.bgm_bpm ?? null;
  const rightBpm = right?.bgm_bpm ?? null;
  const leftRatio = left?.voice_to_bg_ratio ?? null;
  const rightRatio = right?.voice_to_bg_ratio ?? null;

  return (
    <Descriptions
      size="small"
      column={1}
      colon={false}
      labelStyle={{ color: "#aaa", fontSize: 12, paddingRight: 8 }}
      contentStyle={{ color: "#e8e8e8", fontSize: 12 }}
    >
      <Descriptions.Item label="语速">
        <span style={{ ...diffStyle(leftRate, rightRate), marginRight: 4 }}>
          A: {leftRate.toFixed(1)} 字/秒
        </span>
        <span style={diffStyle(rightRate, leftRate)}>B: {rightRate.toFixed(1)} 字/秒</span>
      </Descriptions.Item>
      <Descriptions.Item label="语音情绪">
        <span style={{ ...diffStyle(leftEmotion, rightEmotion), marginRight: 4 }}>
          A: {leftEmotion ?? "无"}
        </span>
        <span style={diffStyle(rightEmotion, leftEmotion)}>B: {rightEmotion ?? "无"}</span>
      </Descriptions.Item>
      <Descriptions.Item label="BGM 情绪">
        <span style={{ ...diffStyle(leftBgmEmotion, rightBgmEmotion), marginRight: 4 }}>
          A: {leftBgmEmotion ?? "无"}
        </span>
        <span style={diffStyle(rightBgmEmotion, leftBgmEmotion)}>B: {rightBgmEmotion ?? "无"}</span>
      </Descriptions.Item>
      <Descriptions.Item label="BPM">
        <span style={{ ...diffStyle(leftBpm, rightBpm), marginRight: 4 }}>
          A: {leftBpm != null ? leftBpm : "无"}
        </span>
        <span style={diffStyle(rightBpm, leftBpm)}>B: {rightBpm != null ? rightBpm : "无"}</span>
      </Descriptions.Item>
      <Descriptions.Item label="人声/背景比">
        <span style={{ ...diffStyle(leftRatio, rightRatio), marginRight: 4 }}>
          A: {leftRatio ?? "无"}
        </span>
        <span style={diffStyle(rightRatio, leftRatio)}>B: {rightRatio ?? "无"}</span>
      </Descriptions.Item>
    </Descriptions>
  );
}

// ---- AI inference compare ----

function AICompareTable({
  left,
  right,
}: {
  left: AnalysisResult["ai_inference"];
  right: AnalysisResult["ai_inference"];
}) {
  if (!left && !right) {
    return <Text style={{ color: "#888", fontSize: 12 }}>双方均无 AI 推断数据</Text>;
  }

  const leftTool = left?.inferred_tool ?? null;
  const rightTool = right?.inferred_tool ?? null;
  const leftConf = left?.overall_confidence ?? 0;
  const rightConf = right?.overall_confidence ?? 0;
  const leftTags = left?.style_tags ?? [];
  const rightTags = right?.style_tags ?? [];

  return (
    <Descriptions
      size="small"
      column={1}
      colon={false}
      labelStyle={{ color: "#aaa", fontSize: 12, paddingRight: 8 }}
      contentStyle={{ color: "#e8e8e8", fontSize: 12 }}
    >
      <Descriptions.Item label="推断工具">
        <span style={{ ...diffStyle(leftTool, rightTool), marginRight: 4 }}>
          A: {leftTool ?? "无"}
        </span>
        <span style={diffStyle(rightTool, leftTool)}>B: {rightTool ?? "无"}</span>
      </Descriptions.Item>
      <Descriptions.Item label="置信度">
        <span style={{ ...diffStyle(leftConf, rightConf), marginRight: 4 }}>
          A: {(leftConf * 100).toFixed(0)}%
        </span>
        <span style={diffStyle(rightConf, leftConf)}>B: {(rightConf * 100).toFixed(0)}%</span>
      </Descriptions.Item>
      <Descriptions.Item label="风格标签">
        <span style={{ ...diffStyleDeep(leftTags, rightTags), marginRight: 4 }}>
          A: {leftTags.length > 0 ? leftTags.join(", ") : "无"}
        </span>
        <span style={diffStyleDeep(rightTags, leftTags)}>
          B: {rightTags.length > 0 ? rightTags.join(", ") : "无"}
        </span>
      </Descriptions.Item>
      <Descriptions.Item label="推断 Prompt">
        <span style={{ ...diffStyle(left?.inferred_prompt, right?.inferred_prompt), marginRight: 4 }}>
          A: {left?.inferred_prompt ? <Tag color="blue">有</Tag> : <Tag>无</Tag>}
        </span>
        <span style={diffStyle(right?.inferred_prompt, left?.inferred_prompt)}>
          B: {right?.inferred_prompt ? <Tag color="blue">有</Tag> : <Tag>无</Tag>}
        </span>
      </Descriptions.Item>
    </Descriptions>
  );
}

// ---- Source recovery compare ----

function SourceCompareTable({
  left,
  right,
}: {
  left: AnalysisResult["source_recovery"];
  right: AnalysisResult["source_recovery"];
}) {
  if (!left && !right) {
    return <Text style={{ color: "#888", fontSize: 12 }}>双方均无源回捞数据</Text>;
  }

  const leftStatus = left?.status ?? "miss";
  const rightStatus = right?.status ?? "miss";
  const leftModel = left?.model_name ?? null;
  const rightModel = right?.model_name ?? null;
  const leftSeed = left?.seed ?? null;
  const rightSeed = right?.seed ?? null;
  const leftSampler = left?.sampler ?? null;
  const rightSampler = right?.sampler ?? null;

  return (
    <Descriptions
      size="small"
      column={1}
      colon={false}
      labelStyle={{ color: "#aaa", fontSize: 12, paddingRight: 8 }}
      contentStyle={{ color: "#e8e8e8", fontSize: 12 }}
    >
      <Descriptions.Item label="匹配状态">
        <span style={{ ...diffStyle(leftStatus, rightStatus), marginRight: 4 }}>
          A: {statusLabel(leftStatus)}
        </span>
        <span style={diffStyle(rightStatus, leftStatus)}>B: {statusLabel(rightStatus)}</span>
      </Descriptions.Item>
      <Descriptions.Item label="模型">
        <span style={{ ...diffStyle(leftModel, rightModel), marginRight: 4 }}>
          A: {leftModel ?? "-"}
        </span>
        <span style={diffStyle(rightModel, leftModel)}>B: {rightModel ?? "-"}</span>
      </Descriptions.Item>
      <Descriptions.Item label="Seed">
        <span style={{ ...diffStyle(leftSeed, rightSeed), marginRight: 4 }}>
          A: {leftSeed != null ? leftSeed : "-"}
        </span>
        <span style={diffStyle(rightSeed, leftSeed)}>B: {rightSeed != null ? rightSeed : "-"}</span>
      </Descriptions.Item>
      <Descriptions.Item label="采样器">
        <span style={{ ...diffStyle(leftSampler, rightSampler), marginRight: 4 }}>
          A: {leftSampler ?? "-"}
        </span>
        <span style={diffStyle(rightSampler, leftSampler)}>B: {rightSampler ?? "-"}</span>
      </Descriptions.Item>
    </Descriptions>
  );
}

function statusLabel(status: string): React.ReactNode {
  switch (status) {
    case "complete_match":
      return <Tag color="success">完整匹配</Tag>;
    case "partial_match":
      return <Tag color="warning">部分匹配</Tag>;
    case "located_only":
      return <Tag color="orange">仅定位</Tag>;
    default:
      return <Tag>未命中</Tag>;
  }
}

// ---- Render one side's collapsible panels ----

interface SidePanelsProps {
  result: AnalysisResult;
  otherResult: AnalysisResult;
}

function SidePanels({ result, otherResult }: SidePanelsProps) {
  const panelItems = useMemo(
    () => [
      {
        key: "tech",
        label: <Text style={{ color: "#e8e8e8", fontSize: 13 }}>{MODULE_LABELS.tech}</Text>,
        children: (
          <TechCompareTable left={result.tech_metadata} right={otherResult.tech_metadata} />
        ),
      },
      {
        key: "visual",
        label: <Text style={{ color: "#e8e8e8", fontSize: 13 }}>{MODULE_LABELS.visual}</Text>,
        children: (
          <VisualCompareTable
            left={result.visual_analysis}
            right={otherResult.visual_analysis}
          />
        ),
      },
      {
        key: "audio",
        label: <Text style={{ color: "#e8e8e8", fontSize: 13 }}>{MODULE_LABELS.audio}</Text>,
        children: (
          <AudioCompareTable
            left={result.audio_analysis}
            right={otherResult.audio_analysis}
          />
        ),
      },
      {
        key: "ai",
        label: <Text style={{ color: "#e8e8e8", fontSize: 13 }}>{MODULE_LABELS.ai}</Text>,
        children: (
          <AICompareTable
            left={result.ai_inference}
            right={otherResult.ai_inference}
          />
        ),
      },
      {
        key: "source_recovery",
        label: <Text style={{ color: "#e8e8e8", fontSize: 13 }}>{MODULE_LABELS.source_recovery}</Text>,
        children: (
          <SourceCompareTable
            left={result.source_recovery}
            right={otherResult.source_recovery}
          />
        ),
      },
    ],
    [result, otherResult],
  );

  return (
    <Collapse
      ghost
      defaultActiveKey={["tech"]}
      style={{ background: "transparent" }}
      items={panelItems.map((item) => ({
        ...item,
        style: {
          marginBottom: 4,
          background: "rgba(255,255,255,0.03)",
          borderRadius: 6,
          border: "1px solid #1f1f1f",
        },
      }))}
    />
  );
}

// ---- Main CompareView component ----

const CompareView: React.FC = () => {
  const compareOpen = useAnalysisStore((s) => s.compareOpen);
  const leftResult = useAnalysisStore((s) => s.leftResult);
  const rightResult = useAnalysisStore((s) => s.rightResult);
  const setCompareOpen = useAnalysisStore((s) => s.setCompareOpen);
  const setLeftResult = useAnalysisStore((s) => s.setLeftResult);
  const setRightResult = useAnalysisStore((s) => s.setRightResult);
  const history = useAnalysisStore((s) => s.history);

  const [leftHash, setLeftHash] = useState<string | undefined>(undefined);
  const [rightHash, setRightHash] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);

  const handleClose = () => {
    setCompareOpen(false);
    setLeftResult(null);
    setRightResult(null);
    setLeftHash(undefined);
    setRightHash(undefined);
  };

  const handleLoad = async () => {
    if (!leftHash || !rightHash) return;
    setLoading(true);
    try {
      const [left, right] = await Promise.all([
        api.getCachedResult(leftHash),
        api.getCachedResult(rightHash),
      ]);
      if (left) setLeftResult(left);
      if (right) setRightResult(right);
    } catch {
      // ignore errors
    } finally {
      setLoading(false);
    }
  };

  // Build select options from history entries
  const options = history.map((entry) => ({
    value: entry.hash,
    label: `${entry.filePath.split(/[\\/]/).pop() ?? entry.filePath} (${entry.hash.slice(0, 8)}...)`,
  }));

  return (
    <Modal
      title={
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <FileOutlined style={{ fontSize: 16, color: "#1677ff" }} />
          <span style={{ color: "#fff" }}>视频对比</span>
        </div>
      }
      open={compareOpen}
      onCancel={handleClose}
      width="90vw"
      style={{ top: "5vh", maxWidth: "90vw" }}
      styles={{
        body: {
          height: "calc(90vh - 110px)",
          padding: 12,
          overflow: "auto",
        },
      }}
      footer={
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <Button onClick={handleClose} icon={<CloseOutlined />}>
            关闭
          </Button>
        </div>
      }
      destroyOnClose
    >
      {(!leftResult || !rightResult) ? (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            gap: 16,
          }}
        >
          <SwapOutlined style={{ fontSize: 32, color: "#1677ff" }} />
          <Text style={{ color: "#e8e8e8", fontSize: 14 }}>
            选择两个已分析过的视频进行对比
          </Text>

          <Space direction="vertical" style={{ width: 360 }}>
            <div>
              <Text style={{ color: "#aaa", fontSize: 12, display: "block", marginBottom: 4 }}>
                视频 A
              </Text>
              <Select
                showSearch
                placeholder="选择视频 A..."
                value={leftHash}
                onChange={(val) => setLeftHash(val)}
                options={options}
                style={{ width: "100%" }}
                filterOption={(input, option) =>
                  (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
                }
              />
            </div>

            <div>
              <Text style={{ color: "#aaa", fontSize: 12, display: "block", marginBottom: 4 }}>
                视频 B
              </Text>
              <Select
                showSearch
                placeholder="选择视频 B..."
                value={rightHash}
                onChange={(val) => setRightHash(val)}
                options={options}
                style={{ width: "100%" }}
                filterOption={(input, option) =>
                  (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
                }
              />
            </div>

            <Button
              type="primary"
              block
              loading={loading}
              disabled={!leftHash || !rightHash}
              onClick={handleLoad}
            >
              开始对比
            </Button>
          </Space>
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 8px 1fr",
            height: "100%",
            gap: 0,
          }}
        >
          {/* Left column: Video A */}
          <div
            style={{
              overflow: "auto",
              paddingRight: 4,
              borderRight: "1px solid #303030",
            }}
          >
            <Title level={5} style={{ color: "#1677ff", marginBottom: 8, fontSize: 14 }}>
              视频 A
            </Title>
            <Text
              style={{
                color: "#aaa",
                fontSize: 12,
                display: "block",
                marginBottom: 12,
                wordBreak: "break-all",
              }}
            >
              {fileNameFromPath(leftResult.file_path)}
            </Text>
            <SidePanels result={leftResult} otherResult={rightResult} />
          </div>

          {/* Divider */}
          <div style={{ width: 8 }} />

          {/* Right column: Video B */}
          <div style={{ overflow: "auto", paddingLeft: 4 }}>
            <Title level={5} style={{ color: "#52c41a", marginBottom: 8, fontSize: 14 }}>
              视频 B
            </Title>
            <Text
              style={{
                color: "#aaa",
                fontSize: 12,
                display: "block",
                marginBottom: 12,
                wordBreak: "break-all",
              }}
            >
              {fileNameFromPath(rightResult.file_path)}
            </Text>
            <SidePanels result={rightResult} otherResult={leftResult} />
          </div>
        </div>
      )}
    </Modal>
  );
};

export default CompareView;
