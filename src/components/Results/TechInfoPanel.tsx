/**
 * TechInfoPanel — displays technical metadata in a descriptive list.
 *
 * Shows all TechMetadata fields with Chinese labels, formatted values,
 * platform fingerprint as orange tag, and human-readable sizes/rates.
 */
import React from "react";
import { Descriptions, Tag, Typography } from "antd";
import type { TechMetadata } from "../../types/metadata";

const { Text } = Typography;

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatBitrate(bps: number): string {
  if (bps >= 1_000_000) return `${(bps / 1_000_000).toFixed(1)} Mbps`;
  if (bps >= 1_000) return `${(bps / 1_000).toFixed(0)} kbps`;
  return `${bps} bps`;
}

interface Props {
  data: TechMetadata;
}

const TechInfoPanel: React.FC<Props> = ({ data }) => {
  return (
    <Descriptions
      size="small"
      column={2}
      colon={false}
      labelStyle={{ color: "#aaa", fontSize: 12, paddingRight: 8 }}
      contentStyle={{ color: "#e8e8e8", fontSize: 12 }}
      style={{ marginTop: 4 }}
    >
      <Descriptions.Item label="封装格式">
        <Tag color="blue" style={{ margin: 0 }}>
          {data.container_format}
        </Tag>
      </Descriptions.Item>

      <Descriptions.Item label="时长">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {formatDuration(data.duration)}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="视频编码">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.video_codec}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="编码配置">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.video_profile || "-"}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="分辨率">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.resolution_width} x {data.resolution_height}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="帧率">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.frame_rate} fps
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="总比特率">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {formatBitrate(data.total_bitrate_bps)}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="视频比特率">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {formatBitrate(data.video_bitrate_bps)}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="音频编码">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.audio_codec}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="音频采样率">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.audio_sample_rate_hz} Hz
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="音频比特率">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {formatBitrate(data.audio_bitrate_bps)}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="GOP 结构">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.gop_structure || "-"}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="色彩空间">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.color_space || "-"}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="HDR 信息">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {data.hdr_info || "-"}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="文件大小">
        <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
          {formatFileSize(data.file_size_bytes)}
        </Text>
      </Descriptions.Item>

      <Descriptions.Item label="平台指纹" span={2}>
        {data.platform_fingerprint ? (
          <Tag color="orange" style={{ margin: 0 }}>
            {data.platform_fingerprint}
          </Tag>
        ) : (
          <Text style={{ color: "#888", fontSize: 12 }}>未检测到</Text>
        )}
      </Descriptions.Item>
    </Descriptions>
  );
};

export default TechInfoPanel;
