/**
 * AudioAnalysisPanel — transcribed text, timestamp segments, speech rate/emotion,
 * BGM card, sound events timeline, audio structure visualization, and voice/bg ratio.
 */
import React from "react";
import {
  Typography,
  Tag,
  Table,
  Card,
  Tooltip,
  Empty,
} from "antd";
import {
  SoundOutlined,
  CustomerServiceOutlined,
  ClockCircleOutlined,
} from "@ant-design/icons";
import type { AudioAnalysis } from "../../types/metadata";

const { Text, Title, Paragraph } = Typography;

function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Simple CSS bar visualization for audio structure */
function AudioStructureBars({
  structure,
}: {
  structure: string;
}) {
  // Parse structure string like "speech 0-10s, bgm 10-30s, silence 30-35s"
  const segments: Array<{
    type: string;
    start: number;
    end: number;
  }> = [];
  const parts = structure.split(/[,;，；]+/);
  for (const part of parts) {
    const trimmed = part.trim();
    const match = trimmed.match(
      /(\S+)\s*[:：]?\s*([\d.]+)\s*[-~到]\s*([\d.]+)\s*s?/i,
    );
    if (match) {
      segments.push({
        type: match[1].toLowerCase(),
        start: parseFloat(match[2]),
        end: parseFloat(match[3]),
      });
    }
  }

  if (segments.length === 0) {
    return (
      <Text style={{ color: "#ccc", fontSize: 12 }}>{structure}</Text>
    );
  }

  const maxEnd = Math.max(...segments.map((s) => s.end), 1);

  const typeColors: Record<string, string> = {
    speech: "#1677ff",
    voice: "#1677ff",
    bgm: "#52c41a",
    music: "#52c41a",
    silence: "#555",
    sound: "#fa8c16",
    effect: "#fa8c16",
  };

  return (
    <div style={{ marginTop: 4 }}>
      <div
        style={{
          display: "flex",
          height: 20,
          borderRadius: 4,
          overflow: "hidden",
          background: "#111",
          gap: 1,
        }}
      >
        {segments.map((seg, i) => {
          const widthPct = ((seg.end - seg.start) / maxEnd) * 100;
          const color =
            typeColors[seg.type] ??
            `hsl(${segments.length > 0 ? (i * 60) % 360 : 200}, 50%, 50%)`;
          return (
            <Tooltip
              key={i}
              title={`${seg.type} ${formatTimestamp(seg.start)}-${formatTimestamp(seg.end)}`}
            >
              <div
                style={{
                  width: `${Math.max(widthPct, 1)}%`,
                  height: "100%",
                  background: color,
                  cursor: "pointer",
                }}
              />
            </Tooltip>
          );
        })}
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 4,
        }}
      >
        <Text style={{ color: "#666", fontSize: 10 }}>0:00</Text>
        <Text style={{ color: "#666", fontSize: 10 }}>
          {formatTimestamp(maxEnd)}
        </Text>
      </div>
      <div
        style={{
          display: "flex",
          gap: 12,
          marginTop: 4,
          flexWrap: "wrap",
        }}
      >
        {Object.entries(typeColors).map(([type, color]) => {
          const hasSegment = segments.some(
            (s) => s.type === type,
          );
          if (!hasSegment) return null;
          return (
            <div
              key={type}
              style={{ display: "flex", alignItems: "center", gap: 4 }}
            >
              <span
                style={{
                  display: "inline-block",
                  width: 10,
                  height: 10,
                  borderRadius: 2,
                  background: color,
                }}
              />
              <Text style={{ color: "#888", fontSize: 10 }}>
                {type}
              </Text>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface Props {
  data: AudioAnalysis;
  onSeekTo?: (seconds: number) => void;
}

const AudioAnalysisPanel: React.FC<Props> = ({ data, onSeekTo }) => {
  const segmentColumns = [
    {
      title: "时间",
      dataIndex: "start",
      key: "time",
      width: 80,
      render: (_: unknown, record: { start: number; end: number }) => (
        <Text style={{ color: "#ccc", fontSize: 11 }}>
          {formatTimestamp(record.start)} - {formatTimestamp(record.end)}
        </Text>
      ),
    },
    {
      title: "文本",
      dataIndex: "text",
      key: "text",
      render: (text: string, record: { start: number }) => (
        <Text
          style={{
            color: "#e8e8e8",
            fontSize: 11,
            cursor: onSeekTo ? "pointer" : "default",
          }}
          onClick={() => onSeekTo?.(record.start)}
        >
          {text}
        </Text>
      ),
    },
  ];

  const hasBgmInfo =
    data.bgm_title ||
    data.bgm_artist ||
    data.bgm_bpm != null ||
    data.bgm_emotion ||
    data.bgm_style_tags.length > 0;

  return (
    <div>
      {/* Full text */}
      {data.full_text && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text style={{ color: "#aaa", fontSize: 12 }}>完整文本</Text>
            <div style={{ display: "flex", gap: 12 }}>
              <Text style={{ color: "#888", fontSize: 11 }}>
                语速: {data.speech_rate.toFixed(1)} 字/秒
              </Text>
              {data.speech_emotion && (
                <Tag color="blue" style={{ margin: 0, fontSize: 10 }}>
                  {data.speech_emotion}
                </Tag>
              )}
            </div>
          </div>
          <Paragraph
            style={{
              color: "#ccc",
              fontSize: 12,
              margin: "8px 0 0 0",
              background: "rgba(255,255,255,0.04)",
              padding: "8px 10px",
              borderRadius: 4,
              maxHeight: 150,
              overflowY: "auto",
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
            }}
          >
            {data.full_text}
          </Paragraph>
        </div>
      )}

      {/* Text segments table */}
      {data.text_segments.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>
            文本分段 ({data.text_segments.length} 段)
          </Text>
          <Table
            columns={segmentColumns}
            dataSource={data.text_segments.map((seg, i) => ({
              ...seg,
              key: i,
            }))}
            size="small"
            pagination={false}
            scroll={{ y: 160 }}
            style={{ marginTop: 4 }}
            rowClassName={() => "audio-segment-row"}
            locale={{ emptyText: "暂无分段数据" }}
          />
        </div>
      )}

      {/* BGM card */}
      {hasBgmInfo && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>背景音乐</Text>
          <Card
            size="small"
            style={{
              marginTop: 4,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid #303030",
            }}
            bodyStyle={{ padding: 10 }}
          >
            {data.bgm_title && (
              <div style={{ marginBottom: 4 }}>
                <CustomerServiceOutlined
                  style={{ color: "#52c41a", marginRight: 6, fontSize: 12 }}
                />
                <Text style={{ color: "#e8e8e8", fontSize: 13, fontWeight: 500 }}>
                  {data.bgm_title}
                </Text>
              </div>
            )}
            {data.bgm_artist && (
              <div style={{ marginBottom: 4 }}>
                <Text style={{ color: "#888", fontSize: 11 }}>
                  艺术家: {data.bgm_artist}
                </Text>
              </div>
            )}
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 4 }}>
              {data.bgm_bpm != null && (
                <Text style={{ color: "#888", fontSize: 11 }}>
                  BPM: {data.bgm_bpm}
                </Text>
              )}
              {data.bgm_emotion && (
                <Tag color="green" style={{ margin: 0, fontSize: 10 }}>
                  {data.bgm_emotion}
                </Tag>
              )}
              {data.bgm_style_tags.map((tag, i) => (
                <Tag key={i} color="cyan" style={{ margin: 0, fontSize: 10 }}>
                  {tag}
                </Tag>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Sound events */}
      {data.sound_events.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>声音事件</Text>
          <div style={{ marginTop: 4, display: "flex", gap: 4, flexWrap: "wrap" }}>
            {data.sound_events.map((event, i) => (
              <Tag
                key={i}
                color="volcano"
                style={{ margin: 0, fontSize: 11 }}
                icon={<SoundOutlined />}
              >
                {event}
              </Tag>
            ))}
          </div>
        </div>
      )}

      {/* Audio structure */}
      {data.audio_structure && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>音频结构</Text>
          <AudioStructureBars structure={data.audio_structure} />
        </div>
      )}

      {/* Voice/background ratio */}
      {data.voice_to_bg_ratio && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>人声/背景比例</Text>
          <Paragraph
            style={{
              color: "#ccc",
              fontSize: 12,
              margin: "4px 0 0 0",
              background: "rgba(255,255,255,0.04)",
              padding: "6px 8px",
              borderRadius: 4,
            }}
          >
            {data.voice_to_bg_ratio}
          </Paragraph>
        </div>
      )}

      {/* Empty state */}
      {!data.full_text &&
        data.text_segments.length === 0 &&
        !hasBgmInfo &&
        data.sound_events.length === 0 &&
        !data.audio_structure &&
        !data.voice_to_bg_ratio && (
          <Empty
            description={
              <Text style={{ color: "#666" }}>暂无音频分析数据</Text>
            }
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        )}
    </div>
  );
};

export default AudioAnalysisPanel;
