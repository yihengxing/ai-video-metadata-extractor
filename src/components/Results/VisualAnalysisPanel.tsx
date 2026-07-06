/**
 * VisualAnalysisPanel — shot breakdown, representative frames, color swatches,
 * object detection tags, motion summary, and face detection summary.
 */
import React from "react";
import { Typography, Tag, Image, Card, Tooltip, Empty } from "antd";
import {
  StarFilled,
  PictureOutlined,
} from "@ant-design/icons";
import type { VisualAnalysis } from "../../types/metadata";

const { Text, Title, Paragraph } = Typography;

function formatDuration(seconds: number): string {
  return `${seconds.toFixed(1)}s`;
}

/** Small color swatch chip */
function ColorSwatch({ color, label }: { color: string; label: string }) {
  return (
    <Tooltip title={label}>
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 4,
          marginRight: 8,
          marginBottom: 4,
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 14,
            height: 14,
            borderRadius: 3,
            background: color,
            border: "1px solid #555",
          }}
        />
        <Text style={{ color: "#ccc", fontSize: 11 }}>{label}</Text>
      </div>
    </Tooltip>
  );
}

/** Map color description words to approximate CSS colors */
function inferSwatches(description: string): Array<{ color: string; label: string }> {
  const lower = description.toLowerCase();
  const swatches: Array<{ color: string; label: string }> = [];

  if (lower.includes("warm")) {
    swatches.push({ color: "#e67e22", label: "暖色调" });
  }
  if (lower.includes("cool")) {
    swatches.push({ color: "#3498db", label: "冷色调" });
  }
  if (lower.includes("neutral")) {
    swatches.push({ color: "#95a5a6", label: "中性" });
  }
  if (lower.includes("cyan")) {
    swatches.push({ color: "#00bcd4", label: "青色" });
  }
  if (lower.includes("orange")) {
    swatches.push({ color: "#ff9800", label: "橙色" });
  }
  if (lower.includes("yellow")) {
    swatches.push({ color: "#f1c40f", label: "黄色" });
  }
  if (lower.includes("red")) {
    swatches.push({ color: "#e74c3c", label: "红色" });
  }
  if (lower.includes("blue")) {
    swatches.push({ color: "#2980b9", label: "蓝色" });
  }
  if (lower.includes("green")) {
    swatches.push({ color: "#27ae60", label: "绿色" });
  }
  if (lower.includes("purple") || lower.includes("violet")) {
    swatches.push({ color: "#8e44ad", label: "紫色" });
  }
  if (lower.includes("pink")) {
    swatches.push({ color: "#e91e63", label: "粉色" });
  }
  if (lower.includes("monochrome") || lower.includes("black") || lower.includes("white") || lower.includes("gray") || lower.includes("grey")) {
    swatches.push({ color: "#7f8c8d", label: "单色" });
  }

  return swatches;
}

interface Props {
  data: VisualAnalysis;
}

const VisualAnalysisPanel: React.FC<Props> = ({ data }) => {
  const swatches = data.color_summary
    ? inferSwatches(data.color_summary.description)
    : [];

  return (
    <div>
      {/* Shot summary */}
      <div style={{ marginBottom: 16 }}>
        <Text style={{ color: "#aaa", fontSize: 12 }}>镜头分析</Text>
        <div style={{ marginTop: 4, display: "flex", gap: 24, flexWrap: "wrap" }}>
          <div>
            <Text style={{ color: "#e8e8e8", fontSize: 18, fontWeight: 600 }}>
              {data.shot_count}
            </Text>
            <Text style={{ color: "#888", fontSize: 11, marginLeft: 4 }}>
              个镜头
            </Text>
          </div>
          <div>
            <Text style={{ color: "#e8e8e8", fontSize: 18, fontWeight: 600 }}>
              {formatDuration(data.avg_shot_duration)}
            </Text>
            <Text style={{ color: "#888", fontSize: 11, marginLeft: 4 }}>
              平均时长
            </Text>
          </div>
        </div>
      </div>

      {/* Transitions */}
      {data.transitions.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>转场类型</Text>
          <div style={{ marginTop: 4, display: "flex", gap: 4, flexWrap: "wrap" }}>
            {data.transitions.map((t, i) => (
              <Tag key={i} color="geekblue" style={{ margin: 0, fontSize: 11 }}>
                {t}
              </Tag>
            ))}
          </div>
        </div>
      )}

      {/* Thumbnail grid */}
      {data.shots.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>
            关键帧缩略图 ({data.representative_frames.length} 代表帧)
          </Text>
          <div
            style={{
              marginTop: 8,
              maxHeight: 160,
              overflowY: "auto",
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
            }}
          >
            <Image.PreviewGroup>
              {data.shots.map((shot) => {
                const thumbPath =
                  shot.thumbnail_path ?? data.keyframe_grid_paths[shot.index] ?? null;
                const isRep =
                  shot.is_representative ||
                  data.representative_frames.includes(
                    thumbPath ?? "",
                  );
                return (
                  <div
                    key={shot.index}
                    style={{
                      position: "relative",
                      width: 80,
                      height: 45,
                      flexShrink: 0,
                      borderRadius: 4,
                      overflow: "hidden",
                      background: "#1a1a1a",
                      border: isRep ? "2px solid #faad14" : "1px solid #333",
                    }}
                  >
                    {thumbPath ? (
                      <Image
                        src={`file://${thumbPath}`}
                        alt={`Shot ${shot.index + 1}`}
                        width={80}
                        height={45}
                        style={{ objectFit: "cover" }}
                        preview={{ mask: null }}
                        fallback="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAiIGhlaWdodD0iNDUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0iIzFhMWExYSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBkb21pbmFudC1iYXNlbGluZT0ibWlkZGxlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjNTU1IiBmb250LXNpemU9IjEwIj7lm77moIc8L3RleHQ+PC9zdmc+"
                      />
                    ) : (
                      <div
                        style={{
                          width: "100%",
                          height: "100%",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <PictureOutlined style={{ color: "#555", fontSize: 14 }} />
                      </div>
                    )}
                    {isRep && (
                      <StarFilled
                        style={{
                          position: "absolute",
                          top: 2,
                          right: 2,
                          color: "#faad14",
                          fontSize: 10,
                        }}
                      />
                    )}
                    <div
                      style={{
                        position: "absolute",
                        bottom: 0,
                        left: 0,
                        right: 0,
                        background: "rgba(0,0,0,0.6)",
                        color: "#ccc",
                        fontSize: 9,
                        textAlign: "center",
                        padding: "1px 0",
                      }}
                    >
                      #{shot.index + 1}
                    </div>
                  </div>
                );
              })}
            </Image.PreviewGroup>
          </div>
        </div>
      )}

      {/* Color summary */}
      {data.color_summary && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>色彩分析</Text>
          <div style={{ marginTop: 4, display: "flex", flexWrap: "wrap" }}>
            {swatches.length > 0 ? (
              swatches.map((s, i) => (
                <ColorSwatch key={i} color={s.color} label={s.label} />
              ))
            ) : (
              <Text style={{ color: "#ccc", fontSize: 12 }}>
                {data.color_summary.description}
              </Text>
            )}
          </div>
          {data.color_summary.description && swatches.length > 0 && (
            <Text style={{ color: "#888", fontSize: 11, display: "block", marginTop: 2 }}>
              {data.color_summary.description}
            </Text>
          )}
        </div>
      )}

      {/* Object detection */}
      {data.object_detections.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>物体检测</Text>
          <div style={{ marginTop: 4, display: "flex", gap: 4, flexWrap: "wrap" }}>
            {data.object_detections.map((obj, i) => {
              const label =
                (obj.label as string) ||
                (obj.class as string) ||
                `物体 ${i + 1}`;
              const count = (obj.count as number) ?? 1;
              return (
                <Tag key={i} color="purple" style={{ margin: 0, fontSize: 11 }}>
                  {label}
                  {count > 1 && ` (${count})`}
                </Tag>
              );
            })}
          </div>
        </div>
      )}

      {/* Motion summary */}
      {data.motion_summary && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>运动分析</Text>
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
            {data.motion_summary}
          </Paragraph>
        </div>
      )}

      {/* Face detection */}
      {data.face_detections.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Text style={{ color: "#aaa", fontSize: 12 }}>人脸检测</Text>
          <div style={{ marginTop: 4 }}>
            {data.face_detections.map((face, i) => {
              const count = (face.count as number) ?? (face.face_count as number) ?? 1;
              const shotInfo = face.shot_index != null
                ? `镜头 #${(face.shot_index as number) + 1}`
                : "";
              return (
                <div key={i} style={{ marginBottom: 2 }}>
                  <Text style={{ color: "#ccc", fontSize: 12 }}>
                    检测到 {count} 张人脸
                    {shotInfo && (
                      <Text style={{ color: "#888", fontSize: 11 }}>
                        {" "}({shotInfo})
                      </Text>
                    )}
                  </Text>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Empty state for missing data */}
      {data.shot_count === 0 &&
        data.object_detections.length === 0 &&
        !data.color_summary &&
        !data.motion_summary &&
        data.face_detections.length === 0 && (
          <Empty
            description={
              <Text style={{ color: "#666" }}>暂无视觉分析数据</Text>
            }
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        )}
    </div>
  );
};

export default VisualAnalysisPanel;
