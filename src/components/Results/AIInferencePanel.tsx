/**
 * AIInferencePanel — AI-generated content inference results.
 *
 * Gray background to visually distinguish from SourceRecoveryPanel.
 * Displays inferred tool, prompt, style tags, workflow chain,
 * imitation suggestions, and model recommendations with confidence badges.
 */
import React from "react";
import { Typography, Tag, Button, Tooltip, message, Divider, Empty } from "antd";
import { CopyOutlined, BulbOutlined } from "@ant-design/icons";
import type { AIInference } from "../../types/metadata";

const { Text, Title, Paragraph } = Typography;

const PANEL_BG = "#1a1a1a";

function confidenceBadge(score: number) {
  if (score > 0.8) {
    return (
      <Tag color="success" style={{ margin: 0, fontSize: 10 }}>
        🟢 {Math.round(score * 100)}%
      </Tag>
    );
  }
  if (score >= 0.5) {
    return (
      <Tag color="warning" style={{ margin: 0, fontSize: 10 }}>
        🟡 {Math.round(score * 100)}%
      </Tag>
    );
  }
  return (
    <Tag color="error" style={{ margin: 0, fontSize: 10 }}>
      🔴 {Math.round(score * 100)}%
    </Tag>
  );
}

interface Props {
  data: AIInference;
}

const AIInferencePanel: React.FC<Props> = ({ data }) => {
  const handleCopyPrompt = () => {
    if (data.inferred_prompt) {
      // Copy as ComfyUI-compatible format
      navigator.clipboard.writeText(data.inferred_prompt).then(() => {
        message.success("已复制推断 Prompt 到剪贴板");
      });
    }
  };

  const hasAnyData =
    data.inferred_tool ||
    data.inferred_prompt ||
    data.style_tags.length > 0 ||
    data.inferred_workflow ||
    data.imitation_suggestions.length > 0 ||
    data.model_recommendations.length > 0;

  return (
    <div
      style={{
        backgroundColor: PANEL_BG,
        borderRadius: 4,
        padding: 12,
      }}
    >
      {!hasAnyData ? (
        <Empty
          description={
            <Text style={{ color: "#666" }}>暂无 AI 推断数据</Text>
          }
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <>
          {/* Overall confidence */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 12,
            }}
          >
            <Text style={{ color: "#aaa", fontSize: 12 }}>整体置信度</Text>
            {confidenceBadge(data.overall_confidence)}
          </div>

          <Divider style={{ margin: "8px 0", borderColor: "#303030" }} />

          {/* Inferred tool */}
          {data.inferred_tool && (
            <div style={{ marginBottom: 12 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <Text style={{ color: "#aaa", fontSize: 12 }}>推断工具</Text>
                {confidenceBadge(data.inferred_tool_confidence)}
              </div>
              <Text
                style={{
                  color: "#e8e8e8",
                  fontSize: 13,
                  fontWeight: 500,
                  display: "block",
                  marginTop: 4,
                }}
              >
                {data.inferred_tool}
              </Text>
            </div>
          )}

          <Divider style={{ margin: "8px 0", borderColor: "#303030" }} />

          {/* Inferred prompt */}
          {data.inferred_prompt && (
            <div style={{ marginBottom: 12 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 4,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Text style={{ color: "#aaa", fontSize: 12 }}>推断 Prompt</Text>
                  {confidenceBadge(data.inferred_prompt_confidence)}
                </div>
                <Tooltip title="复制为 ComfyUI 格式">
                  <Button
                    type="link"
                    size="small"
                    icon={<CopyOutlined />}
                    onClick={handleCopyPrompt}
                    style={{ color: "#1677ff", fontSize: 11, padding: 0 }}
                  >
                    复制
                  </Button>
                </Tooltip>
              </div>
              <Paragraph
                style={{
                  color: "#c8d6e5",
                  fontSize: 12,
                  background: "rgba(0,0,0,0.2)",
                  padding: 8,
                  borderRadius: 4,
                  margin: 0,
                  lineHeight: 1.5,
                }}
              >
                {data.inferred_prompt}
              </Paragraph>
            </div>
          )}

          {/* Style tags */}
          {data.style_tags.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <Text style={{ color: "#aaa", fontSize: 12 }}>风格标签</Text>
              <div
                style={{
                  marginTop: 4,
                  display: "flex",
                  gap: 4,
                  flexWrap: "wrap",
                }}
              >
                {data.style_tags.map((tag, i) => (
                  <Tag key={i} color="magenta" style={{ margin: 0, fontSize: 11 }}>
                    {tag}
                  </Tag>
                ))}
              </div>
            </div>
          )}

          {/* Inferred workflow chain */}
          {data.inferred_workflow && (
            <div style={{ marginBottom: 12 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <Text style={{ color: "#aaa", fontSize: 12 }}>推断工作流</Text>
                {confidenceBadge(data.inferred_workflow_confidence)}
              </div>
              <div
                style={{
                  marginTop: 4,
                  display: "flex",
                  alignItems: "center",
                  flexWrap: "wrap",
                  gap: 6,
                }}
              >
                {data.inferred_workflow
                  .split("->")
                  .map((step, i, arr) => (
                    <React.Fragment key={i}>
                      <Tag
                        color="processing"
                        style={{ margin: 0, fontSize: 11 }}
                      >
                        {step.trim()}
                      </Tag>
                      {i < arr.length - 1 && (
                        <Text style={{ color: "#555", fontSize: 11 }}>→</Text>
                      )}
                    </React.Fragment>
                  ))}
              </div>
            </div>
          )}

          <Divider style={{ margin: "8px 0", borderColor: "#303030" }} />

          {/* Imitation suggestions */}
          {data.imitation_suggestions.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <Text style={{ color: "#aaa", fontSize: 12 }}>
                <BulbOutlined style={{ marginRight: 4, color: "#faad14" }} />
                模仿建议
              </Text>
              <ul
                style={{
                  margin: "4px 0 0 0",
                  paddingLeft: 18,
                  color: "#ccc",
                  fontSize: 12,
                }}
              >
                {data.imitation_suggestions.map((suggestion, i) => (
                  <li key={i} style={{ marginBottom: 2 }}>
                    {suggestion}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Model recommendations */}
          {data.model_recommendations.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <Text style={{ color: "#aaa", fontSize: 12 }}>推荐模型</Text>
              <ul
                style={{
                  margin: "4px 0 0 0",
                  paddingLeft: 18,
                  color: "#ccc",
                  fontSize: 12,
                }}
              >
                {data.model_recommendations.map((model, i) => (
                  <li key={i} style={{ marginBottom: 2 }}>
                    {model}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default AIInferencePanel;
