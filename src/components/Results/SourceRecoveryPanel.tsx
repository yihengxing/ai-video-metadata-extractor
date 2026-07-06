/**
 * SourceRecoveryPanel — displays source recovery / reverse image search results.
 *
 * Renders differently based on hit status:
 * - complete_match: full workflow, prompt, params, source URL, copy/export
 * - partial_match: params only, no workflow
 * - located_only: source URL link
 * - miss: hidden entirely (handled by parent)
 *
 * Confidence score uses colored badge thresholds.
 */
import React, { useState } from "react";
import {
  Tag,
  Typography,
  Button,
  Descriptions,
  Tooltip,
  message,
} from "antd";
import {
  CopyOutlined,
  ExportOutlined,
  LinkOutlined,
  CodeOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  ExclamationCircleOutlined,
} from "@ant-design/icons";
import type { SourceRecoveryHit } from "../../types/metadata";

const { Text, Paragraph, Title } = Typography;

const PANEL_BG = "#0d2137";
const PANEL_BORDER = "3px solid #1677ff";

function confidenceBadge(score: number) {
  if (score > 0.8) {
    return (
      <Tag color="success" style={{ margin: 0 }}>
        🟢 {Math.round(score * 100)}%
      </Tag>
    );
  }
  if (score >= 0.5) {
    return (
      <Tag color="warning" style={{ margin: 0 }}>
        🟡 {Math.round(score * 100)}%
      </Tag>
    );
  }
  return (
    <Tag color="error" style={{ margin: 0 }}>
      🔴 {Math.round(score * 100)}%
    </Tag>
  );
}

function statusBadge(status: string) {
  switch (status) {
    case "complete_match":
      return (
        <Tag color="success" style={{ margin: 0 }}>
          🟢 完整匹配
        </Tag>
      );
    case "partial_match":
      return (
        <Tag color="warning" style={{ margin: 0 }}>
          🟡 部分匹配
        </Tag>
      );
    case "located_only":
      return (
        <Tag color="orange" style={{ margin: 0 }}>
          🟠 仅定位
        </Tag>
      );
    default:
      return null;
  }
}

function formatSourceTrust(trust: string | null): string {
  if (!trust) return "未知";
  const map: Record<string, string> = {
    civitai: "Civitai",
    comfyworkflows: "ComfyWorkflows",
    other: "其他来源",
  };
  return map[trust] ?? trust;
}

interface Props {
  data: SourceRecoveryHit;
}

const SourceRecoveryPanel: React.FC<Props> = ({ data }) => {
  const [workflowExpanded, setWorkflowExpanded] = useState(false);

  const handleCopyWorkflow = () => {
    if (data.workflow_json) {
      navigator.clipboard.writeText(data.workflow_json).then(() => {
        message.success("已复制 workflow JSON 到剪贴板");
      });
    }
  };

  const handleCopyPrompt = () => {
    if (data.prompt) {
      navigator.clipboard.writeText(data.prompt).then(() => {
        message.success("已复制 Prompt 到剪贴板");
      });
    }
  };

  const handleExport = () => {
    if (!data.workflow_json) return;
    try {
      const parsed = JSON.parse(data.workflow_json);
      const blob = new Blob([JSON.stringify(parsed, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `workflow_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      message.success("已导出 workflow JSON");
    } catch {
      message.error("Workflow JSON 格式无效，无法导出");
    }
  };

  // ---- complete_match render ----
  if (data.status === "complete_match") {
    return (
      <div
        style={{
          backgroundColor: PANEL_BG,
          borderLeft: PANEL_BORDER,
          borderRadius: 4,
          padding: 12,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 12,
          }}
        >
          <Title level={5} style={{ margin: 0, color: "#e8e8e8" }}>
            源回捞结果
          </Title>
          <div style={{ display: "flex", gap: 8 }}>
            {statusBadge(data.status)}
            {confidenceBadge(data.confidence_score)}
          </div>
        </div>

        <Descriptions
          size="small"
          column={2}
          colon={false}
          labelStyle={{ color: "#aaa", fontSize: 12, paddingRight: 8 }}
          contentStyle={{ color: "#e8e8e8", fontSize: 12 }}
        >
          <Descriptions.Item label="模型">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.model_name || "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="Seed">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.seed != null ? data.seed : "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="采样器">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.sampler || "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="Steps">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.steps != null ? data.steps : "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="CFG Scale">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.cfg_scale != null ? data.cfg_scale : "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="来源">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {formatSourceTrust(data.source_trust)}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="相似度" span={2}>
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              命中 {data.hit_keyframes}/{data.total_keyframes_sent} 关键帧
              {data.similarity > 0 &&
                `（相似度 ${(data.similarity * 100).toFixed(1)}%）`}
            </Text>
          </Descriptions.Item>
        </Descriptions>

        {/* Prompt */}
        {data.prompt && (
          <div style={{ marginTop: 12 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 4,
              }}
            >
              <Text style={{ color: "#aaa", fontSize: 12 }}>Prompt</Text>
              <Button
                type="link"
                size="small"
                icon={<CopyOutlined />}
                onClick={handleCopyPrompt}
                style={{ color: "#1677ff", fontSize: 11, padding: 0 }}
              >
                复制
              </Button>
            </div>
            <Paragraph
              style={{
                color: "#c8d6e5",
                fontSize: 12,
                background: "rgba(0,0,0,0.2)",
                padding: 8,
                borderRadius: 4,
                margin: 0,
              }}
            >
              {data.prompt}
            </Paragraph>
          </div>
        )}

        {/* Workflow JSON */}
        {data.workflow_json && (
          <div style={{ marginTop: 12 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 4,
              }}
            >
              <Text style={{ color: "#aaa", fontSize: 12 }}>
                ComfyUI Workflow
              </Text>
              <div style={{ display: "flex", gap: 8 }}>
                <Button
                  type="link"
                  size="small"
                  icon={<CodeOutlined />}
                  onClick={() => setWorkflowExpanded(!workflowExpanded)}
                  style={{ color: "#1677ff", fontSize: 11, padding: 0 }}
                >
                  {workflowExpanded ? "收起" : "展开"}
                </Button>
                <Button
                  type="link"
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={handleCopyWorkflow}
                  style={{ color: "#1677ff", fontSize: 11, padding: 0 }}
                >
                  复制
                </Button>
                <Button
                  type="link"
                  size="small"
                  icon={<ExportOutlined />}
                  onClick={handleExport}
                  style={{ color: "#1677ff", fontSize: 11, padding: 0 }}
                >
                  导出
                </Button>
              </div>
            </div>
            {workflowExpanded && (
              <pre
                style={{
                  color: "#c8d6e5",
                  fontSize: 11,
                  background: "rgba(0,0,0,0.3)",
                  padding: 8,
                  borderRadius: 4,
                  margin: 0,
                  maxHeight: 200,
                  overflow: "auto",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                }}
              >
                {(() => {
                  try {
                    return JSON.stringify(
                      JSON.parse(data.workflow_json),
                      null,
                      2,
                    );
                  } catch {
                    return data.workflow_json;
                  }
                })()}
              </pre>
            )}
          </div>
        )}

        {/* Source URL */}
        {data.source_url && (
          <div style={{ marginTop: 12 }}>
            <a
              href={data.source_url}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#1677ff", fontSize: 12 }}
            >
              <LinkOutlined style={{ marginRight: 4 }} />
              {data.source_url}
            </a>
          </div>
        )}
      </div>
    );
  }

  // ---- partial_match render ----
  if (data.status === "partial_match") {
    return (
      <div
        style={{
          backgroundColor: PANEL_BG,
          borderLeft: PANEL_BORDER,
          borderRadius: 4,
          padding: 12,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 12,
          }}
        >
          <Title level={5} style={{ margin: 0, color: "#e8e8e8" }}>
            源回捞结果
          </Title>
          <div style={{ display: "flex", gap: 8 }}>
            {statusBadge(data.status)}
            {confidenceBadge(data.confidence_score)}
          </div>
        </div>

        <Descriptions
          size="small"
          column={2}
          colon={false}
          labelStyle={{ color: "#aaa", fontSize: 12, paddingRight: 8 }}
          contentStyle={{ color: "#e8e8e8", fontSize: 12 }}
        >
          <Descriptions.Item label="模型">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.model_name || "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="Seed">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.seed != null ? data.seed : "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="采样器">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.sampler || "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="Steps">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.steps != null ? data.steps : "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="CFG Scale">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {data.cfg_scale != null ? data.cfg_scale : "-"}
            </Text>
          </Descriptions.Item>

          <Descriptions.Item label="来源">
            <Text style={{ color: "#e8e8e8", fontSize: 12 }}>
              {formatSourceTrust(data.source_trust)}
            </Text>
          </Descriptions.Item>
        </Descriptions>

        {data.prompt && (
          <div style={{ marginTop: 12 }}>
            <Text style={{ color: "#aaa", fontSize: 12 }}>Prompt</Text>
            <Paragraph
              style={{
                color: "#c8d6e5",
                fontSize: 12,
                background: "rgba(0,0,0,0.2)",
                padding: 8,
                borderRadius: 4,
                margin: "4px 0 0 0",
              }}
            >
              {data.prompt}
            </Paragraph>
          </div>
        )}

        <div style={{ marginTop: 12 }}>
          <Text style={{ color: "#faad14", fontSize: 12 }}>
            <WarningOutlined style={{ marginRight: 4 }} />
            该原作未公开完整 workflow
          </Text>
        </div>

        {data.source_url && (
          <div style={{ marginTop: 8 }}>
            <a
              href={data.source_url}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#1677ff", fontSize: 12 }}
            >
              <LinkOutlined style={{ marginRight: 4 }} />
              {data.source_url}
            </a>
          </div>
        )}
      </div>
    );
  }

  // ---- located_only render ----
  if (data.status === "located_only") {
    return (
      <div
        style={{
          backgroundColor: PANEL_BG,
          borderLeft: PANEL_BORDER,
          borderRadius: 4,
          padding: 12,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 12,
          }}
        >
          <Title level={5} style={{ margin: 0, color: "#e8e8e8" }}>
            源回捞结果
          </Title>
          <div style={{ display: "flex", gap: 8 }}>
            {statusBadge(data.status)}
            {confidenceBadge(data.confidence_score)}
          </div>
        </div>

        {data.source_url && (
          <div style={{ marginBottom: 8 }}>
            <a
              href={data.source_url}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#1677ff", fontSize: 12 }}
            >
              <LinkOutlined style={{ marginRight: 4 }} />
              {data.source_url}
            </a>
          </div>
        )}

        <Text style={{ color: "#fa8c16", fontSize: 12 }}>
          <ExclamationCircleOutlined style={{ marginRight: 4 }} />
          可手动访问确认
        </Text>
      </div>
    );
  }

  // miss or null — should not reach here, parent handles
  return null;
};

export default SourceRecoveryPanel;
