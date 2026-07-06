/**
 * ResultsPanel — main container for all analysis result panels.
 *
 * Uses Ant Design Collapse with ordered sections. Each panel is
 * locked/faded until its corresponding module completes. Shows a
 * loading spinner while the module is running and a green checkmark
 * when complete.
 *
 * Panel order: 技术信息 → 源回捞 → 视觉分析 → 音频分析 → AI推断
 *
 * Also includes an export dropdown (导出) for downloading results in
 * various formats via the Electron save dialog.
 */
import React, { useMemo, useCallback } from "react";
import { Collapse, Spin, Badge, Typography, Dropdown, Button, Tooltip, message } from "antd";
import {
  CheckCircleFilled,
  LoadingOutlined,
  LockOutlined,
  CloseCircleFilled,
  MinusCircleOutlined,
  ExportOutlined,
  DownOutlined,
} from "@ant-design/icons";
import { useAnalysisStore } from "../../store/analysisStore";
import { api } from "../../services/api";
import {
  MODULE_LABELS,
  type ModuleStatusValue,
} from "../../types/metadata";

import TechInfoPanel from "./TechInfoPanel";
import SourceRecoveryPanel from "./SourceRecoveryPanel";
import VisualAnalysisPanel from "./VisualAnalysisPanel";
import AudioAnalysisPanel from "./AudioAnalysisPanel";
import AIInferencePanel from "./AIInferencePanel";

const { Text } = Typography;

/** Custom panel header with status indicator and lock icon */
function PanelHeader({
  label,
  moduleStatus,
  isSelected,
}: {
  label: string;
  moduleStatus: ModuleStatusValue | undefined;
  isSelected: boolean;
}) {
  const iconStyle: React.CSSProperties = {
    fontSize: 14,
    marginRight: 8,
  };

  // Module not in progress (not selected or not started): show lock, faded
  if (!moduleStatus || moduleStatus === "pending") {
    return (
      <span style={{ opacity: 0.5 }}>
        <LockOutlined style={{ ...iconStyle, color: "#666" }} />
        <Text style={{ color: "#888", fontSize: 13 }}>{label}</Text>
      </span>
    );
  }

  // Running: show spinner
  if (moduleStatus === "running") {
    return (
      <span>
        <Spin
          indicator={<LoadingOutlined style={{ ...iconStyle, color: "#1677ff" }} />}
          size="small"
        />
        <Text style={{ color: "#1677ff", fontSize: 13 }}>{label}</Text>
      </span>
    );
  }

  // Completed: green checkmark
  if (moduleStatus === "completed") {
    return (
      <span>
        <CheckCircleFilled style={{ ...iconStyle, color: "#52c41a" }} />
        <Text style={{ color: "#e8e8e8", fontSize: 13 }}>{label}</Text>
      </span>
    );
  }

  // Failed: red X
  if (moduleStatus === "failed") {
    return (
      <span>
        <CloseCircleFilled style={{ ...iconStyle, color: "#ff4d4f" }} />
        <Text style={{ color: "#ff4d4f", fontSize: 13 }}>{label}</Text>
      </span>
    );
  }

  // Skipped: gray dash
  if (moduleStatus === "skipped") {
    return (
      <span style={{ opacity: 0.5 }}>
        <MinusCircleOutlined style={{ ...iconStyle, color: "#666" }} />
        <Text style={{ color: "#888", fontSize: 13 }}>{label}</Text>
      </span>
    );
  }

  return <span>{label}</span>;
}

// Panel definitions in display order
const PANEL_DEFS = [
  { key: "tech", label: MODULE_LABELS.tech },
  { key: "source_recovery", label: MODULE_LABELS.source_recovery },
  { key: "visual", label: MODULE_LABELS.visual },
  { key: "audio", label: MODULE_LABELS.audio },
  { key: "ai", label: MODULE_LABELS.ai },
] as const;

const ResultsPanel: React.FC = () => {
  const result = useAnalysisStore((s) => s.result);
  const currentHash = useAnalysisStore((s) => s.currentHash);
  const isAnalyzing = useAnalysisStore((s) => s.isAnalyzing);
  const selectedModules = useAnalysisStore((s) => s.selectedModules);

  /** Trigger export of the current result in the given format. */
  const handleExport = useCallback(
    async (format: string) => {
      if (!currentHash) {
        message.warning("没有可导出的分析结果");
        return;
      }

      try {
        const content = await api.getExport(currentHash, format);

        // Build a reasonable default filename
        const extMap: Record<string, string> = {
          json: ".json",
          markdown: ".md",
          comfyui_workflow: ".json",
          comfyui_prompt: ".txt",
          srt: ".srt",
        };
        const extension = extMap[format] ?? `.${format}`;
        const defaultName = `analysis_${currentHash.slice(0, 8)}${extension}`;

        // Use Electron save dialog if available, otherwise fall back to browser download
        if (window.electronAPI?.saveFile) {
          const saved = await window.electronAPI.saveFile(defaultName, content);
          if (saved) {
            message.success("导出成功");
          }
        } else {
          // Browser fallback
          const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = defaultName;
          a.click();
          URL.revokeObjectURL(url);
          message.success("导出成功");
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "导出失败";
        message.error(msg);
      }
    },
    [currentHash],
  );

  // Check whether source recovery has a workflow available
  const hasWorkflow = result?.source_recovery?.workflow_json != null;
  const hasPrompt = result?.source_recovery?.prompt != null;

  // Determine active keys: always show tech if selected, show completed/running modules
  const activeKeys = useMemo(() => {
    if (!result && !isAnalyzing) {
      // Nothing happening yet — show all that are selected
      return selectedModules;
    }

    const keys: string[] = [];
    for (const def of PANEL_DEFS) {
      const status = result?.module_status?.[def.key];
      // Show if selected, completed, running, or failed
      if (
        selectedModules.includes(def.key as typeof selectedModules[number]) ||
        status === "completed" ||
        status === "running" ||
        status === "failed"
      ) {
        keys.push(def.key);
      }
    }
    return keys;
  }, [result, isAnalyzing, selectedModules]);

  // Determine which panel should be expanded by default
  const defaultActiveKey = useMemo(() => {
    if (!result) return undefined;

    // Prefer completed panels in order
    for (const def of PANEL_DEFS) {
      const status = result.module_status?.[def.key];
      if (status === "completed") return def.key;
    }

    // If none completed, find the running one
    for (const def of PANEL_DEFS) {
      const status = result.module_status?.[def.key];
      if (status === "running") return def.key;
    }

    // Otherwise expand tech if selected
    if (selectedModules.includes("tech")) return "tech";

    return undefined;
  }, [result, selectedModules]);

  // Check if source_recovery has a complete_match → AI panel auto-collapses
  const sourceRecoveryStatus = result?.source_recovery?.status;

  return (
    <div style={{ padding: 8 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 12,
          padding: "0 8px",
        }}
      >
        <Text style={{ color: "#fff", fontSize: 15, fontWeight: 600 }}>
          分析结果
        </Text>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {isAnalyzing && (
            <Badge status="processing" text={<Text style={{ color: "#1677ff", fontSize: 11 }}>分析中...</Text>} />
          )}
          {result && (
            <Dropdown
              menu={{
                items: [
                  {
                    key: "json",
                    label: "JSON (.json)",
                    icon: <ExportOutlined />,
                    onClick: () => handleExport("json"),
                  },
                  {
                    key: "markdown",
                    label: "Markdown 报告 (.md)",
                    icon: <ExportOutlined />,
                    onClick: () => handleExport("markdown"),
                  },
                  {
                    key: "comfyui_workflow",
                    label: "ComfyUI Workflow JSON (.json)",
                    icon: <ExportOutlined />,
                    disabled: !hasWorkflow,
                    onClick: () => handleExport("comfyui_workflow"),
                  },
                  {
                    key: "comfyui_prompt",
                    label: "ComfyUI Prompt (.txt)",
                    icon: <ExportOutlined />,
                    disabled: !hasPrompt,
                    onClick: () => handleExport("comfyui_prompt"),
                  },
                  {
                    key: "srt",
                    label: "SRT 字幕 (.srt)",
                    icon: <ExportOutlined />,
                    onClick: () => handleExport("srt"),
                  },
                ],
              }}
              trigger={["click"]}
            >
              <Button size="small" icon={<ExportOutlined />}>
                导出 <DownOutlined />
              </Button>
            </Dropdown>
          )}
        </div>
      </div>

      {!result && !isAnalyzing && (
        <div
          style={{
            padding: "24px 16px",
            textAlign: "center",
          }}
        >
          <Text style={{ color: "#888", fontSize: 12 }}>
            选择视频文件并点击「开始分析」后，结果将在此显示。
          </Text>
        </div>
      )}

      <Collapse
        accordion
        defaultActiveKey={defaultActiveKey}
        activeKey={undefined}
        ghost
        style={{ background: "transparent" }}
        items={PANEL_DEFS.map((def) => {
          const status = result?.module_status?.[def.key];

          return {
            key: def.key,
            label: (
              <PanelHeader
                label={def.label}
                moduleStatus={status}
                isSelected={selectedModules.includes(
                  def.key as typeof selectedModules[number],
                )}
              />
            ),
            children: renderPanelContent(def.key, result, status),
            style: {
              marginBottom: 4,
              background:
                status === "completed"
                  ? "rgba(255,255,255,0.03)"
                  : "transparent",
              borderRadius: 6,
              border: "1px solid #1f1f1f",
            },
            collapsible:
              status === "completed" || status === "running" || status === "failed"
                ? "header"
                : ("disabled" as const),
            showArrow: status === "completed" || status === "running",
          };
        })}
      />
    </div>
  );
};

/** Render content for each panel based on its key and data availability */
function renderPanelContent(
  key: string,
  result: import("../../types/metadata").AnalysisResult | null,
  status: ModuleStatusValue | undefined,
): React.ReactNode {
  // Still running — show loading skeleton
  if (status === "running") {
    return (
      <div style={{ textAlign: "center", padding: "24px 0" }}>
        <Spin
          indicator={<LoadingOutlined style={{ fontSize: 20, color: "#1677ff" }} />}
        />
        <Text
          style={{
            display: "block",
            color: "#888",
            fontSize: 12,
            marginTop: 8,
          }}
        >
          正在分析中...
        </Text>
      </div>
    );
  }

  // Failed
  if (status === "failed") {
    return (
      <div style={{ textAlign: "center", padding: "16px 0" }}>
        <CloseCircleFilled style={{ fontSize: 20, color: "#ff4d4f" }} />
        <Text
          style={{
            display: "block",
            color: "#ff4d4f",
            fontSize: 12,
            marginTop: 8,
          }}
        >
          分析失败
        </Text>
      </div>
    );
  }

  // No result yet
  if (!result) return null;

  switch (key) {
    case "tech":
      return <TechInfoPanel data={result.tech_metadata} />;

    case "source_recovery": {
      const sr = result.source_recovery;
      // Hide panel entirely for miss or null
      if (!sr || sr.status === "miss") {
        return (
          <div style={{ textAlign: "center", padding: "16px 0" }}>
            <Text style={{ color: "#888", fontSize: 12 }}>
              未找到匹配的源内容
            </Text>
          </div>
        );
      }
      return <SourceRecoveryPanel data={sr} />;
    }

    case "visual":
      if (!result.visual_analysis) {
        return (
          <div style={{ textAlign: "center", padding: "16px 0" }}>
            <Text style={{ color: "#888", fontSize: 12 }}>
              未选择视觉分析模块或暂无数据
            </Text>
          </div>
        );
      }
      return <VisualAnalysisPanel data={result.visual_analysis} />;

    case "audio":
      if (!result.audio_analysis) {
        return (
          <div style={{ textAlign: "center", padding: "16px 0" }}>
            <Text style={{ color: "#888", fontSize: 12 }}>
              未选择音频分析模块或暂无数据
            </Text>
          </div>
        );
      }
      return <AudioAnalysisPanel data={result.audio_analysis} />;

    case "ai":
      if (!result.ai_inference) {
        return (
          <div style={{ textAlign: "center", padding: "16px 0" }}>
            <Text style={{ color: "#888", fontSize: 12 }}>
              未选择 AI 推断模块或暂无数据
            </Text>
          </div>
        );
      }
      return <AIInferencePanel data={result.ai_inference} />;

    default:
      return null;
  }
}

export default ResultsPanel;
