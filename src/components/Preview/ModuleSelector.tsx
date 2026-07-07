/**
 * ModuleSelector — checkboxes for selecting which analysis modules to run,
 * a "开始分析" button, and a progress bar.
 */
import React from "react";
import { Checkbox, Button, Progress, Tooltip, Typography } from "antd";
import {
  ThunderboltOutlined,
  EyeOutlined,
  SoundOutlined,
  RobotOutlined,
  GlobalOutlined,
  CaretRightOutlined,
} from "@ant-design/icons";
import { useAnalysisStore } from "../../store/analysisStore";
import { useAnalysis } from "../../hooks/useAnalysis";
import {
  MODULE_LABELS,
  MODULE_KEYS,
  type ModuleKey,
} from "../../types/metadata";

const { Text } = Typography;

const MODULE_ICONS: Record<ModuleKey, React.ReactNode> = {
  tech: <ThunderboltOutlined />,
  visual: <EyeOutlined />,
  audio: <SoundOutlined />,
  ai: <RobotOutlined />,
  source_recovery: <GlobalOutlined />,
};

const MODULE_SUBTITLES: Record<ModuleKey, string> = {
  tech: "约 3-5 秒 · 本地",
  visual: "约 30-120 秒 · 需模型",
  audio: "约 20-60 秒 · 需模型",
  ai: "约 15-30 秒 · 需 API",
  source_recovery: "约 5-15 秒 · 需 API",
};

const ModuleSelector: React.FC = () => {
  const selectedModules = useAnalysisStore((s) => s.selectedModules);
  const isAnalyzing = useAnalysisStore((s) => s.isAnalyzing);
  const progress = useAnalysisStore((s) => s.progress);
  const currentFile = useAnalysisStore((s) => s.currentFile);
  const currentFileObject = useAnalysisStore((s) => s.currentFileObject);
  const currentSavedPath = useAnalysisStore((s) => s.currentSavedPath);
  const error = useAnalysisStore((s) => s.error);
  const setError = useAnalysisStore((s) => s.setError);
  const toggleModule = useAnalysisStore((s) => s.toggleModule);
  const { startAnalysis } = useAnalysis();

  const handleToggle = (key: ModuleKey) => {
    toggleModule(key);
  };

  const handleStart = async () => {
    if (!currentFile) return;
    // Priority: saved server path > File object > filename
    const input: string | File = currentSavedPath ?? currentFileObject ?? currentFile;
    await startAnalysis(input);
  };

  // Compute an aggregate progress percentage across active modules
  const activeModules = selectedModules.length;
  const totalProgress = activeModules
    ? selectedModules.reduce((sum, m) => sum + (progress[m] ?? 0), 0) / activeModules
    : 0;

  return (
    <div
      style={{
        background: "rgba(255,255,255,0.04)",
        borderRadius: 8,
        padding: "16px 20px",
      }}
    >
      {/* Module checkboxes */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          marginBottom: 14,
        }}
      >
        {MODULE_KEYS.map((key) => {
          const checked = selectedModules.includes(key);
          const disabled = key === "tech" || isAnalyzing;

          const checkbox = (
            <Checkbox
              key={key}
              checked={checked}
              disabled={disabled}
              onChange={() => handleToggle(key)}
            >
              <span style={{ color: "#e8e8e8", fontSize: 13 }}>
                {MODULE_ICONS[key]}{" "}
                {MODULE_LABELS[key]}
              </span>
              <br />
              <Text
                type="secondary"
                style={{ fontSize: 11, marginLeft: 22 }}
              >
                {MODULE_SUBTITLES[key]}
              </Text>
            </Checkbox>
          );

          if (key === "source_recovery") {
            return (
              <Tooltip
                key={key}
                title="将上传关键帧至第三方服务进行源匹配"
              >
                {checkbox}
              </Tooltip>
            );
          }
          return checkbox;
        })}
      </div>

      {/* Action row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
        }}
      >
        <Button
          type="primary"
          icon={<CaretRightOutlined />}
          size="large"
          disabled={!currentFile || isAnalyzing}
          loading={isAnalyzing}
          onClick={handleStart}
        >
          开始分析
        </Button>

        {/* Inline progress bar */}
        <div style={{ flex: 1 }}>
          {isAnalyzing && (
            <Progress
              percent={Math.round(totalProgress)}
              size="small"
              status="active"
              strokeColor="#1677ff"
            />
          )}
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div
          style={{
            marginTop: 12,
            padding: "8px 12px",
            background: "rgba(255, 77, 79, 0.1)",
            border: "1px solid rgba(255, 77, 79, 0.3)",
            borderRadius: 6,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span style={{ color: "#ff7875", fontSize: 13 }}>{error}</span>
          <Button
            type="text"
            size="small"
            style={{ color: "#ff7875" }}
            onClick={() => setError(null)}
          >
            关闭
          </Button>
        </div>
      )}
    </div>
  );
};

export default ModuleSelector;
