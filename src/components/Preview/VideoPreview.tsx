/**
 * VideoPreview — drop zone / video player area.
 *
 * Accepts drag-and-drop of video files; displays a placeholder
 * when no file is loaded and basic file info when a file is selected.
 * Actual video playback via video.js will be wired in a later task.
 */
import React, { useState, useCallback, useRef } from "react";
import { Typography } from "antd";
import {
  VideoCameraOutlined,
  InboxOutlined,
  FileOutlined,
} from "@ant-design/icons";
import { useAnalysisStore } from "../../store/analysisStore";

const { Text } = Typography;

function formatFileSize(bytes: number | undefined): string {
  if (!bytes) return "未知大小";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function shortName(filePath: string): string {
  if (!filePath) return "";
  const parts = filePath.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || filePath;
}

const VideoPreview: React.FC = () => {
  const currentFile = useAnalysisStore((s) => s.currentFile);
  const setFile = useAnalysisStore((s) => s.setFile);
  const [isDragOver, setIsDragOver] = useState(false);
  const dropRef = useRef<HTMLDivElement>(null);

  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(true);
    },
    [],
  );

  const handleDragLeave = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);
    },
    [],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        const file = files[0];
        // In Electron, we can use the path property for full file path
        const filePath = (file as unknown as { path?: string }).path ?? file.name;
        setFile(filePath);
      }
    },
    [setFile],
  );

  const handleClickZone = useCallback(async () => {
    // Use Electron's native file dialog if available
    if (window.electronAPI) {
      const filePath = await window.electronAPI.openFileDialog();
      if (filePath) {
        setFile(filePath);
      }
    }
  }, [setFile]);

  // ---- File loaded state ----
  if (currentFile) {
    return (
      <div
        style={{
          width: "100%",
          // 16:9 aspect ratio container
          aspectRatio: "16 / 9",
          maxHeight: "calc(100% - 8px)",
          background: "#000",
          borderRadius: 8,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          border: "2px solid #1f1f1f",
        }}
      >
        <VideoCameraOutlined style={{ fontSize: 48, color: "#1677ff", marginBottom: 12 }} />
        <FileOutlined style={{ fontSize: 16, color: "#888", marginBottom: 8 }} />
        <Text style={{ color: "#e8e8e8", fontSize: 14 }}>{shortName(currentFile)}</Text>
        <Text type="secondary" style={{ fontSize: 12, marginTop: 4 }}>
          拖拽新文件以替换
        </Text>
      </div>
    );
  }

  // ---- Empty drop zone ----
  const dropZoneBg = isDragOver
    ? "rgba(22, 119, 255, 0.12)"
    : "rgba(255, 255, 255, 0.03)";
  const dropZoneBorder = isDragOver ? "2px dashed #1677ff" : "2px dashed #434343";

  return (
    <div
      ref={dropRef}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClickZone}
      style={{
        width: "100%",
        // 16:9
        aspectRatio: "16 / 9",
        maxHeight: "calc(100% - 8px)",
        background: dropZoneBg,
        borderRadius: 8,
        border: dropZoneBorder,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        transition: "all 0.2s ease",
      }}
    >
      <InboxOutlined
        style={{
          fontSize: 56,
          color: isDragOver ? "#1677ff" : "#595959",
          marginBottom: 16,
          transition: "color 0.2s",
        }}
      />
      <Text
        style={{
          color: isDragOver ? "#1677ff" : "#999",
          fontSize: 15,
          transition: "color 0.2s",
        }}
      >
        拖拽视频文件到此处
      </Text>
      <Text type="secondary" style={{ fontSize: 12, marginTop: 6 }}>
        或点击选择文件（支持 .mp4、.mov、.avi、.mkv）
      </Text>
    </div>
  );
};

export default VideoPreview;
