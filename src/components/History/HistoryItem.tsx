/**
 * HistoryItem — single row in the history panel.
 */
import React from "react";
import { Button, Typography, Tooltip } from "antd";
import {
  DeleteOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import type { HistoryEntry } from "../../store/analysisStore";

const { Text } = Typography;

interface HistoryItemProps {
  entry: HistoryEntry;
  onSelect: (entry: HistoryEntry) => void;
  onDelete: (hash: string) => void;
}

/** Extract a short display name from the absolute file path. */
function shortFilePath(filePath: string): string {
  if (!filePath) return "未知文件";
  const parts = filePath.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || filePath;
}

function formatDate(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

const HistoryItem: React.FC<HistoryItemProps> = ({
  entry,
  onSelect,
  onDelete,
}) => {
  const handleClick = () => {
    onSelect(entry);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete(entry.hash);
  };

  return (
    <div
      onClick={handleClick}
      style={{
        display: "flex",
        alignItems: "center",
        padding: "8px 10px",
        cursor: "pointer",
        borderRadius: 6,
        marginBottom: 4,
        transition: "background 0.15s",
        background: "transparent",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.background = "rgba(255,255,255,0.08)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.background = "transparent";
      }}
    >
      <VideoCameraOutlined
        style={{ color: "#1677ff", marginRight: 8, fontSize: 16 }}
      />
      <div style={{ flex: 1, overflow: "hidden", minWidth: 0 }}>
        <Tooltip title={entry.filePath}>
          <Text
            style={{
              color: "#e8e8e8",
              display: "block",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              fontSize: 13,
            }}
          >
            {shortFilePath(entry.filePath)}
          </Text>
        </Tooltip>
        <Text
          type="secondary"
          style={{ fontSize: 11 }}
        >
          {formatDate(entry.analyzedAt)}
        </Text>
      </div>
      <Tooltip title="删除记录">
        <Button
          type="text"
          size="small"
          danger
          icon={<DeleteOutlined />}
          onClick={handleDelete}
        />
      </Tooltip>
    </div>
  );
};

export default HistoryItem;
