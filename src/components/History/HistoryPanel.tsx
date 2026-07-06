/**
 * HistoryPanel — left sidebar showing past analysis entries.
 */
import React from "react";
import { Input, Typography, Empty } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import { useHistory } from "../../hooks/useHistory";
import HistoryItem from "./HistoryItem";

const { Text } = Typography;

const HistoryPanel: React.FC = () => {
  const { history, searchQuery, setSearchQuery, removeEntry, loadResult } =
    useHistory();

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        padding: 12,
      }}
    >
      {/* Header */}
      <Text
        strong
        style={{
          color: "#fff",
          fontSize: 16,
          marginBottom: 12,
          display: "block",
        }}
      >
        历史记录
      </Text>

      {/* Search */}
      <Input
        placeholder="搜索文件名或哈希..."
        prefix={<SearchOutlined />}
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        allowClear
        size="small"
        style={{ marginBottom: 12 }}
      />

      {/* List */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {history.length === 0 ? (
          <Empty
            description="暂无历史记录"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            style={{ marginTop: 40 }}
          />
        ) : (
          history.map((entry) => (
            <HistoryItem
              key={entry.hash}
              entry={entry}
              onSelect={loadResult}
              onDelete={removeEntry}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default HistoryPanel;
