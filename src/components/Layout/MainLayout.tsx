/**
 * MainLayout — three-column application shell.
 *
 * ┌──────────┬────────────────────────┬──────────────┐
 * │  History │  Preview + Controls    │   Results    │
 * │  (250px) │  (flex)                │   (400px)    │
 * └──────────┴────────────────────────┴──────────────┘
 */
import React from "react";
import { Layout } from "antd";
import HistoryPanel from "../History/HistoryPanel";
import VideoPreview from "../Preview/VideoPreview";
import ModuleSelector from "../Preview/ModuleSelector";
import ResultsPanel from "../Results/ResultsPanel";

const { Sider, Content } = Layout;

const siderStyle: React.CSSProperties = {
  overflow: "auto",
  height: "100vh",
  borderRight: "1px solid #303030",
};

const contentStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  height: "100vh",
  overflow: "auto",
  padding: 24,
};

const rightSiderStyle: React.CSSProperties = {
  overflow: "auto",
  height: "100vh",
  borderLeft: "1px solid #303030",
};

const MainLayout: React.FC = () => {
  return (
    <Layout style={{ height: "100vh" }}>
      {/* Left — History */}
      <Sider
        width={250}
        style={siderStyle}
        theme="dark"
      >
        <HistoryPanel />
      </Sider>

      {/* Center — Preview + Module Selector */}
      <Content style={contentStyle}>
        <div style={{ flex: 1, minHeight: 0 }}>
          <VideoPreview />
        </div>
        <div style={{ flexShrink: 0, marginTop: 16 }}>
          <ModuleSelector />
        </div>
      </Content>

      {/* Right — Results */}
      <Sider
        width={400}
        style={rightSiderStyle}
        theme="dark"
      >
        <ResultsPanel />
      </Sider>
    </Layout>
  );
};

export default MainLayout;
