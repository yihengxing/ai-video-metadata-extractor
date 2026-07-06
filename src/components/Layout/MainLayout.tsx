/**
 * MainLayout — three-column application shell.
 *
 * ┌──────────┬────────────────────────┬──────────────┐
 * │  History │  Preview + Controls    │   Results    │
 * │  (250px) │  (flex)                │   (400px)    │
 * └──────────┴────────────────────────┴──────────────┘
 *
 * Also manages the CompareView modal triggered by File > Compare menu.
 */
import React, { useEffect } from "react";
import { Layout } from "antd";
import HistoryPanel from "../History/HistoryPanel";
import VideoPreview from "../Preview/VideoPreview";
import ModuleSelector from "../Preview/ModuleSelector";
import ResultsPanel from "../Results/ResultsPanel";
import CompareView from "../Compare/CompareView";
import { useAnalysisStore } from "../../store/analysisStore";

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
  const setCompareOpen = useAnalysisStore((s) => s.setCompareOpen);
  const setLeftResult = useAnalysisStore((s) => s.setLeftResult);
  const setRightResult = useAnalysisStore((s) => s.setRightResult);

  // Listen for menu:compare from Electron main process
  useEffect(() => {
    if (window.electronAPI) {
      window.electronAPI.onMenuCompare(() => {
        setLeftResult(null);
        setRightResult(null);
        setCompareOpen(true);
      });

      return () => {
        window.electronAPI.removeMenuCompareListener();
      };
    }
  }, [setCompareOpen, setLeftResult, setRightResult]);

  return (
    <Layout style={{ height: "100vh" }}>
      {/* Left — History */}
      <Sider width={250} style={siderStyle} theme="dark">
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
      <Sider width={400} style={rightSiderStyle} theme="dark">
        <ResultsPanel />
      </Sider>

      {/* Compare modal */}
      <CompareView />
    </Layout>
  );
};

export default MainLayout;
