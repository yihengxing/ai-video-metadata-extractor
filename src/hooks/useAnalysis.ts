/**
 * useAnalysis — orchestrates the full analysis workflow:
 * 1. Validate file & compute hash (invoked by user action)
 * 2. Call POST /analyze
 * 3. Subscribe to WebSocket progress
 * 4. On completion → fetch full result, add to history
 */
import { useCallback } from "react";
import { useAnalysisStore } from "../store/analysisStore";
import { useWebSocket } from "./useWebSocket";
import { api } from "../services/api";

export function useAnalysis() {
  const store = useAnalysisStore();

  // Connect WebSocket whenever we have an active hash
  useWebSocket(store.currentHash);

  const startAnalysis = useCallback(
    async (fileOrPath: string | File) => {
      let filePath: string;
      if (typeof fileOrPath === "string") {
        filePath = fileOrPath;
      } else {
        // Browser mode: use the file name as display path
        filePath = fileOrPath.name;
      }
      store.setFile(filePath);
      store.setError(null);
      store.setProgress("", 0);

      const modules = store.selectedModules;

      try {
        store.setIsAnalyzing(true);

        // In Electron: send path. In browser: upload file.
        let file_hash: string;
        if (typeof fileOrPath === "string") {
          ({ file_hash } = await api.startAnalysis(fileOrPath, modules));
        } else {
          ({ file_hash } = await api.uploadAndAnalyze(fileOrPath, modules));
        }

        store.setHash(file_hash);

        // Poll for completion since backend is in-memory for now.
        // WebSocket will deliver progress messages; we poll status for Done.
        await waitForCompletion(file_hash);
        store.setIsAnalyzing(false);

        // Fetch full result
        const result = await api.getCachedResult(file_hash);
        if (result) {
          store.setResult(result);
          store.addToHistory({
            hash: file_hash,
            filePath: filePath,
            analyzedAt: result.analyzed_at,
          });
        }
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "分析启动失败";
        store.setError(message);
        store.setIsAnalyzing(false);
      }
    },
    [store],
  );

  return {
    startAnalysis,
    isAnalyzing: store.isAnalyzing,
    progress: store.progress,
    result: store.result,
    error: store.error,
  };
}

/**
 * Simple polling waiter — queries GET /analyze/{hash}/status every 1.5 s
 * until the state.status is "completed".
 */
async function waitForCompletion(fileHash: string): Promise<void> {
  const POLL_INTERVAL_MS = 1_500;
  const MAX_POLLS = 300; // ~7.5 min timeout
  const ERROR_POLL_LIMIT = 10;

  let polls = 0;
  let errorStreak = 0;
  let lastError = "";

  while (polls < MAX_POLLS) {
    if (polls > 0) await sleep(POLL_INTERVAL_MS);
    polls += 1;

    try {
      const state = await api.getAnalysisStatus(fileHash);
      errorStreak = 0;
      lastError = "";

      if (state.status === "failed") {
        const msg = (state as Record<string,unknown>).error as string || "后端分析失败";
        throw new Error(msg);
      }
      if (state.status === "completed" || state.status === "skipped") {
        return;
      }
    } catch (err) {
      errorStreak += 1;
      lastError = err instanceof Error ? err.message : String(err);
      console.error(`[Poll #${polls}] 查询状态失败 (${errorStreak}/${ERROR_POLL_LIMIT}):`, lastError);
      if (errorStreak >= ERROR_POLL_LIMIT) {
        throw new Error(`连续 ${ERROR_POLL_LIMIT} 次查询失败: ${lastError}`);
      }
    }
  }

  throw new Error("分析超时 — 后端在合理时间内未完成");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
