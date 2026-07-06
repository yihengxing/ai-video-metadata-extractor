/**
 * useWebSocket — connects to backend WebSocket endpoint for real-time
 * analysis progress updates.  Auto-reconnects on disconnect.
 */
import { useEffect, useRef, useState, useCallback } from "react";
import { useAnalysisStore } from "../store/analysisStore";
import { api } from "../services/api";
import type { AnalysisProgress } from "../types/metadata";

const RECONNECT_DELAY_MS = 3_000;
const MAX_RECONNECTS = 5;

export function useWebSocket(fileHash: string | null) {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const setProgress = useAnalysisStore((s) => s.setProgress);

  const connect = useCallback(() => {
    if (!fileHash) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const url = api.getWebSocketUrl(fileHash);
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      setError(null);
      reconnectCount.current = 0;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data: AnalysisProgress = JSON.parse(event.data as string);
        setProgress(data.module, data.progress_pct);
      } catch {
        // ignore malformed messages
      }
    };

    ws.onerror = () => {
      setError("WebSocket 连接错误");
    };

    ws.onclose = (event: CloseEvent) => {
      setConnected(false);
      wsRef.current = null;

      // Only reconnect if it wasn't a clean closure and we haven't exceeded max
      if (!event.wasClean && reconnectCount.current < MAX_RECONNECTS) {
        reconnectCount.current += 1;
        reconnectTimer.current = setTimeout(() => {
          connect();
        }, RECONNECT_DELAY_MS);
      }
    };

    wsRef.current = ws;
  }, [fileHash, setProgress]);

  useEffect(() => {
    connect();
    return () => {
      // Cleanup on unmount or hash change
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      const ws = wsRef.current;
      if (ws) {
        ws.onclose = null; // prevent reconnect on intentional close
        ws.close(1000, "Component unmounted");
        wsRef.current = null;
      }
    };
  }, [fileHash, connect]);

  return { connected, error };
}
