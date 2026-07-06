/**
 * useHistory — manages the analysis history list (load, search, remove, load-result).
 * History is persisted to localStorage via the Zustand store.
 */
import { useState, useMemo } from "react";
import { useAnalysisStore } from "../store/analysisStore";
import type { HistoryEntry } from "../store/analysisStore";

export function useHistory() {
  const history = useAnalysisStore((s) => s.history);
  const removeFromHistory = useAnalysisStore((s) => s.removeFromHistory);
  const setResult = useAnalysisStore((s) => s.setResult);
  const setHash = useAnalysisStore((s) => s.setHash);
  const setFile = useAnalysisStore((s) => s.setFile);

  const [searchQuery, setSearchQuery] = useState("");

  const filteredHistory = useMemo(() => {
    if (!searchQuery.trim()) return history;
    const q = searchQuery.toLowerCase();
    return history.filter(
      (entry) =>
        entry.filePath.toLowerCase().includes(q) ||
        entry.hash.toLowerCase().includes(q),
    );
  }, [history, searchQuery]);

  const removeEntry = (hash: string) => {
    removeFromHistory(hash);
  };

  const loadResult = async (entry: HistoryEntry) => {
    setHash(entry.hash);
    setFile(entry.filePath);
    try {
      const { api } = await import("../services/api");
      const result = await api.getCachedResult(entry.hash);
      if (result) {
        setResult(result);
      }
    } catch {
      // cache miss, silently skip
    }
  };

  return {
    history: filteredHistory,
    searchQuery,
    setSearchQuery,
    removeEntry,
    loadResult,
  };
}
