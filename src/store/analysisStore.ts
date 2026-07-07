/**
 * Zustand store — central state for the analysis workflow.
 *
 * Manages: current file / hash, module selection, analysis progress,
 * result, history, and compare mode.
 */
import { create } from "zustand";
import type { AnalysisResult, ModuleKey } from "../types/metadata";
import { MODULE_KEYS } from "../types/metadata";

export interface HistoryEntry {
  hash: string;
  filePath: string;
  analyzedAt: string; // ISO 8601
}

export interface AnalysisState {
  // ---- Current analysis ----
  currentFile: string | null;
  currentFileObject: File | null;  // Browser mode: the actual File object
  currentSavedPath: string | null; // Server-side path saved after upload
  currentHash: string | null;
  selectedModules: ModuleKey[];
  isAnalyzing: boolean;
  progress: Record<string, number>; // module -> 0-100
  result: AnalysisResult | null;
  error: string | null;

  // ---- History ----
  history: HistoryEntry[];

  // ---- Compare mode ----
  compareOpen: boolean;
  leftResult: AnalysisResult | null;
  rightResult: AnalysisResult | null;

  // ---- Actions ----
  setFile: (path: string, fileObj?: File) => void;
  setSavedPath: (savedPath: string) => void;
  setHash: (hash: string) => void;
  toggleModule: (module: ModuleKey) => void;
  setIsAnalyzing: (v: boolean) => void;
  setProgress: (module: string, pct: number) => void;
  setResult: (result: AnalysisResult) => void;
  setError: (error: string | null) => void;
  addToHistory: (entry: HistoryEntry) => void;
  removeFromHistory: (hash: string) => void;
  setCompareOpen: (open: boolean) => void;
  setLeftResult: (result: AnalysisResult | null) => void;
  setRightResult: (result: AnalysisResult | null) => void;
  reset: () => void;
}

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem("analysisHistory");
    if (raw) return JSON.parse(raw);
  } catch {
    // ignore corrupt data
  }
  return [];
}

function persistHistory(history: HistoryEntry[]): void {
  try {
    localStorage.setItem("analysisHistory", JSON.stringify(history));
  } catch {
    // ignore quota errors
  }
}

const initialState = {
  currentFile: null as string | null,
  currentFileObject: null as File | null,
  currentSavedPath: null as string | null,
  currentHash: null as string | null,
  selectedModules: ["tech"] as ModuleKey[],
  isAnalyzing: false,
  progress: {} as Record<string, number>,
  result: null as AnalysisResult | null,
  error: null as string | null,
  history: loadHistory(),
  compareOpen: false,
  leftResult: null as AnalysisResult | null,
  rightResult: null as AnalysisResult | null,
};

export const useAnalysisStore = create<AnalysisState>((set, get) => ({
  ...initialState,

  setFile: (path: string, fileObj?: File) =>
    set({ currentFile: path, currentFileObject: fileObj ?? null }),

  setHash: (hash: string) => set({ currentHash: hash }),

  setSavedPath: (savedPath: string) => set({ currentSavedPath: savedPath }),

  toggleModule: (module: ModuleKey) => {
    // "tech" is always mandatory and cannot be toggled off
    if (module === "tech") return;
    const selected = get().selectedModules;
    const idx = selected.indexOf(module);
    if (idx >= 0) {
      set({ selectedModules: selected.filter((m) => m !== module) });
    } else {
      set({ selectedModules: [...selected, module] });
    }
  },

  setIsAnalyzing: (v: boolean) => set({ isAnalyzing: v }),

  setProgress: (module: string, pct: number) =>
    set((s) => ({
      progress: { ...s.progress, [module]: pct },
    })),

  setResult: (result: AnalysisResult) => set({ result, isAnalyzing: false }),

  setError: (error: string | null) => set({ error, isAnalyzing: false }),

  addToHistory: (entry: HistoryEntry) => {
    const current = get().history;
    // Avoid duplicates — replace if same hash exists
    const filtered = current.filter((e) => e.hash !== entry.hash);
    const updated = [entry, ...filtered].slice(0, 50); // cap at 50 entries
    persistHistory(updated);
    set({ history: updated });
  },

  removeFromHistory: (hash: string) => {
    const updated = get().history.filter((e) => e.hash !== hash);
    persistHistory(updated);
    set({ history: updated });
  },

  setCompareOpen: (open: boolean) => set({ compareOpen: open }),

  setLeftResult: (result: AnalysisResult | null) => set({ leftResult: result }),

  setRightResult: (result: AnalysisResult | null) => set({ rightResult: result }),

  reset: () => set({ ...initialState, history: get().history }), // preserve history
}));
