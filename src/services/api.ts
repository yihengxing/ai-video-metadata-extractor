/**
 * HTTP + WebSocket client for communicating with the Python backend.
 *
 * The backend URL is obtained dynamically from the Electron main process
 * via the preload bridge (window.electronAPI.getBackendUrl).
 */
import type { AnalysisResult, AnalysisProgress } from "../types/metadata";

class ApiClient {
  private baseUrl: string = "";

  /** Must be called once before any other API calls. Resolves the backend URL from Electron. */
  async init(): Promise<void> {
    if (this.baseUrl) return;
    if (window.electronAPI) {
      this.baseUrl = await window.electronAPI.getBackendUrl();
    } else {
      // Fallback for browser-only dev (no Electron shell)
      this.baseUrl = "http://127.0.0.1:8000";
    }
  }

  /** Ensure init has been called. */
  private async ensureInit(): Promise<void> {
    if (!this.baseUrl) await this.init();
  }

  // ---- Health ----

  async checkHealth(): Promise<boolean> {
    await this.ensureInit();
    try {
      const res = await fetch(`${this.baseUrl}/health`);
      return res.ok;
    } catch {
      return false;
    }
  }

  // ---- Analysis ----

  async startAnalysis(
    filePath: string,
    modules: string[],
  ): Promise<{ file_hash: string }> {
    await this.ensureInit();
    const res = await fetch(`${this.baseUrl}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_path: filePath, modules }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
    }
    return res.json();
  }

  /** Browser-mode: upload file content directly (no local path access). */
  async uploadAndAnalyze(
    file: File,
    modules: string[],
  ): Promise<{ file_hash: string }> {
    await this.ensureInit();
    const form = new FormData();
    form.append("file", file);
    form.append("modules", modules.join(","));
    const res = await fetch(`${this.baseUrl}/analyze/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
    }
    return res.json();
  }

  async getAnalysisStatus(
    fileHash: string,
  ): Promise<Record<string, unknown>> {
    await this.ensureInit();
    const res = await fetch(`${this.baseUrl}/analyze/${fileHash}/status`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
    }
    return res.json();
  }

  // ---- Cache ----

  async getCachedResult(fileHash: string): Promise<AnalysisResult | null> {
    await this.ensureInit();
    const res = await fetch(`${this.baseUrl}/cache/${fileHash}`);
    if (res.status === 404) return null;
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
    }
    return res.json();
  }

  async deleteCache(fileHash: string): Promise<void> {
    await this.ensureInit();
    const res = await fetch(`${this.baseUrl}/cache/${fileHash}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
    }
  }

  // ---- WebSocket URL helper ----

  getWebSocketUrl(fileHash: string): string {
    // Convert http:// -> ws://   and   https:// -> wss://
    const wsBase = this.baseUrl.replace(/^http/, "ws");
    return `${wsBase}/ws/${fileHash}`;
  }

  // ---- Export ----

  async getExport(fileHash: string, format: string): Promise<string> {
    await this.ensureInit();
    const res = await fetch(
      `${this.baseUrl}/export/${fileHash}?format=${encodeURIComponent(format)}`,
    );
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
    }
    return res.text();
  }

  // ---- Settings ----

  async getSettings(): Promise<Record<string, unknown>> {
    await this.ensureInit();
    const res = await fetch(`${this.baseUrl}/settings`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async updateSettings(updates: Record<string, unknown>): Promise<void> {
    await this.ensureInit();
    const res = await fetch(`${this.baseUrl}/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
    }
  }
}

export const api = new ApiClient();
