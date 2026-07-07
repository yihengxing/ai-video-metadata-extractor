import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import { app } from 'electron';

export class PythonManager {
  private process: ChildProcess | null = null;
  private port: number = 0;
  private backendDir: string;

  constructor() {
    this.backendDir = path.join(app.getAppPath(), 'backend');
  }

  /**
   * Find a working Python executable.
   * Checks in order: embedded Python (bundled with app), python3, python.
   */
  private findPythonExecutable(): string {
    // Check for embedded Python first (production build)
    const isDev = !app.isPackaged;
    let embeddedPath: string;

    if (isDev) {
      // In development, python-embedded sits at the project root
      embeddedPath = path.join(app.getAppPath(), 'python-embedded', 'python.exe');
    } else {
      // In production, python-embedded is in the app resources
      embeddedPath = path.join(process.resourcesPath, 'python-embedded', 'python.exe');
    }

    if (process.platform === 'win32') {
      if (fs.existsSync(embeddedPath)) {
        console.log(`[PythonManager] Found embedded Python: ${embeddedPath}`);
        return embeddedPath;
      }
    } else {
      // On macOS/Linux, check for python3 in embedded dir
      const embeddedUnix = path.join(isDev ? app.getAppPath() : process.resourcesPath, 'python-embedded', 'bin', 'python3');
      if (fs.existsSync(embeddedUnix)) {
        console.log(`[PythonManager] Found embedded Python: ${embeddedUnix}`);
        return embeddedUnix;
      }
    }

    // Fall back to system Python
    console.log('[PythonManager] Embedded Python not found, trying system python3...');
    return 'python3';
  }

  /**
   * Start the Python FastAPI backend as a subprocess.
   * Uses --port 0 so uvicorn assigns a free port, then parses the port from stdout/stderr.
   */
  async start(): Promise<void> {
    console.log('[PythonManager] Starting Python backend...');

    const pythonExe = this.findPythonExecutable();
    const args = ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '0'];

    console.log(`[PythonManager] Spawning: ${pythonExe} ${args.join(' ')}`);
    console.log(`[PythonManager] Working directory: ${this.backendDir}`);

    this.process = spawn(pythonExe, args, {
      cwd: this.backendDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    // Parse the assigned port from stderr (uvicorn prints there) and stdout
    const portPromise = new Promise<number>((resolve, reject) => {
      let outputBuffer = '';

      const onData = (chunk: Buffer) => {
        outputBuffer += chunk.toString();
        console.log(`[PythonManager:python] ${chunk.toString().trim()}`);

        // uvicorn prints: "Uvicorn running on http://127.0.0.1:{port} (Press CTRL+C to quit)"
        const match = outputBuffer.match(/Uvicorn running on\s+http:\/\/127\.0\.0\.1:(\d+)/i);
        if (match) {
          const port = parseInt(match[1], 10);
          resolve(port);
        }
      };

      const onError = (chunk: Buffer) => {
        console.error(`[PythonManager:python:err] ${chunk.toString().trim()}`);
      };

      this.process?.stdout?.on('data', onData);
      this.process?.stderr?.on('data', (chunk: Buffer) => {
        onError(chunk);
        // uvicorn also prints the running message on stderr
        onData(chunk);
      });

      this.process?.on('error', (err) => {
        reject(new Error(`Failed to start Python process: ${err.message}`));
      });

      this.process?.on('exit', (code, signal) => {
        if (!this.port) {
          reject(new Error(`Python process exited before port was detected. Code: ${code}, Signal: ${signal}`));
        }
      });

      // Timeout after 30 seconds
      setTimeout(() => {
        if (!this.port) {
          reject(new Error('Timed out waiting for Python backend to report its port (30s)'));
        }
      }, 30000);
    });

    try {
      this.port = await portPromise;
      console.log(`[PythonManager] Detected backend port: ${this.port}`);
    } catch (err) {
      console.error(`[PythonManager] Failed to detect port: ${(err as Error).message}`);
      await this.stop();
      throw err;
    }

    // Health-check: GET /health every 500ms until 200 or timeout (30s)
    try {
      await this.waitForHealth();
      console.log('[PythonManager] Backend is healthy and ready.');
    } catch (err) {
      console.error(`[PythonManager] Health check failed: ${(err as Error).message}`);
      await this.stop();
      throw err;
    }
  }

  /**
   * Poll the backend /health endpoint until it returns 200.
   */
  private waitForHealth(maxWaitMs: number = 30000): Promise<void> {
    const startTime = Date.now();
    const pollInterval = 500;

    return new Promise((resolve, reject) => {
      const poll = () => {
        // Check if we timed out
        if (Date.now() - startTime > maxWaitMs) {
          reject(new Error(`Health check timed out after ${maxWaitMs}ms`));
          return;
        }

        const req = http.get(`http://127.0.0.1:${this.port}/health`, (res) => {
          if (res.statusCode === 200) {
            res.resume(); // consume response data to free memory
            resolve();
          } else {
            console.log(`[PythonManager] Health check returned ${res.statusCode}, retrying...`);
            res.resume();
            setTimeout(poll, pollInterval);
          }
        });

        req.on('error', (err: NodeJS.ErrnoException) => {
          // Connection refused is normal during startup
          if (err.code === 'ECONNREFUSED' || err.code === 'ECONNRESET') {
            console.log(`[PythonManager] Backend not ready yet (${err.code}), retrying in ${pollInterval}ms...`);
            setTimeout(poll, pollInterval);
          } else {
            reject(new Error(`Health check failed: ${err.message}`));
          }
        });

        req.setTimeout(2000, () => {
          req.destroy();
          console.log('[PythonManager] Health check request timed out, retrying...');
          setTimeout(poll, pollInterval);
        });
      };

      // Start polling after a short initial delay to let uvicorn bind
      setTimeout(poll, 1000);
    });
  }

  /**
   * Return the port the Python backend is listening on.
   */
  getPort(): number {
    return this.port;
  }

  /**
   * Stop the Python backend subprocess.
   * Sends SIGTERM, waits up to 5s, then SIGKILL if still alive.
   */
  async stop(): Promise<void> {
    if (!this.process) return;

    console.log('[PythonManager] Stopping Python backend...');

    return new Promise((resolve) => {
      const killTimer = setTimeout(() => {
        console.log('[PythonManager] Force killing Python process (SIGKILL)...');
        if (this.process && this.process.pid) {
          try {
            process.kill(this.process.pid, 'SIGKILL');
          } catch {
            // Process may have already exited
          }
        }
      }, 5000);

      this.process!.on('exit', (code, signal) => {
        clearTimeout(killTimer);
        console.log(`[PythonManager] Python process exited (code: ${code}, signal: ${signal})`);
        this.process = null;
        this.port = 0;
        resolve();
      });

      // Send graceful termination
      if (this.process!.pid) {
        const isWin = process.platform === 'win32';
        if (isWin) {
          // On Windows, SIGTERM isn't supported; use taskkill for graceful shutdown, then SIGKILL
          spawn('taskkill', ['/pid', String(this.process!.pid), '/f', '/t']);
        } else {
          this.process!.kill('SIGTERM');
        }
      }
    });
  }
}
