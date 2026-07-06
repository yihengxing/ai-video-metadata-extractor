import { app, BrowserWindow, ipcMain, dialog, Menu } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import { PythonManager } from './python-manager';

let mainWindow: BrowserWindow | null = null;
let pythonManager: PythonManager;

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    title: 'AI 视频元数据提取工具',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    backgroundColor: '#141414', // dark theme
  });

  // Dev: load Vite dev server. Prod: load built files.
  if (process.env.VITE_DEV_SERVER_URL) {
    await mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    await mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  // Menu bar with File > Compare
  const menu = Menu.buildFromTemplate([
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Video...',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            mainWindow?.webContents.send('menu:open-file');
          },
        },
        {
          label: 'Compare',
          accelerator: 'CmdOrCtrl+D',
          click: () => mainWindow?.webContents.send('menu:compare'),
        },
        { type: 'separator' },
        {
          label: 'Settings...',
          accelerator: 'CmdOrCtrl+,',
          click: () => mainWindow?.webContents.send('menu:settings'),
        },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    { label: 'Edit', submenu: [{ role: 'copy' }, { role: 'selectAll' }] },
    { label: 'View', submenu: [{ role: 'reload' }, { role: 'toggleDevTools' }] },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About',
          click: () =>
            dialog.showMessageBox({
              title: 'About',
              message: 'AI 视频元数据提取工具 v1.3.0',
            }),
        },
      ],
    },
  ]);
  Menu.setApplicationMenu(menu);
}

// IPC handlers
ipcMain.handle('get-backend-url', () => `http://localhost:${pythonManager.getPort()}`);

ipcMain.handle('dialog:openFile', async () => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Open Video File',
    filters: [
      { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv'] },
      { name: 'All Files', extensions: ['*'] },
    ],
    properties: ['openFile'],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle('dialog:saveFile', async (_event, defaultName: string, content: string) => {
  if (!mainWindow) return false;
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: defaultName,
    filters: [{ name: 'All Files', extensions: ['*'] }],
  });
  if (!result.canceled && result.filePath) {
    fs.writeFileSync(result.filePath, content, 'utf-8');
    return true;
  }
  return false;
});

app.whenReady().then(async () => {
  pythonManager = new PythonManager();
  await pythonManager.start(); // Launch Python FastAPI on dynamic port
  await createWindow();
});

app.on('window-all-closed', async () => {
  await pythonManager?.stop();
  app.quit();
});

app.on('before-quit', async () => {
  await pythonManager?.stop();
});
