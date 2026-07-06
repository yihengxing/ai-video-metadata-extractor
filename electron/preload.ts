import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),
  onMenuCompare: (callback: () => void) => {
    ipcRenderer.on('menu:compare', (_event) => callback());
  },
  onMenuSettings: (callback: () => void) => {
    ipcRenderer.on('menu:settings', (_event) => callback());
  },
  onMenuOpenFile: (callback: () => void) => {
    ipcRenderer.on('menu:open-file', (_event) => callback());
  },
  openFileDialog: () => ipcRenderer.invoke('dialog:openFile'),
  saveFile: (name: string, content: string) => ipcRenderer.invoke('dialog:saveFile', name, content),
  removeMenuCompareListener: () => {
    ipcRenderer.removeAllListeners('menu:compare');
  },
  removeMenuSettingsListener: () => {
    ipcRenderer.removeAllListeners('menu:settings');
  },
  removeMenuOpenFileListener: () => {
    ipcRenderer.removeAllListeners('menu:open-file');
  },
});
