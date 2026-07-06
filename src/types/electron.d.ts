export interface ElectronAPI {
  getBackendUrl: () => Promise<string>;
  onMenuCompare: (callback: () => void) => void;
  onMenuSettings: (callback: () => void) => void;
  onMenuOpenFile: (callback: () => void) => void;
  openFileDialog: () => Promise<string | null>;
  saveFile: (defaultName: string, content: string) => Promise<boolean>;
  removeMenuCompareListener: () => void;
  removeMenuSettingsListener: () => void;
  removeMenuOpenFileListener: () => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
