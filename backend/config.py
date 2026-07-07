"""后端配置管理。API Key 通过 Electron 主进程传入，不落盘明文。"""
from __future__ import annotations
import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from typing import Optional


CONFIG_DIR = Path.home() / ".ai-video-analyzer"
CONFIG_FILE = CONFIG_DIR / "config.json"
KEY_FILE = CONFIG_DIR / ".key"

SUPPORTED_EXTENSIONS = {".mp4", ".webm", ".flv", ".mkv", ".mov", ".avi"}


class Settings:
    """运行时配置。敏感值由 Electron 通过 HTTP 启动请求传入，加密落盘。"""

    def __init__(self) -> None:
        self.data: dict = {
            "llm_api_key": "",
            "llm_provider": "claude",  # "claude" | "openai" | "gemini" | "qwen" | "custom"
            "llm_custom_endpoint": "",  # 自定义 OpenAI 兼容端点 URL
            "llm_custom_model": "",     # 自定义模型名
            "saucenao_api_key": "",
            "acrcloud_key": "",
            "acrcloud_secret": "",
            "source_recovery_consent": False,  # v1.3: 源回捞上传同意标记
            "theme": "dark",
            "cache_dir": str(CONFIG_DIR / "cache"),
            "models_dir": str(CONFIG_DIR / "models"),
        }
        self._fernet: Optional[Fernet] = None
        self._load()

    def _init_crypto(self) -> Fernet:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if KEY_FILE.exists():
            key = KEY_FILE.read_bytes()
        else:
            key = Fernet.generate_key()
            os.chmod(str(CONFIG_DIR), 0o700)
            KEY_FILE.write_bytes(key)
            os.chmod(str(KEY_FILE), 0o600)
        self._fernet = Fernet(key)
        return self._fernet

    @property
    def fernet(self) -> Fernet:
        if self._fernet is None:
            self._init_crypto()
        return self._fernet  # type: ignore[return-value]

    def _load(self) -> None:
        if CONFIG_FILE.exists():
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            # Decrypt sensitive fields
            for field in ("llm_api_key", "saucenao_api_key", "acrcloud_key", "acrcloud_secret"):
                if field in raw and raw[field]:
                    try:
                        raw[field] = self.fernet.decrypt(raw[field].encode()).decode()
                    except Exception:
                        raw[field] = ""
            self.data.update(raw)

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        to_save = dict(self.data)
        # Encrypt sensitive fields
        for field in ("llm_api_key", "saucenao_api_key", "acrcloud_key", "acrcloud_secret"):
            if field in to_save:
                val = to_save[field] or ""
                to_save[field] = self.fernet.encrypt(val.encode()).decode()
        os.chmod(str(CONFIG_DIR), 0o700)
        CONFIG_FILE.write_text(json.dumps(to_save, indent=2, ensure_ascii=False))
        os.chmod(str(CONFIG_FILE), 0o600)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value

    @property
    def llm_api_key(self) -> str:
        return self.data.get("llm_api_key", "")

    @property
    def saucenao_api_key(self) -> str:
        return self.data.get("saucenao_api_key", "")

    @property
    def source_recovery_consent(self) -> bool:
        return bool(self.data.get("source_recovery_consent", False))

    @property
    def supported_extensions(self) -> set[str]:
        return SUPPORTED_EXTENSIONS


# 模块级单例
settings = Settings()
