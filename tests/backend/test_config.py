import tempfile
from pathlib import Path
from unittest.mock import patch


def test_save_and_load_empty_api_key():
    """清空 API key 后保存再读取，确认空字符串正确加密持久化。"""
    from backend.config import CONFIG_DIR, CONFIG_FILE, KEY_FILE, Settings

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_file = config_dir / "config.json"
        key_file = config_dir / ".key"

        with patch("backend.config.CONFIG_DIR", config_dir), \
             patch("backend.config.CONFIG_FILE", config_file), \
             patch("backend.config.KEY_FILE", key_file):
            s = Settings()
            # 先设一个有值的内容
            s.set("llm_api_key", "initial-key")
            s.save()

            # 再清空它
            s.set("llm_api_key", "")
            s.save()

            # 重新加载，确认空字符串被持久化
            s2 = Settings()
            assert s2.get("llm_api_key") == "", \
                f"Expected empty string, got: {s2.get('llm_api_key')!r}"
