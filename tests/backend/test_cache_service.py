import json
import tempfile
import os
from backend.services.cache_service import CacheService


def test_cache_put_and_get():
    tmp = tempfile.mkdtemp()
    try:
        svc = CacheService(cache_dir=tmp)
        hash_val = "a" * 64
        data = {"test": "value", "module_status": {}}
        svc.put(hash_val, data)
        assert svc.exists(hash_val)
        result = svc.get(hash_val)
        assert result == data
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_miss_returns_none():
    svc = CacheService()
    assert svc.get("nonexistent_hash_12345") is None
    assert not svc.exists("nonexistent_hash_12345")


def test_cache_delete():
    tmp = tempfile.mkdtemp()
    try:
        svc = CacheService(cache_dir=tmp)
        hash_val = "b" * 64
        svc.put(hash_val, {"x": 1})
        assert svc.exists(hash_val)
        svc.delete(hash_val)
        assert not svc.exists(hash_val)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
