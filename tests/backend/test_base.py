import pytest
from backend.modules.base import Extractor, Matcher
from backend.models.schemas import TechMetadata, SourceRecoveryHit


class FakeExtractor(Extractor):
    async def extract(self, file_path: str, tech_meta: TechMetadata, progress_cb=None) -> dict:
        return {"test": "ok"}

    @property
    def module_name(self) -> str:
        return "fake"


class FakeMatcher(Matcher):
    async def match(self, keyframe_paths: list[str], progress_cb=None) -> list[SourceRecoveryHit]:
        return []

    @property
    def matcher_name(self) -> str:
        return "fake_matcher"


def test_extractor_has_module_name():
    ext = FakeExtractor()
    assert ext.module_name == "fake"


def test_extractor_raises_on_extract_not_implemented():
    class BadExtractor(Extractor):
        pass
    with pytest.raises(TypeError):
        BadExtractor()


def test_matcher_has_matcher_name():
    m = FakeMatcher()
    assert m.matcher_name == "fake_matcher"
