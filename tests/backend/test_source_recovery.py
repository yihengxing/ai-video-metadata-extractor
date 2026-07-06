"""Tests for SourceRecoveryMatcher — SauceNAO reverse search + metadata retrieval.

Covers:
- matcher_name property
- Graceful degradation when no API key configured
- SauceNAO returns no results
- Complete match: SauceNAO -> Civitai full chain
- Partial match: Civitai without workflow
- Early stop after 2 high-confidence (>90%) hits
- Civitai API failure downgrades to located_only or partial_match
- comfyworkflows scraping
- Hit aggregation and voting
- Progress reporting
"""
from __future__ import annotations
import pytest
import json
import os
import sys
from unittest.mock import patch
from backend.models.schemas import SourceRecoveryHit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_jpeg() -> bytes:
    """Create a minimal valid JPEG image in memory (1x1 red pixel)."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09"
        b"\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
        b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff"
        b"\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04"
        b"\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q"
        b"\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\x09\x0a\x16"
        b"\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
        b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99"
        b"\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
        b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
        b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1"
        b"\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00"
        b"\x00?\x00\xd2\xcf \x11\x00\x03\x11\x01\x02\x11\x01\x03\x11\x01"
        b"\xff\xd9"
    )


def _write_test_keyframes(temp_dir: str, count: int = 5) -> list[str]:
    """Write *count* tiny JPEG files into *temp_dir* and return their paths."""
    paths: list[str] = []
    jpeg_data = _make_test_jpeg()
    for i in range(count):
        p = os.path.join(temp_dir, f"keyframe_{i:04d}.jpg")
        with open(p, "wb") as f:
            f.write(jpeg_data)
        paths.append(p)
    return paths


def _make_saucenao_json_response(
    results: list[dict] | None = None,
    status: int = 0,
) -> dict:
    """Build a SauceNAO JSON response."""
    if results is None:
        results = []
    return {
        "header": {
            "status": status,
            "results_requested": 5,
            "results_returned": len(results),
        },
        "results": results,
    }


def _make_saucenao_result(
    similarity: float = 95.0,
    url: str = "https://civitai.com/images/12345",
    title: str = "Test Generation",
    source_site: str = "civitai",
    index_name: str = "Civitai",
) -> dict:
    """Build a single SauceNAO result entry."""
    return {
        "header": {
            "similarity": str(similarity),
            "thumbnail": "",
            "index_name": index_name,
        },
        "data": {
            "ext_urls": [url],
            "title": title,
            "source": source_site,
            "civitai_id": "12345" if "civitai" in url else None,
        },
    }


def _make_civitai_image_response(
    image_id: int = 12345,
    prompt: str = "a cyberpunk city at night, neon lights, rain",
    seed: int = 42424242,
    sampler: str = "Euler a",
    steps: int = 30,
    cfg_scale: float = 7.0,
    model_name: str = "DreamShaper XL",
    model_hash: str = "abc123def456",
) -> dict:
    """Build a Civitai image API response."""
    return {
        "id": image_id,
        "url": f"https://civitai.com/images/{image_id}",
        "hash": "abc123",
        "width": 1024,
        "height": 1024,
        "nsfw": False,
        "meta": {
            "prompt": prompt,
            "negativePrompt": "ugly, blurry",
            "seed": seed,
            "sampler": sampler,
            "steps": steps,
            "cfgScale": cfg_scale,
            "Model": model_name,
            "model": model_name,
            "Size": "1024x1024",
        },
        "modelId": 999,
        "model": {
            "name": model_name,
            "hash": model_hash,
        },
    }


def _make_civitai_workflow_response() -> dict:
    """Build a Civitai workflow API response."""
    return {
        "items": [
            {
                "id": 1,
                "imageId": 12345,
                "workflow": {
                    "nodes": [
                        {"id": 1, "type": "KSampler", "inputs": {}},
                        {"id": 2, "type": "CLIPTextEncode", "inputs": {}},
                    ],
                    "links": [],
                },
            }
        ],
    }


# ===================================================================
# Tests
# ===================================================================


class TestSourceRecoveryMatcherName:
    """Test matcher_name property."""

    def test_matcher_name(self):
        """matcher_name should return 'saucenao_router'."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        matcher = SourceRecoveryMatcher()
        assert matcher.matcher_name == "saucenao_router"


class TestSourceRecoveryNoApiKey:
    """Test graceful degradation when no SauceNAO API key is configured."""

    @pytest.mark.asyncio
    async def test_match_no_api_key_returns_empty(self, tmp_path):
        """When SauceNAO API key is not configured, return empty list."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=5)

        # Ensure API key is empty
        original_key = settings.saucenao_api_key
        original_consent = settings.source_recovery_consent
        settings.data["saucenao_api_key"] = ""
        settings.data["source_recovery_consent"] = True

        try:
            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)
            assert result == []
        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoverySaucenaoNoResults:
    """Test handling when SauceNAO returns no results."""

    @pytest.mark.asyncio
    async def test_match_saucenao_no_results(self, respx_mock, tmp_path):
        """Mock SauceNAO returning empty results — should return empty list."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=5)

        # Ensure API key is set
        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            # Mock SauceNAO to return empty results
            respx_mock.post("https://saucenao.com/search.php").respond(
                json=_make_saucenao_json_response(results=[], status=0)
            )

            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)
            assert result == []
        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoveryCompleteMatch:
    """Test full chain: SauceNAO -> Civitai with workflow."""

    @pytest.mark.asyncio
    async def test_match_complete_hit(self, respx_mock, tmp_path):
        """Mock the complete chain and verify full SourceRecoveryHit."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=5)

        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            # Mock SauceNAO
            saucenao_route = respx_mock.post(
                "https://saucenao.com/search.php"
            ).respond(
                json=_make_saucenao_json_response(
                    results=[
                        _make_saucenao_result(
                            similarity=96.5,
                            url="https://civitai.com/images/12345",
                            title="Cyberpunk Night Scene",
                        ),
                        _make_saucenao_result(
                            similarity=92.0,
                            url="https://civitai.com/images/12345",
                            title="Cyberpunk Night Scene",
                        ),
                    ]
                )
            )

            # Mock Civitai image API
            civitai_image_route = respx_mock.get(
                "https://civitai.com/api/v1/images/12345"
            ).respond(
                json=_make_civitai_image_response(
                    image_id=12345,
                    prompt="a cyberpunk city at night, neon lights, rain",
                    seed=42424242,
                    sampler="Euler a",
                    steps=30,
                    cfg_scale=7.0,
                    model_name="DreamShaper XL",
                )
            )

            # Mock Civitai workflow API
            civitai_workflow_route = respx_mock.get(
                "https://civitai.com/api/v1/images/12345/workflows"
            ).respond(
                json=_make_civitai_workflow_response()
            )

            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)

            assert len(result) == 1
            hit = result[0]
            assert isinstance(hit, SourceRecoveryHit)
            assert hit.status == "complete_match"
            assert hit.source_url == "https://civitai.com/images/12345"
            assert hit.similarity == pytest.approx(0.965, rel=0.01)
            assert hit.prompt == "a cyberpunk city at night, neon lights, rain"
            assert hit.seed == 42424242
            assert hit.sampler == "Euler a"
            assert hit.steps == 30
            assert hit.cfg_scale == pytest.approx(7.0, rel=0.01)
            assert hit.model_name == "DreamShaper XL"
            assert hit.workflow_json is not None
            assert hit.source_trust == "civitai"
            assert hit.confidence_score > 0.5

            # Verify all SauceNAO mocks were called
            assert saucenao_route.called
            assert civitai_image_route.called
            assert civitai_workflow_route.called

        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoveryPartialMatch:
    """Test Civitai match without workflow JSON."""

    @pytest.mark.asyncio
    async def test_match_partial_hit_no_workflow(self, respx_mock, tmp_path):
        """Mock Civitai without workflow — should be partial_match."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=5)

        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            # Mock SauceNAO
            respx_mock.post("https://saucenao.com/search.php").respond(
                json=_make_saucenao_json_response(
                    results=[
                        _make_saucenao_result(
                            similarity=94.0,
                            url="https://civitai.com/images/12345",
                            title="Test Image",
                        ),
                    ]
                )
            )

            # Mock Civitai image API with data but no workflow
            respx_mock.get(
                "https://civitai.com/api/v1/images/12345"
            ).respond(
                json=_make_civitai_image_response(
                    prompt="a beautiful landscape",
                    seed=11111,
                )
            )

            # Mock Civitai workflow API returning empty
            respx_mock.get(
                "https://civitai.com/api/v1/images/12345/workflows"
            ).respond(
                json={"items": []}
            )

            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)

            assert len(result) == 1
            hit = result[0]
            assert hit.status == "partial_match"
            assert hit.prompt == "a beautiful landscape"
            assert hit.seed == 11111
            assert hit.workflow_json is None
            assert hit.source_trust == "civitai"

        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoveryEarlyStop:
    """Test early-stop optimization when >=2 high-confidence hits found."""

    @pytest.mark.asyncio
    async def test_match_early_stop(self, respx_mock, tmp_path):
        """Verify stops after 2 high-confidence (>90%) hits for same source."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=10)

        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            import httpx as httpx_module

            # Track how many SauceNAO calls are made
            call_count = [0]

            def saucenao_handler(request):
                call_count[0] += 1
                return httpx_module.Response(
                    status_code=200,
                    json=_make_saucenao_json_response(
                        results=[
                            _make_saucenao_result(
                                similarity=97.0,
                                url="https://civitai.com/images/99999",
                                title="High Confidence Match",
                            ),
                        ]
                    ),
                )

            respx_mock.post("https://saucenao.com/search.php").mock(
                side_effect=saucenao_handler
            )

            # Mock Civitai
            respx_mock.get(
                "https://civitai.com/api/v1/images/99999"
            ).respond(
                json=_make_civitai_image_response(image_id=99999)
            )

            respx_mock.get(
                "https://civitai.com/api/v1/images/99999/workflows"
            ).respond(
                json=_make_civitai_workflow_response()
            )

            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)

            # Should have stopped early — no more than 3 calls
            # (2 needed for confidence + possibly 1 more in-flight)
            assert call_count[0] <= 3, (
                f"Expected early stop after 2-3 requests, got {call_count[0]}"
            )

            assert len(result) == 1
            assert result[0].status == "complete_match"

        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoveryCivitaiFailure:
    """Test downgrade logic when Civitai API fails."""

    @pytest.mark.asyncio
    async def test_match_civitai_failure_downgrades(self, respx_mock, tmp_path):
        """When Civitai API returns 500, downgrade to located_only."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=5)

        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            # Mock SauceNAO with a Civitai result
            respx_mock.post("https://saucenao.com/search.php").respond(
                json=_make_saucenao_json_response(
                    results=[
                        _make_saucenao_result(
                            similarity=88.0,
                            url="https://civitai.com/images/55555",
                            title="Broken Image",
                        ),
                    ]
                )
            )

            # Mock Civitai API failure (500)
            respx_mock.get(
                "https://civitai.com/api/v1/images/55555"
            ).respond(
                status_code=500
            )

            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)

            assert len(result) == 1
            hit = result[0]
            assert hit.status == "located_only"
            assert hit.source_url == "https://civitai.com/images/55555"
            assert hit.prompt is None
            assert hit.seed is None

        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoveryComfyWorkflows:
    """Test comfyworkflows.com scraping path."""

    @pytest.mark.asyncio
    async def test_match_comfyworkflows(self, respx_mock, tmp_path):
        """When SauceNAO returns a comfyworkflows URL, parse workflow JSON."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=5)

        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            # Mock SauceNAO with comfyworkflows result
            respx_mock.post("https://saucenao.com/search.php").respond(
                json=_make_saucenao_json_response(
                    results=[
                        _make_saucenao_result(
                            similarity=91.0,
                            url="https://comfyworkflows.com/workflows/my-awesome-workflow",
                            title="My Awesome Workflow",
                            source_site="comfyworkflows",
                        ),
                    ]
                )
            )

            # Mock comfyworkflows page
            workflow_json_str = json.dumps({
                "nodes": [
                    {"id": 1, "type": "CheckpointLoaderSimple"},
                    {"id": 2, "type": "CLIPTextEncode"},
                    {"id": 3, "type": "KSampler"},
                ]
            })

            respx_mock.get(
                "https://comfyworkflows.com/workflows/my-awesome-workflow"
            ).respond(
                html=f"""<html><body>
                <div id="workflow-data" data-workflow='{workflow_json_str}'>
                </div>
                <script>window.__WORKFLOW__ = {workflow_json_str};</script>
                </body></html>"""
            )

            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)

            assert len(result) == 1
            hit = result[0]
            assert hit.status == "partial_match"
            assert hit.source_url == "https://comfyworkflows.com/workflows/my-awesome-workflow"
            assert hit.source_trust == "comfyworkflows"
            assert hit.workflow_json is not None
            # Verify workflow JSON was parsed
            parsed = json.loads(hit.workflow_json)
            assert len(parsed["nodes"]) == 3

        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoveryHitAggregation:
    """Test hit aggregation logic."""

    @pytest.mark.asyncio
    async def test_aggregation_picks_most_voted_url(self, respx_mock, tmp_path):
        """When multiple URLs appear, pick the one with most votes."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=5)

        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            # SauceNAO returns mixed results — URL A appears 3 times, URL B 2 times
            respx_mock.post("https://saucenao.com/search.php").respond(
                json=_make_saucenao_json_response(
                    results=[
                        _make_saucenao_result(
                            similarity=90.0,
                            url="https://civitai.com/images/77777",
                            title="Winner A",
                        ),
                        _make_saucenao_result(
                            similarity=85.0,
                            url="https://civitai.com/images/77777",
                            title="Winner A2",
                        ),
                        _make_saucenao_result(
                            similarity=80.0,
                            url="https://civitai.com/images/77777",
                            title="Winner A3",
                        ),
                        _make_saucenao_result(
                            similarity=88.0,
                            url="https://civitai.com/images/88888",
                            title="Other B",
                        ),
                        _make_saucenao_result(
                            similarity=82.0,
                            url="https://civitai.com/images/88888",
                            title="Other B2",
                        ),
                    ]
                )
            )

            # Mock Civitai for vote_winner
            respx_mock.get(
                "https://civitai.com/api/v1/images/77777"
            ).respond(
                json=_make_civitai_image_response(image_id=77777, prompt="winning prompt")
            )

            respx_mock.get(
                "https://civitai.com/api/v1/images/77777/workflows"
            ).respond(
                json=_make_civitai_workflow_response()
            )

            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)

            assert len(result) == 1
            hit = result[0]
            assert hit.source_url == "https://civitai.com/images/77777"
            assert hit.prompt == "winning prompt"
            # Each of 5 SauceNAO POSTs returns 3 votes for 77777 → 15 total
            assert hit.hit_keyframes == 15

        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoveryProgressReporting:
    """Test that progress callback fires at each stage."""

    @pytest.mark.asyncio
    async def test_progress_callback_fires(self, respx_mock, tmp_path):
        """Progress callback should report stages from 0% to 100%."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=3)

        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            # Mock SauceNAO
            respx_mock.post("https://saucenao.com/search.php").respond(
                json=_make_saucenao_json_response(
                    results=[
                        _make_saucenao_result(
                            similarity=95.0,
                            url="https://civitai.com/images/12345",
                        ),
                    ]
                )
            )

            # Mock Civitai
            respx_mock.get(
                "https://civitai.com/api/v1/images/12345"
            ).respond(
                json=_make_civitai_image_response()
            )
            respx_mock.get(
                "https://civitai.com/api/v1/images/12345/workflows"
            ).respond(
                json=_make_civitai_workflow_response()
            )

            # Collect progress reports
            progress_reports: list[tuple[str, float, str]] = []

            def progress_cb(module: str, pct: float, msg: str) -> None:
                progress_reports.append((module, pct, msg))

            matcher = SourceRecoveryMatcher()
            await matcher.match(keyframe_paths, progress_cb=progress_cb)

            assert len(progress_reports) >= 3, (
                f"Expected at least 3 progress reports, got {len(progress_reports)}"
            )

            # Check first and last
            assert progress_reports[0][0] == "saucenao_router"
            assert progress_reports[-1][0] == "saucenao_router"
            assert progress_reports[-1][1] == 100.0

            # Ensure progress is non-decreasing
            pcts = [p for _, p, _ in progress_reports]
            assert pcts == sorted(pcts), f"Progress should be non-decreasing: {pcts}"

        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent


class TestSourceRecoveryPixivSource:
    """Test located_only result for non-Civitai/non-comfyworkflows sources."""

    @pytest.mark.asyncio
    async def test_match_pixiv_source_located_only(self, respx_mock, tmp_path):
        """Pixiv URLs should yield located_only with source_trust '需人工核验'."""
        from backend.modules.source_recovery import SourceRecoveryMatcher
        from backend.config import settings

        keyframe_paths = _write_test_keyframes(str(tmp_path), count=5)

        original_key = settings.saucenao_api_key
        settings.data["saucenao_api_key"] = "test-api-key"
        original_consent = settings.source_recovery_consent
        settings.data["source_recovery_consent"] = True

        try:
            respx_mock.post("https://saucenao.com/search.php").respond(
                json=_make_saucenao_json_response(
                    results=[
                        _make_saucenao_result(
                            similarity=89.0,
                            url="https://www.pixiv.net/en/artworks/999999",
                            title="Pixiv Artwork",
                            source_site="pixiv",
                        ),
                    ]
                )
            )

            matcher = SourceRecoveryMatcher()
            result = await matcher.match(keyframe_paths)

            assert len(result) == 1
            hit = result[0]
            assert hit.status == "located_only"
            assert hit.source_url == "https://www.pixiv.net/en/artworks/999999"
            assert hit.prompt is None
            assert hit.seed is None
            assert hit.source_trust == "需人工核验"

        finally:
            settings.data["saucenao_api_key"] = original_key
            settings.data["source_recovery_consent"] = original_consent
