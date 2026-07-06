"""Source Recovery Matcher — SauceNAO reverse search + metadata retrieval (v1.3).

Pipeline:
1. Preprocess keyframes for upload
2. POST each keyframe to SauceNAO API (multipart image search)
3. Aggregate results by source URL (voting)
4. For top-voted hit: route to Civitai API or comfyworkflows scraper
5. Classify hit status and compute confidence
"""
from __future__ import annotations
import re
import json
import logging
from typing import Optional

import httpx

from backend.modules.base import Matcher, ProgressCallback
from backend.models.schemas import SourceRecoveryHit, HitStatus
from backend.config import settings
from backend.utils.keyframe_utils import preprocess_for_upload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SAUCENAO_URL = "https://saucenao.com/search.php"
_CIVITAI_API_BASE = "https://civitai.com/api/v1"
_EARLY_STOP_SIMILARITY_THRESHOLD = 90.0
_EARLY_STOP_VOTE_COUNT = 2

# Regex to extract Civitai image ID from URL
_CIVITAI_ID_RE = re.compile(r"civitai\.com/images/(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# SourceRecoveryMatcher
# ---------------------------------------------------------------------------


class SourceRecoveryMatcher(Matcher):
    """Reverse-search keyframes via SauceNAO, then fetch metadata from
    Civitai API or scrape comfyworkflows.com for full workflow recovery.

    Implements the Matcher interface defined in backend.modules.base.
    """

    @property
    def matcher_name(self) -> str:
        return "saucenao_router"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def match(
        self,
        keyframe_paths: list[str],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> list[SourceRecoveryHit]:
        """Execute the full source recovery pipeline.

        Progress: 0 % → 20 % (preprocessing) → 40 % (searching) →
        70 % (fetching metadata) → 90 % (aggregating) → 100 %.
        """
        # I1: Enforce source recovery consent before any uploads
        if not settings.source_recovery_consent:
            logger.warning("源回捞上传同意未授权，跳过源回捞")
            _report(progress_cb, 100.0, "源回捞上传同意未授权，已跳过")
            return []

        api_key = settings.saucenao_api_key
        if not api_key:
            logger.warning("SauceNAO API key 未配置，跳过源回捞")
            _report(progress_cb, 100.0, "SauceNAO API key 未配置，已跳过")
            return []

        # -- 0 % ---------------------------------------------------------
        _report(progress_cb, 0.0, "开始源回捞...")

        # -- 0 % → 20 % : preprocess keyframes ---------------------------
        _report(progress_cb, 5.0, "预处理关键帧...")
        preprocessed: list[bytes] = []
        total_frames = len(keyframe_paths)
        for idx, path in enumerate(keyframe_paths):
            try:
                data = preprocess_for_upload(path)
                preprocessed.append(data)
            except Exception as e:
                logger.warning("关键帧预处理失败 %s: %s", path, e)
            if total_frames > 0:
                sub_pct = 20.0 * ((idx + 1) / total_frames)
                _report(progress_cb, sub_pct, f"预处理关键帧 {idx + 1}/{total_frames}...")

        if not preprocessed:
            _report(progress_cb, 100.0, "无有效关键帧")
            return []

        # -- 20 % → 40 % : search SauceNAO -------------------------------
        _report(progress_cb, 20.0, "SauceNAO 反向搜索中...")
        saucenao_hits: list[dict] = []
        frames_sent = 0
        early_stopped = False

        async with httpx.AsyncClient(timeout=30.0) as client:
            for idx, image_data in enumerate(preprocessed):
                frames_sent = idx + 1

                hits = await self._search_saucenao(client, image_data, api_key)
                for h in hits:
                    h["_frame_index"] = idx
                saucenao_hits.extend(hits)

                # Progress within 20-40 % band
                sub_pct = 20.0 + 20.0 * (frames_sent / len(preprocessed))
                _report(
                    progress_cb, sub_pct,
                    f"SauceNAO 搜索 {frames_sent}/{len(preprocessed)}",
                )

                # Early-stop check: if >=2 frames match the same source URL
                # with >90 % similarity, stop sending remaining frames.
                if self._should_early_stop(saucenao_hits):
                    early_stopped = True
                    logger.info(
                        "SauceNAO 早停: %d/%d 帧已发送，足够高置信度命中",
                        frames_sent, len(preprocessed),
                    )
                    break

            # -- 40 % → 70 % : fetch metadata from source -----------------
            _report(progress_cb, 40.0, "聚合搜索结果...")

            if not saucenao_hits:
                _report(progress_cb, 100.0, "SauceNAO 无结果")
                return []

            # Aggregate hits by source URL
            primary = self._aggregate_hits(saucenao_hits)
            hit_keyframes = primary["vote_count"]
            primary_url = primary["url"]
            max_similarity = primary["max_similarity"]

            _report(progress_cb, 50.0, f"选定主命中: {primary_url}")

            # Route to appropriate metadata fetcher
            hit = await self._route_and_fetch(
                client, primary_url, progress_cb
            )

            # -- 90 % → 100 % : finalize hit -------------------------------
            _report(progress_cb, 90.0, "汇总命中结果...")

            # Populate common fields
            hit.source_url = primary_url
            hit.similarity = max_similarity / 100.0  # Normalize SauceNAO 0-100 to 0-1
            hit.hit_keyframes = hit_keyframes
            hit.total_keyframes_sent = frames_sent

            # Compute confidence score from similarity + vote count
            vote_ratio = min(hit_keyframes / max(frames_sent, 1), 1.0)
            hit.confidence_score = round(
                (max_similarity / 100.0) * 0.7 + vote_ratio * 0.3, 4
            )

            # If status not set by fetcher, default to located_only
            if not hit.status or hit.status == "miss":
                hit.status = "located_only"

            # Set source_trust if not set by fetcher
            if not hit.source_trust:
                hit.source_trust = _classify_trust(primary_url)

            _report(progress_cb, 100.0, "源回捞完成")

        return [hit]

    # ------------------------------------------------------------------
    # SauceNAO client
    # ------------------------------------------------------------------

    async def _search_saucenao(
        self,
        client: httpx.AsyncClient,
        image_data: bytes,
        api_key: str,
    ) -> list[dict]:
        """POST image to SauceNAO, return list of parsed result dicts."""
        try:
            response = await client.post(
                _SAUCENAO_URL,
                files={"file": ("keyframe.jpg", image_data, "image/jpeg")},
                data={
                    "api_key": api_key,
                    "output_type": "2",   # JSON
                    "numres": "5",
                },
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as e:
            logger.warning("SauceNAO HTTP 请求失败: %s", e)
            return []
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("SauceNAO 响应解析失败: %s", e)
            return []

        if body.get("header", {}).get("status", -1) != 0:
            logger.warning("SauceNAO 返回非零状态: %s", body.get("header", {}))
            return []

        results = body.get("results", [])
        parsed: list[dict] = []

        for r in results:
            header = r.get("header", {})
            data = r.get("data", {})

            similarity_str = header.get("similarity", "0")
            try:
                similarity = float(similarity_str)
            except (ValueError, TypeError):
                similarity = 0.0

            ext_urls = data.get("ext_urls", [])
            url = ext_urls[0] if ext_urls else ""

            parsed.append({
                "similarity": similarity,
                "url": url,
                "title": data.get("title", ""),
                "source_site": data.get("source", ""),
                "index_name": header.get("index_name", ""),
            })

        return parsed

    # ------------------------------------------------------------------
    # Early-stop logic
    # ------------------------------------------------------------------

    def _should_early_stop(self, hits: list[dict]) -> bool:
        """Return True if >=2 frames matched the same URL with >90 % similarity."""
        from collections import Counter

        high_conf_urls: list[str] = []
        for h in hits:
            if h.get("similarity", 0) > _EARLY_STOP_SIMILARITY_THRESHOLD:
                url = h.get("url", "")
                if url:
                    high_conf_urls.append(url)

        if not high_conf_urls:
            return False

        url_counts = Counter(high_conf_urls)
        return max(url_counts.values()) >= _EARLY_STOP_VOTE_COUNT

    # ------------------------------------------------------------------
    # Hit aggregation
    # ------------------------------------------------------------------

    def _aggregate_hits(self, saucenao_hits: list[dict]) -> dict:
        """Group SauceNAO hits by source URL, pick the most-voted.

        Returns:
            dict with keys: url, vote_count, max_similarity, title
        """
        from collections import defaultdict

        url_groups: dict[str, list[dict]] = defaultdict(list)
        for h in saucenao_hits:
            url = h.get("url", "")
            if url:
                url_groups[url].append(h)

        if not url_groups:
            # Fallback: return first hit as-is
            first = saucenao_hits[0] if saucenao_hits else {}
            return {
                "url": first.get("url", ""),
                "vote_count": 1,
                "max_similarity": first.get("similarity", 0.0),
                "title": first.get("title", ""),
            }

        # Pick URL with most votes; break ties with max similarity
        best_url = ""
        best_count = 0
        best_similarity = 0.0
        best_title = ""

        for url, hits in url_groups.items():
            count = len(hits)
            max_sim = max(h.get("similarity", 0.0) for h in hits)
            if count > best_count or (count == best_count and max_sim > best_similarity):
                best_url = url
                best_count = count
                best_similarity = max_sim
                best_title = hits[0].get("title", "")

        return {
            "url": best_url,
            "vote_count": best_count,
            "max_similarity": best_similarity,
            "title": best_title,
        }

    # ------------------------------------------------------------------
    # Router: dispatch to Civitai / comfyworkflows / other
    # ------------------------------------------------------------------

    async def _route_and_fetch(
        self,
        client: httpx.AsyncClient,
        url: str,
        progress_cb: Optional[ProgressCallback],
    ) -> SourceRecoveryHit:
        """Route *url* to the correct metadata fetcher and return a hit."""
        if "civitai.com" in url:
            return await self._fetch_civitai_meta(client, url, progress_cb)
        elif "comfyworkflows.com" in url:
            return await self._scrape_comfyworkflows(client, url, progress_cb)
        else:
            _report(progress_cb, 70.0, "非 Civitai/ComfyWorkflows 源，仅定位")
            return SourceRecoveryHit(
                status="located_only",
                source_url=url,
            )

    # ------------------------------------------------------------------
    # Civitai API client
    # ------------------------------------------------------------------

    async def _fetch_civitai_meta(
        self,
        client: httpx.AsyncClient,
        url: str,
        progress_cb: Optional[ProgressCallback],
    ) -> SourceRecoveryHit:
        """Fetch image metadata + optional workflow from Civitai API."""
        image_id = _extract_civitai_id(url)
        if not image_id:
            logger.warning("无法从 URL 提取 Civitai 图片 ID: %s", url)
            return SourceRecoveryHit(
                status="located_only",
                source_url=url,
            )

        _report(progress_cb, 55.0, f"查询 Civitai 图片 {image_id}...")

        # -- Fetch image metadata -----------------------------------------
        try:
            img_resp = await client.get(
                f"{_CIVITAI_API_BASE}/images/{image_id}",
            )
            img_resp.raise_for_status()
            img_data = img_resp.json()
        except httpx.HTTPError as e:
            logger.warning("Civitai API 请求失败 (图片 %s): %s", image_id, e)
            return SourceRecoveryHit(
                status="located_only",
                source_url=url,
            )
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("Civitai API 响应解析失败: %s", e)
            return SourceRecoveryHit(
                status="located_only",
                source_url=url,
            )

        # Parse image metadata
        meta = img_data.get("meta", {}) or {}
        prompt = meta.get("prompt")
        seed = meta.get("seed")
        sampler = meta.get("sampler")
        steps = meta.get("steps")
        cfg_scale = meta.get("cfgScale")

        # Model info — prefer nested model object, fallback to meta
        model_name = None
        model_obj = img_data.get("model", {}) or {}
        if model_obj:
            model_name = model_obj.get("name")
        if not model_name:
            model_name = meta.get("Model") or meta.get("model")

        # -- Fetch workflow if available ----------------------------------
        _report(progress_cb, 65.0, f"检查 Civitai 图片 {image_id} 工作流...")
        workflow_json: Optional[str] = None

        try:
            wf_resp = await client.get(
                f"{_CIVITAI_API_BASE}/images/{image_id}/workflows",
            )
            wf_resp.raise_for_status()
            wf_data = wf_resp.json()
            items = wf_data.get("items", [])
            if items:
                wf_item = items[0]
                wf_raw = wf_item.get("workflow")
                if wf_raw:
                    workflow_json = json.dumps(wf_raw, ensure_ascii=False)
        except httpx.HTTPError as e:
            logger.info("Civitai 工作流查询失败 (图片 %s): %s", image_id, e)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("Civitai 工作流响应解析失败: %s", e)

        # -- Classify status ----------------------------------------------
        has_params = prompt or seed is not None or sampler or steps
        if workflow_json and has_params:
            status: HitStatus = "complete_match"
        elif has_params:
            status = "partial_match"
        elif workflow_json:
            status = "partial_match"
        else:
            status = "located_only"

        _report(progress_cb, 70.0, f"Civitai 元数据获取完成: {status}")

        return SourceRecoveryHit(
            status=status,
            source_url=url,
            workflow_json=workflow_json,
            prompt=prompt,
            seed=seed,
            sampler=sampler,
            steps=steps,
            cfg_scale=cfg_scale,
            model_name=model_name,
            source_trust="civitai",
        )

    # ------------------------------------------------------------------
    # comfyworkflows scraper
    # ------------------------------------------------------------------

    async def _scrape_comfyworkflows(
        self,
        client: httpx.AsyncClient,
        url: str,
        progress_cb: Optional[ProgressCallback],
    ) -> SourceRecoveryHit:
        """Scrape comfyworkflows.com page for embedded workflow JSON."""
        _report(progress_cb, 55.0, "抓取 comfyworkflows 页面...")

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
        except httpx.HTTPError as e:
            logger.warning("comfyworkflows 页面抓取失败: %s", e)
            return SourceRecoveryHit(
                status="located_only",
                source_url=url,
            )

        workflow_json = _extract_workflow_from_html(html)

        if workflow_json:
            _report(progress_cb, 70.0, "comfyworkflows 工作流提取成功")
            return SourceRecoveryHit(
                status="partial_match",
                source_url=url,
                workflow_json=workflow_json,
                source_trust="comfyworkflows",
            )
        else:
            logger.warning("comfyworkflows 页面未找到工作流 JSON: %s", url)
            return SourceRecoveryHit(
                status="located_only",
                source_url=url,
            )


# ====================================================================
# Module-level helpers
# ====================================================================


def _report(
    cb: Optional[ProgressCallback],
    pct: float,
    msg: str,
) -> None:
    """Fire progress callback if provided."""
    if cb:
        cb("saucenao_router", pct, msg)


def _extract_civitai_id(url: str) -> Optional[str]:
    """Extract Civitai image ID from a URL like civitai.com/images/12345."""
    m = _CIVITAI_ID_RE.search(url)
    return m.group(1) if m else None


def _classify_trust(url: str) -> str:
    """Classify source trustworthiness based on domain."""
    if "civitai.com" in url:
        return "civitai"
    if "comfyworkflows.com" in url:
        return "comfyworkflows"
    return "需人工核验"


def _extract_workflow_from_html(html: str) -> Optional[str]:
    """Try to extract embedded workflow JSON from a comfyworkflows page.

    Attempts multiple extraction strategies:
    1. ``window.__WORKFLOW__`` JS global
    2. ``data-workflow`` attribute on a container element
    3. Raw JSON-LD script tag
    """
    # Strategy 1: window.__WORKFLOW__ = {...};
    m = re.search(
        r"window\.__WORKFLOW__\s*=\s*(\{.*?\})\s*;",
        html,
        re.DOTALL,
    )
    if m:
        try:
            parsed = json.loads(m.group(1))
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    # Strategy 2: data-workflow="..." attribute
    m = re.search(
        r"data-workflow\s*=\s*['\"](.+?)['\"]",
        html,
        re.DOTALL,
    )
    if m:
        raw = m.group(1)
        # The value may be HTML-entity-encoded JSON; try to decode
        import html as _html
        decoded = _html.unescape(raw)
        try:
            parsed = json.loads(decoded)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            # Return raw string if it looks like JSON
            if decoded.strip().startswith("{"):
                return decoded

    # Strategy 3: JSON-LD script tag
    m = re.search(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    if m:
        try:
            parsed = json.loads(m.group(1))
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    return None
