"""MiniMax image generation backend.

Backs the MiniMax 开放平台 image-01 model (and any future additions) via
the public ``https://api.minimaxi.com/v1/image_generation`` endpoint.

The provider accepts ``MINIMAX_API_KEY`` (preferred) or the legacy
``MINIMAX_CN_API_KEY`` env var; the latter is a no-op alias so existing
configs don't break — both keys come from
https://platform.minimaxi.com/user-center/basic-information/interface-key.

Note: this key is the **开放平台** key (api.minimaxi.com), separate from
the **chat** key (api.minimax.chat / MINIMAX_API_KEY on the agent loop).
On well-configured installs both endpoints share one key; on installs
where they don't, image generation will surface ``auth_required`` and the
user can set ``MINIMAX_PLATFORM_KEY`` (or ``MINIMAX_IMAGE_KEY``) as a
provider-specific override.

Selection precedence for the key:
    1. ``MINIMAX_IMAGE_KEY`` env var (escape hatch — same value can be
       exported at the OS level for ad-hoc testing without touching .env)
    2. ``image_gen.minimax.api_key`` in config.yaml (provider-scoped)
    3. ``MINIMAX_API_KEY`` env var (default — chat key, often reusable)
    4. ``MINIMAX_CN_API_KEY`` env var (last-ditch alias)

Aspect-ratio routing is different from the FAL/OpenAI backends:
``image-01`` accepts literal strings like ``"16:9"``, ``"9:16"``, ``"1:1"``
instead of the abstract ``"landscape"`` / ``"portrait"`` / ``"square"`` the
ABC exposes. We translate in ``_ASPECT_TO_LITERAL``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://api.minimaxi.com"
GENERATION_ENDPOINT = f"{API_BASE}/v1/image_generation"

# Single known image model as of 2026-07. MiniMax 平台 exposes image-01
# (their production text-to-image model). Other ids may follow; we keep
# the catalog here so `hermes tools` shows them once announced.
MODELS: Dict[str, Dict[str, Any]] = {
    "image-01": {
        "display": "MiniMax image-01",
        "speed": "~10-30s",
        # Honest tag — image-01 is solid for anime / illustration / concept
        # art. It's not Flux-Kontext, so we don't claim photorealistic SOTA.
        "strengths": "Anime, illustration, character art; CN/EN bilingual",
        # Pricing is per MiniMax 开放平台 — confirm at
        # https://platform.minimaxi.com/docs/pricing/overview
        "price": "see platform pricing",
        # image-01 supports a literal aspect_ratio enum, not FAL-style presets.
        # Default 1:1 output, supporting 1:1 / 16:9 / 9:16 / 4:3 / 3:4 / etc.
        "aspect_literals": ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"],
        "supports_edit": False,  # MiniMax has a separate 图生图 endpoint
    },
}

DEFAULT_MODEL = "image-01"

# Translate abstract aspect → MiniMax literal. image-01 only supports a
# handful of ratios so we snap to the closest valid value rather than
# erroring out on unsupported shapes.
_ASPECT_TO_LITERAL: Dict[str, str] = {
    "square": "1:1",
    "landscape": "16:9",
    "portrait": "9:16",
}


# ---------------------------------------------------------------------------
# Config / key resolution
# ---------------------------------------------------------------------------


def _load_minimax_config() -> Dict[str, Any]:
    """Read the ``image_gen.minimax`` section from config.yaml (returns {} on failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            minimax_cfg = section.get("minimax")
            return minimax_cfg if isinstance(minimax_cfg, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.minimax config: %s", exc)
    return {}


def _resolve_api_key() -> Optional[str]:
    """Resolve the MiniMax image-gen API key with the documented precedence chain.

    Returns ``None`` when nothing usable is configured so the provider can
    surface a clean ``auth_required`` error pointing the user at setup.
    """
    # 1. Provider-specific env override (escape hatch for testing / cron).
    override = os.environ.get("MINIMAX_IMAGE_KEY")
    if override and override.strip():
        return override.strip()

    # 2. config.yaml: image_gen.minimax.api_key
    cfg = _load_minimax_config()
    cfg_key = cfg.get("api_key")
    if isinstance(cfg_key, str) and cfg_key.strip():
        return cfg_key.strip()

    # 3. Default env var (the same key the chat provider uses — works when
    # the user has a unified MiniMax 开放平台 account).
    env_key = os.environ.get("MINIMAX_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 4. Legacy alias from the MINIMAX_CN provider slot.
    cn_key = os.environ.get("MINIMAX_CN_API_KEY")
    if cn_key and cn_key.strip():
        return cn_key.strip()

    return None


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Pick the active model id, honoring env override → config → default."""
    env_override = os.environ.get("MINIMAX_IMAGE_MODEL")
    if env_override and env_override in MODELS:
        return env_override, MODELS[env_override]

    cfg = _load_minimax_config()
    candidate = cfg.get("model")
    if isinstance(candidate, str) and candidate in MODELS:
        return candidate, MODELS[candidate]

    return DEFAULT_MODEL, MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


def _http_post_json(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: float = 120.0) -> Dict[str, Any]:
    """POST JSON to ``url`` and return the parsed response.

    Raises ``urllib.error.HTTPError`` on non-2xx; callers surface the body
    in the error_response. Network errors surface as ``urllib.error.URLError``.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax 开放平台 image-01 backend."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        # API key is the only external dependency; no SDK install required.
        return _resolve_api_key() is not None

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
            }
            for model_id, meta in MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "paid",
            "tag": "image-01 — text-to-image via MiniMax 开放平台 (api.minimaxi.com)",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax 开放平台 API key (image-01 用)",
                    "url": "https://platform.minimaxi.com/user-center/basic-information/interface-key",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # image-01 is text-to-image only here. MiniMax has a separate
        # 图生图 endpoint we don't bridge yet — when we do, set
        # "max_reference_images" to the platform's cap (currently 1).
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )

        # Routing — image-01 is text-only here. If the caller asks for
        # image-to-image, refuse cleanly rather than silently dropping the
        # source image (which would generate something the user didn't
        # intend).
        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            sources.append(ref)
        if sources:
            return error_response(
                error=(
                    "MiniMax image-01 (this provider) only supports text-to-image. "
                    "Drop image_url / reference_image_urls, or switch backends via "
                    "`hermes tools` → Image Generation."
                ),
                error_type="modality_unsupported",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        api_key = _resolve_api_key()
        if not api_key:
            return error_response(
                error=(
                    "No MiniMax API key configured. Set MINIMAX_API_KEY in "
                    "~/.hermes/.env (preferred) or run `hermes tools` → Image "
                    "Generation → MiniMax. Get a key at "
                    "https://platform.minimaxi.com/user-center/basic-information/interface-key"
                ),
                error_type="auth_required",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        model_id, _ = _resolve_model()
        aspect_literal = _ASPECT_TO_LITERAL.get(aspect, "1:1")

        # ``prompt_optimizer`` enriches short prompts server-side; off by
        # default so the cost/behavior matches what the user typed. Users
        # who want it can flip it on in config.yaml.
        cfg = _load_minimax_config()
        prompt_optimizer = bool(cfg.get("prompt_optimizer", False))

        # Number of images — capped server-side, default to 1 to match
        # other backends. Allow override via ``image_gen.minimax.n``.
        n = cfg.get("n", 1)
        try:
            n = max(1, min(int(n), 4))  # platform doesn't document a hard cap, but keep sane
        except (TypeError, ValueError):
            n = 1

        body: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "aspect_ratio": aspect_literal,
            "response_format": "url",
            "n": n,
            "prompt_optimizer": prompt_optimizer,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = _http_post_json(GENERATION_ENDPOINT, body=body, headers=headers, timeout=120.0)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — best-effort error context
                pass
            logger.warning("MiniMax image generation HTTP %s: %s", exc.code, detail)
            return error_response(
                error=f"MiniMax API returned HTTP {exc.code}: {detail or exc.reason}",
                error_type=f"http_{exc.code}",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except urllib.error.URLError as exc:
            logger.warning("MiniMax image generation URL error: %s", exc)
            return error_response(
                error=f"Could not reach MiniMax API: {exc.reason}",
                error_type="network_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.warning("MiniMax image generation failed", exc_info=True)
            return error_response(
                error=f"MiniMax image generation failed: {exc}",
                error_type=type(exc).__name__,
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Response shape (from the public docs):
        # {
        #   "id": "<request-id>",
        #   "data": {"image_urls": ["https://...signed-oss-url..."]},
        #   "metadata": {"success_count": "N", "failed_count": "0"},
        #   "base_resp": {"status_code": 0, "status_msg": "success"}
        # }
        # image_urls are signed OSS URLs — 24h expiry. We cache them
        # locally to dodge the TTL during gateway delivery (same rationale
        # as the xAI provider, see save_url_image()).
        base_resp = response.get("base_resp") or {}
        if base_resp.get("status_code", 0) != 0:
            return error_response(
                error=f"MiniMax API error: {base_resp.get('status_msg', 'unknown')}",
                error_type="api_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data_section = response.get("data") or {}
        urls = data_section.get("image_urls") or []
        if not urls:
            return error_response(
                error="MiniMax returned no image_urls in response",
                error_type="empty_response",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # We always report the first URL as the canonical ``image`` so
        # the tool contract is single-valued. Cache it locally to dodge
        # the 24h OSS expiry; fall back to the bare URL if the cache
        # write fails (network blip on this host, etc.).
        first_url = urls[0]
        try:
            cached_path = save_url_image(first_url, prefix=f"minimax_{model_id}")
            image_ref = str(cached_path)
        except Exception as exc:
            logger.warning("Could not cache MiniMax image locally (%s); using bare URL.", exc)
            image_ref = first_url

        metadata = response.get("metadata") or {}
        extra: Dict[str, Any] = {
            "aspect_literal": aspect_literal,
            "n_returned": len(urls),
            "request_id": response.get("id"),
        }
        if metadata:
            extra["minimax_metadata"] = metadata

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality="text",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())