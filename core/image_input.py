"""Bounded image input with blocking SDK media work kept off the event loop."""

import asyncio
import base64
from collections import deque
from pathlib import Path

from astrbot.api import logger


async def resolve_image_path(component) -> str:
    # SDK converters may decode base64 and write files before their first await.
    # Their HTTP sessions are created inside this worker's loop; no bot/client
    # objects from the main loop are passed into it.
    def convert():
        return asyncio.run(component.convert_to_file_path())

    return await asyncio.to_thread(convert)


async def _iter_quoted_image_refs(extractor, reply):
    """Fetch the next forward only after the previous images were consumed."""
    for ref in extractor._reply_parser.extract_image_refs_from_reply_component(reply):
        yield ref
    reply_id = str(getattr(reply, "id", "") or "").strip()
    if not reply_id:
        return
    payload = await extractor._client.get_msg(reply_id)
    if not payload:
        return
    parsed = await asyncio.to_thread(
        extractor._payload_parser.parse_get_msg_payload, payload
    )
    for ref in parsed["image_refs"]:
        yield ref

    pending = deque(parsed["forward_ids"])
    seen = set()
    while pending and len(seen) < extractor._settings.max_forward_fetch:
        forward_id = str(pending.popleft() or "").strip()
        if not forward_id or forward_id in seen:
            continue
        seen.add(forward_id)
        payload = await extractor._client.get_forward_msg(forward_id)
        if not payload:
            continue
        parsed = await asyncio.to_thread(
            extractor._payload_parser.parse_get_forward_payload, payload
        )
        for ref in parsed["image_refs"]:
            yield ref
        pending.extend(parsed["forward_ids"])


async def iter_quoted_image_paths(event, reply):
    """Fetch raw quoted references, resolving each only when requested."""
    from astrbot.api.message_components import Image
    from astrbot.core.utils.quoted_message.extractor import QuotedMessageExtractor
    from astrbot.core.utils.quoted_message.image_refs import (
        get_existing_local_path,
        normalize_image_ref,
    )
    from astrbot.core.utils.quoted_message.image_resolver import (
        _build_image_resolve_actions,
    )

    extractor = QuotedMessageExtractor(event)
    # Both images() and _fetch_quoted_content() traverse every forward eagerly.
    # Keep traversal lazy so the caller's four-image limit stops network I/O too.
    seen = set()
    async for ref in _iter_quoted_image_refs(extractor, reply):
        ref = str(ref or "").strip()
        if not ref or ref in seen:
            continue
        seen.add(ref)
        try:
            normalized = await asyncio.to_thread(normalize_image_ref, ref)
            if not normalized:
                if await asyncio.to_thread(get_existing_local_path, ref):
                    continue
                for action, params in _build_image_resolve_actions(event, ref):
                    # OneBot calls stay on the bot's original event loop.
                    data = await extractor._client.call(action, params)
                    if not isinstance(data, dict):
                        continue
                    for key in ("url", "file"):
                        value = data.get(key)
                        if isinstance(value, str):
                            normalized = await asyncio.to_thread(
                                normalize_image_ref, value
                            )
                            if normalized:
                                break
                    if normalized:
                        break
            if normalized:
                yield await resolve_image_path(Image(file=normalized))
        # SDK converters also raise generic Exception for unavailable attachments.
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to resolve quoted image: %s", type(exc).__name__)


async def prepare_provider_images(image_refs: list[str]) -> list[str]:
    """Let modern SDKs read paths; preencode for legacy synchronous providers."""
    from importlib import import_module

    from astrbot.api.message_components import Image

    # Modern ProviderRequest uses MediaResolver, which reads files in a worker.
    # Passing base64:// here would force an extra decode/write/read/encode cycle.
    entities = import_module("astrbot.core.provider.entities")
    if getattr(entities, "MediaResolver", None) is not None:
        return image_refs[:4]

    def encode(path):
        return "base64://" + base64.b64encode(Path(path).read_bytes()).decode("ascii")

    prepared = []
    for ref in image_refs[:4]:
        if ref.startswith("base64://"):
            prepared.append(ref)
            continue
        path = await resolve_image_path(Image(file=ref))
        prepared.append(await asyncio.to_thread(encode, path))
    return prepared
