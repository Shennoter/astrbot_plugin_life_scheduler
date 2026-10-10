"""Bounded image input with blocking SDK media work kept off the event loop."""

import asyncio
import base64
from itertools import chain
from pathlib import Path

from astrbot.api import logger


async def resolve_image_path(component) -> str:
    # SDK converters may decode base64 and write files before their first await.
    # Their HTTP sessions are created inside this worker's loop; no bot/client
    # objects from the main loop are passed into it.
    def convert():
        return asyncio.run(component.convert_to_file_path())

    return await asyncio.to_thread(convert)


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
    # The SDK's public images() resolves the complete batch. Fetch metadata
    # separately so the caller can stop before resolving its fifth image.
    content = await extractor._fetch_quoted_content(reply, fetch_remote=True)
    if content is None:
        return
    seen = set()
    for ref in chain(
        content.embedded_image_refs,
        content.direct_image_refs,
        content.forward_image_refs,
    ):
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
    """Encode at most four local images before the provider's synchronous I/O."""
    from astrbot.api.message_components import Image

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
