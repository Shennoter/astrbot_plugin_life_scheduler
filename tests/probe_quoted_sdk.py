"""Exercise actual SDK quoted-message parsing with a fake OneBot transport.

Usage: python3 tests/probe_quoted_sdk.py SDK_ROOT LEGACY_SDK_ROOT
SDK_ROOT contains astrbot/core/utils/quoted_message; LEGACY_SDK_ROOT supplies
the original 4.10.4 image converter and provider encoder. No service is started.
"""

import argparse
import ast
import asyncio
import base64
import hashlib
import io
import os
import sys
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlsplit
from urllib.request import url2pathname

from PIL import Image as PillowImage
from test_life_context import _load_main


def load_source(name, path, *, functions=None, namespace=None):
    source = path.read_text()
    print(path.name, "sha256", hashlib.sha256(source.encode()).hexdigest())
    module = types.ModuleType(name)
    module.__dict__.update(namespace or {})
    sys.modules[name] = module
    tree = ast.parse(source)
    if functions:
        tree.body = [
            node
            for node in tree.body
            if getattr(node, "name", None) in functions
            or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 - local SDK source
    return module


def load_method(path, cls_name, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if getattr(node, "name", None) == cls_name)
    method = next(node for node in cls.body if getattr(node, "name", None) == name)
    print(name, "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    exec(  # noqa: S102 - execute the unmodified local SDK method
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def install_sdk(module, sdk, legacy, root):
    sys.modules["astrbot"].logger = module.logger
    components = types.ModuleType("astrbot.core.message.components")
    components.Image = module.Image
    components.Reply = module.Reply
    for name in ("At", "AtAll", "File", "Forward", "Node", "Nodes", "Plain", "Video"):
        setattr(components, name, type(name, (), {}))
    sys.modules[components.__name__] = components
    event = types.ModuleType("astrbot.core.platform.astr_message_event")
    event.AstrMessageEvent = object
    sys.modules[event.__name__] = event
    utils = sdk / "astrbot/core/utils"
    load_source(
        "astrbot.core.utils.media_utils",
        utils / "media_utils.py",
        functions={"is_file_uri", "file_uri_to_path"},
        namespace={
            "MediaRefStr": str,
            "urlsplit": urlsplit,
            "urlparse": urlparse,
            "url2pathname": url2pathname,
            "Path": Path,
            "os": os,
        },
    )
    for name in ("image_ref_utils", "string_utils"):
        load_source(f"astrbot.core.utils.{name}", utils / f"{name}.py")
    package_name = "astrbot.core.utils.quoted_message"
    package = types.ModuleType(package_name)
    package.__path__ = [str(utils / "quoted_message")]
    sys.modules[package_name] = package
    for name in (
        "settings",
        "image_refs",
        "chain_parser",
        "onebot_client",
        "image_resolver",
        "extractor",
    ):
        load_source(f"{package_name}.{name}", utils / "quoted_message" / f"{name}.py")
    converter = load_method(
        legacy / "astrbot/core/message/components.py",
        "Image",
        "convert_to_file_path",
        {
            "os": os,
            "base64": base64,
            "uuid": uuid,
            "get_astrbot_data_path": lambda: str(root),
        },
    )
    module.Image.convert_to_file_path = converter
    return load_method(
        legacy / "astrbot/core/provider/entities.py",
        "ProviderRequest",
        "_encode_image_bs64",
        {"base64": base64},
    )


async def run_case(module, root, encoder, *, count, direct=0, forward=False):
    loop_thread = threading.get_ident()
    calls = []
    image_bytes = io.BytesIO()
    PillowImage.new("RGB", (1200, 1600), "white").save(image_bytes, format="PNG")
    encoded = "base64://" + base64.b64encode(image_bytes.getvalue()).decode()
    segments = [{"type": "image", "data": {"file": f"image-{i}"}} for i in range(count)]

    async def call_action(action, **params):
        assert threading.get_ident() == loop_thread
        calls.append(action)
        await asyncio.sleep(0)
        if action == "get_msg":
            message = (
                [{"type": "forward", "data": {"id": "forward-1"}}]
                if forward
                else segments
            )
            return {"data": {"message": message}}
        if action == "get_forward_msg":
            return {"data": {"messages": [{"content": segments}]}}
        assert action == "get_image", (action, params)
        return {"data": {"url": encoded}}

    reply = module.Reply()
    reply.id, reply.chain, reply.message_str = "123", [], ""
    event = types.SimpleNamespace(
        message_obj=types.SimpleNamespace(
            message=[module.Image(file=encoded) for _ in range(direct)] + [reply]
        ),
        bot=types.SimpleNamespace(api=types.SimpleNamespace(call_action=call_action)),
        get_group_id=lambda: "456",
        session_id="sdk-probe",
        plain_result=lambda text: text,
    )
    expected = min(4, count + direct)
    provider_count = []

    async def text_chat(*args, image_urls, **kwargs):
        for ref in image_urls:
            assert ref.startswith("base64://")
            result = await encoder(None, ref)
            assert base64.b64decode(result.split(",", 1)[1]) == image_bytes.getvalue()
        provider_count.append(len(image_urls))
        return types.SimpleNamespace(completion_text='{"clothing":"白衬衫"}')

    plugin = object.__new__(module.LifeSchedulerPlugin)
    plugin.config = {}
    plugin.wardrobe_mgr = module.WardrobeDataManager(root / "wardrobe.json")
    plugin.data_mgr = module.ScheduleDataManager(root / "schedule.json")
    context = types.SimpleNamespace(
        get_using_provider=lambda: types.SimpleNamespace(text_chat=text_chat),
    )
    plugin.generator = module.SchedulerGenerator(context, {}, plugin.data_mgr)
    before_files = len(list((root / "temp").iterdir()))
    gaps = []
    running = True

    async def heartbeat():
        previous = time.perf_counter()
        while running:
            await asyncio.sleep(0.001)
            now = time.perf_counter()
            gaps.append(now - previous)
            previous = now

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    try:
        replies = [result async for result in plugin.wardrobe_add(event, "")]
    finally:
        running = False
        await beat
    assert replies[-1].startswith("已放进衣柜"), replies
    assert provider_count == [expected], provider_count
    assert calls.count("get_image") == min(count, 4 - direct), calls
    assert calls.count("get_msg") == int(direct < 4), calls
    converted = len(list((root / "temp").iterdir())) - before_files
    assert converted == expected, converted
    assert max(gaps) < 0.5, max(gaps)
    print(
        {
            "input_images": count,
            "direct": direct,
            "forward": forward,
            "get_image_calls": calls.count("get_image"),
            "converted_files": converted,
            "provider_images": provider_count[0],
            "max_heartbeat_gap_s": round(max(gaps), 4),
            "saved": True,
        }
    )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", type=Path)
    parser.add_argument("legacy_sdk", type=Path)
    args = parser.parse_args()
    module = _load_main()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "temp").mkdir()
        encoder = install_sdk(module, args.sdk, args.legacy_sdk, root)
        for count, direct, forward in (
            (1, 0, False),
            (3, 0, False),
            (32, 0, False),
            (32, 2, False),
            (32, 4, False),
            (32, 0, True),
        ):
            await run_case(
                module, root, encoder, count=count, direct=direct, forward=forward
            )
    print(
        "Real SDK parsing/conversion/encoding passed; real OneBot reconnect NOT reproduced."
    )


if __name__ == "__main__":
    asyncio.run(main())
