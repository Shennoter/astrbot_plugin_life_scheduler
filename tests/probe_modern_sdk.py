"""Real SDK image pipeline and forward traversal; transport/model are substitutes.

Usage: python3 tests/probe_modern_sdk.py SDK_ROOT LEGACY_SDK_ROOT
Loads unchanged SDK media utilities, request assembly and image conversion.
"""

import argparse
import asyncio
import base64
import dataclasses
import io
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

from PIL import Image as PillowImage
from probe_quoted_sdk import install_sdk, load_method, load_source
from test_life_context import _load_main


def install_modern(module, sdk, legacy, root):
    install_sdk(module, sdk, legacy, root)
    path_module = types.ModuleType("astrbot.core.utils.astrbot_path")
    path_module.get_astrbot_temp_path = lambda: str(root / "temp")
    io_module = types.ModuleType("astrbot.core.utils.io")

    async def unexpected(*args, **kwargs):
        raise AssertionError("unexpected network/audio operation")

    io_module.download_file = unexpected
    io_module.DownloadFileHTTPError = type("DownloadFileHTTPError", (Exception,), {})
    audio = types.ModuleType("astrbot.core.utils.tencent_record_helper")
    audio.tencent_silk_to_wav = audio.wav_to_tencent_silk = unexpected
    for item in (path_module, io_module, audio):
        sys.modules[item.__name__] = item
    datetime_path = sdk / "astrbot/core/utils/datetime_utils.py"
    if datetime_path.exists():
        load_source("astrbot.core.utils.datetime_utils", datetime_path)
    media = load_source(
        "astrbot.core.utils.media_utils", sdk / "astrbot/core/utils/media_utils.py"
    )
    module.Image.convert_to_file_path = load_method(
        sdk / "astrbot/core/message/components.py",
        "Image",
        "convert_to_file_path",
        {"MediaResolver": media.MediaResolver},
    )
    request = load_source(
        "astrbot.core.provider.entities",
        sdk / "astrbot/core/provider/entities.py",
        functions={"ProviderRequest"},
        namespace={
            **{
                name: getattr(media, name)
                for name in (
                    "resolve_image_ref_to_base64_data",
                    "is_recoverable_image_error",
                )
                if hasattr(media, name)
            },
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "MediaResolver": media.MediaResolver,
            "logger": module.logger,
            # SDK annotations are evaluated lazily via this module's future flag.
        },
    )
    return request.ProviderRequest


async def run_case(
    module, request_type, root, *, layout, direct=0, large=False, count=32, inline=False
):
    payload = io.BytesIO()
    PillowImage.new("RGB", (4096, 4096) if large else (1200, 1600), "white").save(
        payload, format="BMP" if large else "PNG"
    )
    raw = payload.getvalue()
    encoded = base64.b64encode(raw).decode()
    mime = "image/bmp" if large else "image/png"
    expected_url = f"data:{mime};base64,{encoded}"
    expected = min(count + direct, 4)
    # Four distinct paths preserve the collector's existing deduplication rules.
    paths = []
    for i in range(4):
        path = root / f"source-{i}.{'bmp' if large else 'png'}"
        path.write_bytes(raw)
        paths.append(str(path))
    calls = []
    loop_thread = threading.get_ident()

    def image_segment(index):
        return {"type": "image", "data": {"file": f"image-{index}"}}

    def forward_segment(index):
        return {"type": "forward", "data": {"id": str(index)}}

    async def call_action(action, **params):
        assert threading.get_ident() == loop_thread
        calls.append(action)
        await asyncio.sleep(0)
        if action == "get_msg":
            if layout == "fanout":
                segments = [forward_segment(i) for i in range(32)]
            elif layout == "chain":
                segments = [forward_segment(0)]
            else:
                segments = [image_segment(i) for i in range(count)]
                if layout == "direct":
                    segments += [forward_segment(i) for i in range(32)]
            return {"data": {"message": segments}}
        if action == "get_forward_msg":
            index = int(params.get("message_id", params.get("id")))
            segments = [image_segment(index)]
            if layout == "chain":
                segments.append(forward_segment((index + 1) % 32))
            return {"data": {"messages": [{"content": segments}]}}
        assert action == "get_image"
        index = int(params["file"].split("-")[-1])
        return {
            "data": {
                "url": "base64://" + encoded if inline else paths[(index + direct) % 4]
            }
        }

    reply = module.Reply()
    reply.id, reply.chain, reply.message_str = "123", [], ""
    event = types.SimpleNamespace(
        message_obj=types.SimpleNamespace(
            message=[module.Image(file=p) for p in paths[:direct]] + [reply]
        ),
        bot=types.SimpleNamespace(api=types.SimpleNamespace(call_action=call_action)),
        get_group_id=lambda: "456",
        session_id="modern-sdk-probe",
        plain_result=lambda text: text,
    )
    provider_counts = []

    async def text_chat(prompt, *, image_urls, **kwargs):
        assert len(image_urls) == expected
        assert all(
            not ref.startswith(("base64://", "data:")) and Path(ref).is_file()
            for ref in image_urls
        ), "modern provider must receive local paths without preencoding"
        assembled = await request_type(
            prompt=prompt, image_urls=image_urls
        ).assemble_context()
        images = [p for p in assembled["content"] if p["type"] == "image_url"]
        assert len(images) == expected
        assert all(p["image_url"]["url"] == expected_url for p in images)
        provider_counts.append(len(images))
        return types.SimpleNamespace(completion_text='{"clothing":"白衬衫"}')

    plugin = object.__new__(module.LifeSchedulerPlugin)
    plugin.config = {}
    plugin.wardrobe_mgr = module.WardrobeDataManager(root / "wardrobe.json")
    plugin.data_mgr = module.ScheduleDataManager(root / "schedule.json")
    context = types.SimpleNamespace(
        get_using_provider=lambda: types.SimpleNamespace(text_chat=text_chat)
    )
    plugin.generator = module.SchedulerGenerator(context, {}, plugin.data_mgr)
    reads, writes, encodes = [], [], []
    forward_parses = []
    payload_parser = sys.modules[
        "astrbot.core.utils.quoted_message.chain_parser"
    ].OneBotPayloadParser
    parse_forward = payload_parser.parse_get_forward_payload

    def parse(parser, payload):
        forward_parses.append(threading.get_ident())
        return parse_forward(parser, payload)

    read_bytes, write_bytes, b64encode = (
        Path.read_bytes,
        Path.write_bytes,
        base64.b64encode,
    )

    def read(path):
        data = read_bytes(path)
        reads.append((len(data), threading.get_ident()))
        return data

    def write(path, data):
        writes.append(len(data))
        return write_bytes(path, data)

    def encode(data, *args, **kwargs):
        encodes.append(len(data))
        return b64encode(data, *args, **kwargs)

    running, gaps = True, []

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
        with (
            patch.object(Path, "read_bytes", read),
            patch.object(Path, "write_bytes", write),
            patch.object(base64, "b64encode", encode),
            patch.object(payload_parser, "parse_get_forward_payload", parse),
        ):
            replies = [result async for result in plugin.wardrobe_add(event, "")]
    finally:
        running = False
        await beat
    expected_forward = 4 - direct if layout in {"chain", "fanout"} else 0
    stats = {
        "layout": layout,
        "direct": direct,
        "input_count": count,
        "inline": inline,
        "image_bytes": len(raw),
        "forward_fetches": calls.count("get_forward_msg"),
        "forward_parses": len(forward_parses),
        "expected_forward_fetches": expected_forward,
        "get_image": calls.count("get_image"),
        "reads": len(reads),
        "writes": len(writes),
        "written_bytes": sum(writes),
        "encodes": len(encodes),
        "encoded_bytes": sum(encodes),
        "max_heartbeat_gap_s": round(max(gaps), 4),
    }
    print(stats, flush=True)
    assert replies[-1].startswith("已放进衣柜"), replies
    assert provider_counts == [expected]
    assert calls.count("get_forward_msg") == expected_forward, stats
    assert len(forward_parses) == expected_forward, stats
    assert all(thread != loop_thread for thread in forward_parses), stats
    assert calls.count("get_image") == expected - direct, stats
    assert len(writes) == (expected if inline else 0), stats
    assert len(reads) == len(encodes) == expected, stats
    assert all(thread != loop_thread for _, thread in reads)
    assert max(gaps) < 1.0, stats
    reloaded = module.WardrobeDataManager(root / "wardrobe.json")
    assert reloaded.all() == plugin.wardrobe_mgr.all()


async def check_injection(module, sdk, request_type):
    message = load_source(
        "astrbot.core.agent.message", sdk / "astrbot/core/agent/message.py"
    )
    plugin = object.__new__(module.LifeSchedulerPlugin)
    plugin.config = {"inject_as_temp": True, "schedule_time": "00:00"}
    plugin.data_mgr = types.SimpleNamespace(
        get=lambda _: types.SimpleNamespace(
            status="ok", outfit="白衬衫", schedule="09:00 阅读"
        )
    )
    plugin.wardrobe_mgr = types.SimpleNamespace(for_prompt=lambda **_: "1. 蓝色裙子")
    req = request_type(prompt="现在做什么？", system_prompt="原有人设")
    req.extra_user_content_parts.append(message.TextPart(text="其他插件内容"))
    event = types.SimpleNamespace(unified_msg_origin="test:FriendMessage:1")
    await plugin.on_llm_request(event, req)
    runtime = message.Message.model_validate(await req.assemble_context())
    assert req.system_prompt == "原有人设"
    assert "白衬衫" in str(runtime.model_dump())
    assert "<life_wardrobe>" in str(runtime.model_dump())
    assert message.dump_messages_with_checkpoints([runtime]) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "现在做什么？"},
                {"type": "text", "text": "其他插件内容"},
            ],
        }
    ]
    del plugin.config["inject_as_temp"]
    default_req = request_type(prompt="test", system_prompt="原有人设")
    await plugin.on_llm_request(event, default_req)
    assert "白衬衫" in default_req.system_prompt
    assert default_req.extra_user_content_parts == []
    print("PASS temporary injection/history removal and default system injection")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", type=Path)
    parser.add_argument("legacy_sdk", type=Path)
    parser.add_argument("--case", choices=("forwards", "large", "all"), default="all")
    args = parser.parse_args()
    module = _load_main()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "temp").mkdir()
        request = install_modern(module, args.sdk, args.legacy_sdk, root)
        await check_injection(module, args.sdk, request)
        if args.case in {"forwards", "all"}:
            for layout in ("fanout", "chain", "direct"):
                for direct in (0, 2, 4):
                    await run_case(module, request, root, layout=layout, direct=direct)
        if args.case in {"large", "all"}:
            await run_case(module, request, root, layout="direct", direct=4, large=True)
            for count, inline in (
                (1, False),
                (3, False),
                (1, True),
                (3, True),
                (32, True),
            ):
                await run_case(
                    module,
                    request,
                    root,
                    layout="images",
                    count=count,
                    inline=inline,
                    large=True,
                )
    print(
        "PASS actual SDK pipeline; OneBot transport and remote model are substitutes."
    )


if __name__ == "__main__":
    asyncio.run(main())
