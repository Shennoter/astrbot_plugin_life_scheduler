"""Verify injection/history serialization with real SDK code and no model calls.

Usage: python tests/probe_prompt_cache.py SDK_ROOT [--group-chat-plus FORK_ROOT]
"""

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from probe_quoted_sdk import load_source
from test_life_context import _load_main


async def check(sdk, group_chat_plus=None):
    module = _load_main()
    message = load_source(
        "astrbot.core.agent.message", sdk / "astrbot/core/agent/message.py"
    )
    entities = load_source(
        "astrbot.core.provider.entities",
        sdk / "astrbot/core/provider/entities.py",
        functions={"ProviderRequest"},
        namespace={"dataclass": dataclasses.dataclass, "field": dataclasses.field},
    )
    plugin = object.__new__(module.LifeSchedulerPlugin)
    plugin.config = {"schedule_time": "00:00"}
    data = types.SimpleNamespace(
        status="ok", outfit="白衬衫", schedule="09:00 阅读\n14:00 散步"
    )
    plugin.data_mgr = types.SimpleNamespace(get=lambda _: data)
    plugin.wardrobe_mgr = types.SimpleNamespace(for_prompt=lambda **_: "1. 蓝色裙子")
    event = types.SimpleNamespace(unified_msg_origin="test:GroupMessage:1")
    prompts = []
    for outfit in ("白衬衫", "红色外套"):
        data.outfit = outfit
        req = entities.ProviderRequest(
            prompt="历史对话\n现在做什么？", system_prompt="固定人格"
        )
        req.extra_user_content_parts.append(message.TextPart(text="其他插件内容"))
        await plugin.on_llm_request(event, req)
        runtime = message.Message.model_validate(await req.assemble_context())
        assert outfit in str(runtime.model_dump())
        saved = message.dump_messages_with_checkpoints([runtime])
        assert outfit not in str(saved)
        assert "<life_wardrobe>" not in str(saved)
        assert "其他插件内容" in str(saved)
        assert "历史对话" in str(saved)
        prompts.append(req.system_prompt)
    assert prompts[0] == prompts[1] == "固定人格" + module.LIFE_CONTEXT_RULES
    print(
        "PASS real SDK: current state reaches request and is removed from saved history"
    )

    if group_chat_plus:
        hook_path = group_chat_plus / "tests/test_prompt_cache_prefix.py"
        hook_spec = importlib.util.spec_from_file_location(
            "gcp_final_hook_probe", hook_path
        )
        hook_tests = importlib.util.module_from_spec(hook_spec)
        hook_spec.loader.exec_module(hook_tests)
        with pytest.MonkeyPatch.context() as patch:
            gcp = hook_tests.plugin.__wrapped__(patch)
            gcp_event = hook_tests.Event()
            reply = gcp.reply
            gcp_event.set_extra(reply.PLUGIN_REQUEST_MARKER, True)
            gcp_event.set_extra(reply.PLUGIN_CUSTOM_SYSTEM_PROMPT, "固定人格")
            gcp_event.set_extra(reply.PLUGIN_CUSTOM_STATIC_INSTRUCTIONS, "固定群聊规则")
            gcp_event.set_extra(reply.PLUGIN_CUSTOM_PROMPT, req.prompt)
            parts = req.extra_user_content_parts
            await gcp.prepare(gcp_event, req)
            assert module.LIFE_CONTEXT_RULES in req.system_prompt
            assert "红色外套" not in req.system_prompt
            assert req.extra_user_content_parts is parts
        print(
            "PASS final group-chat hook preserves fixed life rules and temporary parts"
        )
        path = group_chat_plus / "tests/test_proactive_request_parts.py"
        spec = importlib.util.spec_from_file_location("gcp_proactive_probe", path)
        boundary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(boundary)
        for relative in (
            "utils/proactive_chat_manager.py",
            "private_chat/private_chat_utils/private_chat_proactive_chat_manager.py",
        ):
            invoke = boundary.invoke.__wrapped__(types.SimpleNamespace(param=relative))
            provider = types.SimpleNamespace(text_chat=AsyncMock())
            await invoke(provider, req)
            kwargs = provider.text_chat.call_args.kwargs
            assert kwargs["extra_user_content_parts"] is req.extra_user_content_parts
            provider_request = entities.ProviderRequest(
                prompt=kwargs["prompt"],
                system_prompt=kwargs["system_prompt"],
                extra_user_content_parts=kwargs["extra_user_content_parts"],
            )
            payload = await provider_request.assemble_context()
            assert "红色外套" in json.dumps(payload, ensure_ascii=False)
            print("PASS proactive request forwards real temporary context:", relative)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", type=Path)
    parser.add_argument("--group-chat-plus", type=Path)
    args = parser.parse_args()
    asyncio.run(check(args.sdk, args.group_chat_plus))
