import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from test_life_context import _load_main


class _TextPart:
    def __init__(self, *, text):
        self.text = text
        self.temporary = False

    def mark_as_temp(self):
        self.temporary = True
        return self


class PromptInjectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = _load_main()
        self.plugin = object.__new__(self.module.LifeSchedulerPlugin)
        self.plugin.config = {"schedule_time": "00:00"}
        self.data = types.SimpleNamespace(
            status="ok", outfit="白衬衫", schedule="09:00 看书\n14:00 散步"
        )
        self.plugin.data_mgr = Mock()
        self.plugin.data_mgr.get.return_value = self.data
        self.plugin.wardrobe_mgr = Mock()
        self.plugin.wardrobe_mgr.for_prompt.return_value = "1. 蓝色裙子"
        self.plugin.generator = types.SimpleNamespace(
            generate_schedule=AsyncMock(return_value=self.data)
        )
        self.event = types.SimpleNamespace(unified_msg_origin="test:FriendMessage:1")
        message = types.ModuleType("astrbot.core.agent.message")
        message.TextPart = _TextPart
        self.sdk_patch = patch.dict(sys.modules, {message.__name__: message})
        self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)

    def request(self, system_prompt="原有人设"):
        return types.SimpleNamespace(
            system_prompt=system_prompt,
            prompt="今天做什么？",
            contexts=[{"role": "user", "content": "上一轮"}],
            extra_user_content_parts=[_TextPart(text="其他插件内容")],
        )

    async def test_disabled_keeps_legacy_system_injection(self):
        self.plugin.config["inject_as_temp"] = False
        req = self.request()
        await self.plugin.on_llm_request(self.event, req)
        self.assertTrue(req.system_prompt.startswith("原有人设"))
        self.assertIn("白衬衫", req.system_prompt)
        self.assertIn("<life_wardrobe>", req.system_prompt)
        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.plugin.generator.generate_schedule.assert_not_awaited()

    async def test_default_uses_temp_and_keeps_only_fixed_rules_in_system(self):
        req = self.request()
        await self.plugin.on_llm_request(self.event, req)
        self.assertEqual(req.system_prompt, "原有人设" + self.module.LIFE_CONTEXT_RULES)
        self.assertNotIn("白衬衫", req.system_prompt)
        self.assertNotIn("蓝色裙子", req.system_prompt)
        self.assertIn("life_schedule_edit", req.system_prompt)
        self.assertTrue(req.extra_user_content_parts[-1].temporary)
        self.assertIn("白衬衫", req.extra_user_content_parts[-1].text)
        self.assertIn("蓝色裙子", req.extra_user_content_parts[-1].text)
        self.plugin.generator.generate_schedule.assert_not_awaited()

    async def test_temp_contains_same_context_and_preserves_other_request_fields(self):
        original = self.request("")
        self.plugin.config["inject_as_temp"] = False
        await self.plugin.on_llm_request(self.event, original)
        self.plugin.config["inject_as_temp"] = True
        req = self.request()
        existing = req.extra_user_content_parts[0]
        await self.plugin.on_llm_request(self.event, req)
        self.assertEqual(req.system_prompt, "原有人设" + self.module.LIFE_CONTEXT_RULES)
        self.assertEqual(req.prompt, "今天做什么？")
        self.assertEqual(req.contexts, [{"role": "user", "content": "上一轮"}])
        self.assertIs(req.extra_user_content_parts[0], existing)
        self.assertEqual(len(req.extra_user_content_parts), 2)
        part = req.extra_user_content_parts[-1]
        self.assertTrue(part.temporary)
        self.assertEqual(
            self.module.LIFE_CONTEXT_RULES + part.text, original.system_prompt
        )
        self.assertNotIn("<state_following_rules>", part.text)
        self.assertNotIn("<life_wardrobe_rules>", part.text)

    async def test_temp_refreshes_each_request_without_changing_system_prompt(self):
        self.plugin.config["inject_as_temp"] = True
        first = self.request(None)
        await self.plugin.on_llm_request(self.event, first)
        self.data.outfit = "黑色长裙"
        self.data.schedule = "10:00 游泳"
        self.plugin.wardrobe_mgr.for_prompt.return_value = "1. 红色外套"
        second = self.request(None)
        await self.plugin.on_llm_request(self.event, second)
        self.assertEqual(first.system_prompt, self.module.LIFE_CONTEXT_RULES)
        self.assertEqual(second.system_prompt, first.system_prompt)
        self.assertIn("白衬衫", first.extra_user_content_parts[-1].text)
        self.assertIn("黑色长裙", second.extra_user_content_parts[-1].text)
        self.assertNotIn("白衬衫", second.extra_user_content_parts[-1].text)
        self.assertIn("红色外套", second.extra_user_content_parts[-1].text)
        self.assertNotIn("红色外套", second.system_prompt)

    async def test_missing_cache_generates_before_temp_injection(self):
        self.plugin.config["inject_as_temp"] = True
        self.plugin.data_mgr.get.return_value = None
        req = self.request()
        await self.plugin.on_llm_request(self.event, req)
        self.plugin.generator.generate_schedule.assert_awaited_once()
        self.assertTrue(req.extra_user_content_parts[-1].temporary)
        self.assertIn("白衬衫", req.extra_user_content_parts[-1].text)

    async def test_failed_or_busy_schedule_leaves_request_untouched(self):
        for temp in (False, True):
            for busy in (False, True):
                with self.subTest(temp=temp, busy=busy):
                    self.plugin.config["inject_as_temp"] = temp
                    self.data.status = "failed"
                    self.plugin.data_mgr.get.return_value = None if busy else self.data
                    self.plugin.generator.generate_schedule.side_effect = RuntimeError(
                        "schedule_generating"
                    )
                    req = self.request()
                    await self.plugin.on_llm_request(self.event, req)
                    self.assertEqual(req.system_prompt, "原有人设")
                    self.assertEqual(len(req.extra_user_content_parts), 1)

    async def test_older_sdk_without_temp_api_keeps_system_injection(self):
        self.plugin.config["inject_as_temp"] = True
        legacy = types.SimpleNamespace(TextPart=type("LegacyTextPart", (), {}))
        with patch.dict(sys.modules, {"astrbot.core.agent.message": legacy}):
            req = self.request()
            await self.plugin.on_llm_request(self.event, req)
        self.assertIn("白衬衫", req.system_prompt)
        self.assertEqual(len(req.extra_user_content_parts), 1)

    async def test_older_request_without_extra_parts_keeps_schedule_available(self):
        req = self.request()
        del req.extra_user_content_parts
        await self.plugin.on_llm_request(self.event, req)
        self.assertIn("白衬衫", req.system_prompt)
        self.assertIn("<life_wardrobe>", req.system_prompt)
        self.assertFalse(hasattr(req, "extra_user_content_parts"))

    async def test_older_sdk_without_message_module_falls_back(self):
        req = self.request()
        with patch.dict(sys.modules, {"astrbot.core.agent.message": None}):
            await self.plugin.on_llm_request(self.event, req)
        self.assertIn("白衬衫", req.system_prompt)
        self.assertEqual(len(req.extra_user_content_parts), 1)

    async def test_existing_fixed_rules_are_not_duplicated(self):
        req = self.request("原有人设" + self.module.LIFE_CONTEXT_RULES)
        await self.plugin.on_llm_request(self.event, req)
        self.assertEqual(req.system_prompt.count(self.module.LIFE_CONTEXT_RULES), 1)

    def test_schema_defaults_to_cache_friendly_injection(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text()
        )
        self.assertEqual(schema["inject_as_temp"]["type"], "bool")
        self.assertIs(schema["inject_as_temp"]["default"], True)
