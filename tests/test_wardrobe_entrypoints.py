import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from test_life_context import _load_main


class WardrobeEntrypointTests(unittest.IsolatedAsyncioTestCase):
    """Exercise #21 through commands/tools without a running OneBot adapter."""

    def setUp(self):
        self.module = _load_main()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plugin = object.__new__(self.module.LifeSchedulerPlugin)
        config_path = self.root / "config.json"

        class Config(dict):
            def save_config(self):
                config_path.write_text(json.dumps(self))

        self.plugin.config = Config()
        self.plugin.wardrobe_mgr = self.module.WardrobeDataManager(
            self.root / "wardrobe.json", self.plugin.config
        )
        self.plugin.data_mgr = self.module.ScheduleDataManager(
            self.root / "schedule.json"
        )
        self.provider = types.SimpleNamespace(
            text_chat=AsyncMock(
                return_value=types.SimpleNamespace(
                    completion_text='{"clothing":"白衬衫","scene":"不要保存的背景"}'
                )
            )
        )
        context = types.SimpleNamespace(
            get_using_provider=lambda: self.provider,
            conversation_manager=types.SimpleNamespace(
                get_curr_conversation_id=AsyncMock(return_value=None)
            ),
        )
        self.plugin.generator = self.module.SchedulerGenerator(
            context, self.plugin.config, self.plugin.data_mgr
        )

    def image(self, path="quoted.png", *, error=None):
        component = self.module.Image()
        component.convert_to_file_path = AsyncMock(return_value=path, side_effect=error)
        return component

    def reply(self, *images):
        reply = self.module.Reply()
        reply.chain = list(images)
        reply.id = "123"
        return reply

    def event(self, *components, admin=True):
        return types.SimpleNamespace(
            message_obj=types.SimpleNamespace(message=list(components)),
            session_id="test-session",
            unified_msg_origin="test:GroupMessage:1",
            plain_result=lambda text: text,
            is_admin=lambda: admin,
        )

    async def run_command(self, event, note=""):
        return [result async for result in self.plugin.wardrobe_add(event, note)]

    def assert_saved(self):
        entries = self.plugin.wardrobe_mgr.all()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["description"], "服装：白衬衫。")
        saved_config = json.loads((self.root / "config.json").read_text())
        reloaded = self.module.WardrobeDataManager(
            self.root / "wardrobe.json", saved_config
        )
        self.assertEqual(reloaded.all(), entries)

    async def test_quoted_image_command_persists_outfit(self):
        replies = await self.run_command(self.event(self.reply(self.image())))
        self.assertTrue(replies[-1].startswith("已放进衣柜"))
        self.assert_saved()
        self.assertEqual(
            self.provider.text_chat.call_args.kwargs["image_urls"], ["quoted.png"]
        )

    async def test_quoted_image_tool_persists_outfit(self):
        result = await self.plugin.llm_wardrobe_add(
            self.event(self.reply(self.image()))
        )
        self.assertTrue(result.startswith("已新增衣柜方案"))
        self.assert_saved()

    async def test_reply_id_fallback_persists_outfit(self):
        with patch.object(
            self.module,
            "extract_quoted_message_images",
            AsyncMock(return_value=["fallback.png"]),
        ) as fallback:
            replies = await self.run_command(self.event(self.reply()))
        fallback.assert_awaited_once()
        self.assertTrue(replies[-1].startswith("已放进衣柜"))
        self.assert_saved()

    async def test_duplicate_and_many_images_send_at_most_four_unique_paths(self):
        images = [self.image(f"{n}.png") for n in range(20)]
        await self.run_command(self.event(self.image("0.png"), self.reply(*images)))
        self.assertEqual(
            self.provider.text_chat.call_args.kwargs["image_urls"],
            ["0.png", "1.png", "2.png", "3.png"],
        )
        self.assert_saved()

    async def test_many_quoted_images_stop_resolving_after_four(self):
        images = [self.image(f"{n}.png") for n in range(32)]
        await self.run_command(self.event(self.reply(*images)))
        self.assertEqual(
            sum(image.convert_to_file_path.await_count for image in images), 4
        )
        self.assert_saved()

    async def test_many_direct_images_stop_resolving_after_four(self):
        images = [self.image(f"{n}.png") for n in range(32)]
        with patch.object(
            self.module, "extract_quoted_message_images", AsyncMock()
        ) as fallback:
            await self.run_command(self.event(*images, self.reply()))
        fallback.assert_not_awaited()
        self.assertEqual(
            sum(image.convert_to_file_path.await_count for image in images), 4
        )
        self.assert_saved()

    async def test_mixed_empty_duplicate_and_failed_images_still_collect_four(self):
        images = [
            self.image(" "),
            self.image(error=OSError("unavailable")),
            self.image(" 0.png "),
            self.image("0.png"),
            self.image("1.png"),
            self.image("2.png"),
            self.image("3.png"),
            self.image("unused.png"),
        ]
        await self.run_command(
            self.event(images[0], images[1], images[2], self.reply(*images[3:]))
        )
        self.assertEqual(
            self.provider.text_chat.call_args.kwargs["image_urls"],
            ["0.png", "1.png", "2.png", "3.png"],
        )
        images[-1].convert_to_file_path.assert_not_awaited()
        self.assert_saved()

    async def test_many_images_tool_also_stops_at_four(self):
        images = [self.image(f"{n}.png") for n in range(32)]
        result = await self.plugin.llm_wardrobe_add(self.event(self.reply(*images)))
        self.assertTrue(result.startswith("已新增衣柜方案"))
        self.assertEqual(
            sum(image.convert_to_file_path.await_count for image in images), 4
        )
        self.assert_saved()

    async def test_fallback_deduplicates_and_stops_at_four(self):
        with patch.object(
            self.module,
            "extract_quoted_message_images",
            AsyncMock(
                return_value=["", " 0.png ", "0.png"]
                + [f"{n}.png" for n in range(1, 32)]
            ),
        ):
            await self.run_command(self.event(self.reply()))
        self.assertEqual(
            self.provider.text_chat.call_args.kwargs["image_urls"],
            ["0.png", "1.png", "2.png", "3.png"],
        )
        self.assert_saved()

    async def test_shared_collector_preserves_renew_and_display_commands(self):
        images = [self.image(f"{n}.png") for n in range(5)]
        self.plugin.generator.generate_schedule = AsyncMock(
            return_value=types.SimpleNamespace(outfit="白衬衫", schedule="09:00 看书")
        )
        results = [
            result
            async for result in self.plugin.life_renew(
                self.event(self.reply(*images)), ""
            )
        ]
        self.assertIn("白衬衫", results[-1])
        self.assertEqual(
            self.plugin.generator.generate_schedule.call_args.kwargs["image_paths"],
            ["0.png", "1.png", "2.png", "3.png"],
        )
        images[-1].convert_to_file_path.assert_not_awaited()
        entry = self.plugin.wardrobe_mgr.add("白衬衫")
        self.plugin.wardrobe_images = types.SimpleNamespace(
            exists=lambda entry_id: False, save=AsyncMock()
        )
        results = [
            result
            async for result in self.plugin.wardrobe_show_image(
                self.event(self.reply(self.image())), "穿搭1"
            )
        ]
        self.assertIn("穿搭1", results[-1])
        self.plugin.wardrobe_images.save.assert_awaited_once_with(
            entry["id"], "quoted.png"
        )

    async def test_empty_or_unavailable_image_does_not_write(self):
        for event in (
            self.event(),
            self.event(self.reply(self.image(error=OSError("unavailable")))),
        ):
            with self.subTest(event=bool(event.message_obj.message)):
                replies = await self.run_command(event)
                self.assertEqual(replies, ["请附上穿搭图片或文字描述"])
        self.provider.text_chat.assert_not_awaited()
        self.assertEqual(self.plugin.wardrobe_mgr.all(), [])

    async def test_provider_failure_is_reported_without_writing(self):
        self.provider.text_chat.side_effect = RuntimeError("provider unavailable")
        replies = await self.run_command(self.event(self.reply(self.image())))
        self.assertIn("放进衣柜失败", replies[-1])
        self.assertEqual(self.plugin.wardrobe_mgr.all(), [])

    async def test_non_admin_tool_does_not_resolve_or_write(self):
        image = self.image()
        result = await self.plugin.llm_wardrobe_add(
            self.event(self.reply(image), admin=False)
        )
        self.assertIn("只有管理员", result)
        image.convert_to_file_path.assert_not_awaited()
        self.provider.text_chat.assert_not_awaited()
        self.assertEqual(self.plugin.wardrobe_mgr.all(), [])

    async def test_waiting_for_provider_keeps_event_loop_running(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def waiting_provider(*args, **kwargs):
            entered.set()
            await release.wait()
            return types.SimpleNamespace(completion_text='{"clothing":"白衬衫"}')

        self.provider.text_chat.side_effect = waiting_provider
        task = asyncio.create_task(
            self.run_command(self.event(self.reply(self.image())))
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            self.assertFalse(task.done())
            self.assertEqual(self.plugin.wardrobe_mgr.all(), [])
            release.set()
            await asyncio.wait_for(task, timeout=2)
            self.assert_saved()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
