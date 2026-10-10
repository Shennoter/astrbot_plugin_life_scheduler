import asyncio
import base64
import json
import tempfile
import threading
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
        # These entrypoint tests use symbolic paths. Real encoding is exercised
        # separately below and by the SDK integration probe.
        self.encoding_patch = patch(
            f"{self.module.SchedulerGenerator.__module__}.prepare_provider_images",
            AsyncMock(side_effect=lambda refs: refs),
        )
        self.encoding_patch.start()
        self.addCleanup(self.encoding_patch.stop)

    @staticmethod
    async def quoted_refs(refs):
        for ref in refs:
            yield ref

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
            "iter_quoted_image_paths",
            return_value=self.quoted_refs(["fallback.png"]),
        ) as fallback:
            replies = await self.run_command(self.event(self.reply()))
        fallback.assert_called_once()
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
        with patch.object(self.module, "iter_quoted_image_paths") as fallback:
            await self.run_command(self.event(*images, self.reply()))
        fallback.assert_not_called()
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
            "iter_quoted_image_paths",
            return_value=self.quoted_refs(
                ["", " 0.png ", "0.png"] + [f"{n}.png" for n in range(1, 32)]
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

    async def test_small_batches_convert_off_loop_and_allow_heartbeat(self):
        for count in (1, 3):
            for quoted in (False, True):
                with self.subTest(count=count, quoted=quoted):
                    loop_thread = threading.get_ident()
                    conversion_threads = []
                    entered = threading.Event()
                    release = threading.Event()
                    ticks = []
                    images = []

                    async def heartbeat(entered=entered, ticks=ticks, release=release):
                        while not entered.is_set():
                            await asyncio.sleep(0.001)
                        ticks.append(1)
                        release.set()

                    for index in range(count):
                        target = self.root / f"small-{index}.png"

                        async def convert(
                            path=target,
                            conversion_threads=conversion_threads,
                            entered=entered,
                            release=release,
                        ):
                            conversion_threads.append(threading.get_ident())
                            entered.set()
                            release.wait(0.15)
                            path.write_bytes(base64.b64decode("aW1hZ2U="))
                            return str(path)

                        image = self.image()
                        image.convert_to_file_path.side_effect = convert
                        images.append(image)
                    beat = asyncio.create_task(heartbeat())
                    try:
                        refs = await self.plugin._collect_image_paths(
                            self.event(self.reply(*images))
                            if quoted
                            else self.event(*images)
                        )
                        self.assertEqual(len(refs), count)
                        self.assertNotIn(loop_thread, conversion_threads)
                        self.assertTrue(
                            ticks, "heartbeat did not run during conversion"
                        )
                    finally:
                        release.set()
                        beat.cancel()
                        await asyncio.gather(beat, return_exceptions=True)

    async def test_reply_fetch_resolves_only_remaining_slots(self):
        for direct_count in (0, 2, 4):
            with self.subTest(direct_count=direct_count):
                calls = []
                loop_thread = threading.get_ident()

                async def call_action(
                    action,
                    params=None,
                    *,
                    loop_thread=loop_thread,
                    calls=calls,
                    **kwargs,
                ):
                    self.assertEqual(threading.get_ident(), loop_thread)
                    calls.append(action)
                    if action == "get_msg":
                        return [f"id-{index}" for index in range(32)]
                    return {"url": f"https://example.test/{params['file']}.png"}

                event = self.event(
                    *[self.image(f"direct-{i}.png") for i in range(direct_count)],
                    self.reply(),
                )
                event.bot = types.SimpleNamespace(
                    api=types.SimpleNamespace(call_action=call_action)
                )
                refs = await self.plugin._collect_image_paths(event)
                self.assertEqual(len(refs), 4)
                self.assertEqual(calls.count("get_image"), 4 - direct_count)
                self.assertEqual(calls.count("get_msg"), int(direct_count < 4))

    async def test_single_image_provider_file_encoding_runs_off_loop(self):
        self.encoding_patch.stop()
        source = self.root / "provider.png"
        payload = b"actual image bytes for provider encoding"
        source.write_bytes(payload)
        loop_thread = threading.get_ident()
        read_threads = []
        original_read = Path.read_bytes

        def read_bytes(path):
            read_threads.append(threading.get_ident())
            return original_read(path)

        async def text_chat(*args, image_urls, **kwargs):
            ref = image_urls[0]
            if ref.startswith("base64://"):
                decoded = base64.b64decode(ref.removeprefix("base64://"))
            else:
                decoded = Path(ref).read_bytes()
            self.assertEqual(decoded, payload)
            return types.SimpleNamespace(completion_text='{"clothing":"白衬衫"}')

        self.provider.text_chat.side_effect = text_chat
        with patch.object(Path, "read_bytes", read_bytes):
            await self.run_command(self.event(self.reply(self.image(str(source)))))
        self.assert_saved()
        self.assertTrue(read_threads)
        self.assertNotIn(loop_thread, read_threads)

    async def test_provider_cap_precedes_file_reads_and_retries_reuse_encoding(self):
        self.encoding_patch.stop()
        paths = []
        for i in range(4):
            path = self.root / f"provider-{i}.png"
            path.write_bytes(f"image-{i}".encode())
            paths.append(str(path))
        # Accessing any excess path fails; the limit must precede all file work.
        paths.extend([str(self.root / "must-not-open.png")] * 28)
        self.provider.text_chat.side_effect = [
            types.SimpleNamespace(completion_text=""),
            types.SimpleNamespace(completion_text="done"),
        ]
        with patch.object(
            Path, "read_bytes", autospec=True, side_effect=Path.read_bytes
        ) as read:
            result = await self.plugin.generator._call_llm("test", image_urls=paths)
        self.assertEqual(result, "done")
        self.assertEqual(read.call_count, 4)
        calls = self.provider.text_chat.call_args_list
        self.assertEqual(calls[0].kwargs["image_urls"], calls[1].kwargs["image_urls"])
        self.assertEqual(len(calls[0].kwargs["image_urls"]), 4)

    async def test_reply_fetch_converts_single_image_off_loop(self):
        threads = []
        target = self.root / "fetched.png"

        async def convert(component):
            threads.append(threading.get_ident())
            target.write_bytes(
                base64.b64decode(component.file.removeprefix("base64://"))
            )
            return str(target)

        async def call_action(action, **kwargs):
            self.assertEqual(action, "get_msg")
            return ["base64://aW1hZ2U="]

        event = self.event(self.reply())
        event.bot = types.SimpleNamespace(
            api=types.SimpleNamespace(call_action=call_action)
        )
        with patch.object(self.module.Image, "convert_to_file_path", convert):
            refs = await self.plugin._collect_image_paths(event)
        self.assertEqual(refs, [str(target)])
        self.assertEqual(target.read_bytes(), b"image")
        self.assertNotIn(threading.get_ident(), threads)
