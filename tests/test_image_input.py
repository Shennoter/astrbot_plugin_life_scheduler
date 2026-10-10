import sys
import types
import unittest
from unittest.mock import patch

from test_life_context import _load_main


class ImageInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = _load_main()

    async def collect(
        self,
        *,
        direct=0,
        quoted=0,
        embedded=0,
        nested=False,
        broken=False,
        images_per_forward=1,
    ):
        calls = []
        self.forward_parses = []
        forwards = [str(i) for i in range(32)]

        def parsed(images=(), ids=()):
            return {"image_refs": list(images), "forward_ids": list(ids)}

        async def get_msg(reply_id):
            calls.append(("get_msg", reply_id))
            return parsed(
                [f"https://example.test/quoted-{i}" for i in range(quoted)],
                forwards[:1] if nested else forwards,
            )

        async def get_forward_msg(fid):
            calls.append(("get_forward_msg", fid))
            return parsed(
                [
                    f"https://example.test/forward-{fid}-{i}"
                    for i in range(images_per_forward)
                ],
                [str((int(fid) + 1) % 32)] if nested else [],
            )

        def parse_forward(data):
            self.forward_parses.append(data)
            return data

        class Extractor:
            def __init__(self, event):
                self._client = types.SimpleNamespace(
                    get_msg=get_msg, get_forward_msg=get_forward_msg
                )
                self._settings = types.SimpleNamespace(max_forward_fetch=32)
                self._reply_parser = types.SimpleNamespace(
                    extract_image_refs_from_reply_component=lambda reply: [
                        f"https://example.test/embedded-{i}" for i in range(embedded)
                    ]
                )
                self._payload_parser = types.SimpleNamespace(
                    parse_get_msg_payload=lambda data: data,
                    parse_get_forward_payload=parse_forward,
                )

            async def _fetch_quoted_content(self, reply, *, fetch_remote):
                # Reproduce the SDK's eager traversal, including its hop cap.
                data = await get_msg(reply.id)
                pending = list(data["forward_ids"])
                seen = set()
                refs = []
                while pending and len(seen) < 32:
                    fid = pending.pop(0)
                    if fid in seen:
                        continue
                    seen.add(fid)
                    item = parse_forward(await get_forward_msg(fid))
                    refs.extend(item["image_refs"])
                    pending.extend(item["forward_ids"])
                return types.SimpleNamespace(
                    embedded_image_refs=[
                        f"https://example.test/embedded-{i}" for i in range(embedded)
                    ],
                    direct_image_refs=data["image_refs"],
                    forward_image_refs=refs,
                )

        async def convert(image):
            if broken:
                raise OSError("missing attachment")
            return image.file

        reply = self.module.Reply()
        reply.id, reply.chain = "123", []
        event = types.SimpleNamespace(
            message_obj=types.SimpleNamespace(
                message=[
                    self.module.Image(file=f"https://example.test/direct-{i}")
                    for i in range(direct)
                ]
                + [reply]
            )
        )
        plugin = object.__new__(self.module.LifeSchedulerPlugin)
        with (
            patch.object(
                sys.modules["astrbot.core.utils.quoted_message.extractor"],
                "QuotedMessageExtractor",
                Extractor,
            ),
            patch.object(self.module.Image, "convert_to_file_path", convert),
        ):
            paths = await plugin._collect_image_paths(event)
        return paths, calls

    async def test_forward_fetch_stops_after_remaining_successful_images(self):
        for nested in (False, True):
            for direct in (0, 2, 4):
                with self.subTest(nested=nested, direct=direct):
                    paths, calls = await self.collect(direct=direct, nested=nested)
                    self.assertEqual(len(paths), 4)
                    self.assertEqual(
                        sum(action == "get_forward_msg" for action, _ in calls),
                        4 - direct,
                    )
                    self.assertEqual(len(self.forward_parses), 4 - direct)

    async def test_one_forward_fills_remaining_slots_without_parsing_next(self):
        for direct in (0, 2):
            with self.subTest(direct=direct):
                paths, calls = await self.collect(direct=direct, images_per_forward=8)
                self.assertEqual(len(paths), 4)
                self.assertEqual(calls, [("get_msg", "123"), ("get_forward_msg", "0")])
                self.assertEqual(len(self.forward_parses), 1)

    async def test_quoted_direct_images_prevent_unneeded_forward_fetch(self):
        paths, calls = await self.collect(quoted=4)
        self.assertEqual(len(paths), 4)
        self.assertEqual(calls, [("get_msg", "123")])
        self.assertEqual(self.forward_parses, [])

    async def test_embedded_reply_refs_limit_forward_fetch_and_parse(self):
        for embedded in (2, 4):
            with self.subTest(embedded=embedded):
                paths, calls = await self.collect(embedded=embedded)
                self.assertEqual(len(paths), 4)
                self.assertEqual(
                    calls,
                    ([] if embedded == 4 else [("get_msg", "123")])
                    + [("get_forward_msg", str(i)) for i in range(4 - embedded)],
                )
                self.assertEqual(len(self.forward_parses), 4 - embedded)

    async def test_failed_images_and_forward_cycle_preserve_hop_cap(self):
        paths, calls = await self.collect(nested=True, broken=True)
        self.assertEqual(paths, [])
        self.assertEqual(len(calls), 33)
        self.assertEqual(len(set(calls)), 33)

    async def test_modern_provider_receives_paths_without_preencoding(self):
        entities = sys.modules["astrbot.core.provider.entities"]
        image_input = sys.modules[self.module.resolve_image_path.__module__]
        refs = [f"/images/{i}.png" for i in range(32)]
        with (
            patch.object(entities, "MediaResolver", object, create=True),
            patch.object(
                image_input, "resolve_image_path", return_value="/images/0.png"
            ) as convert,
            patch("pathlib.Path.read_bytes", return_value=b"image") as read,
        ):
            prepared = await image_input.prepare_provider_images(refs)
        self.assertEqual(prepared, refs[:4])
        convert.assert_not_called()
        read.assert_not_called()
