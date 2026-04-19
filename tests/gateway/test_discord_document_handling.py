"""Tests for Discord incoming document/file attachment handling.

Covers the document branch in DiscordAdapter._handle_message() —
the `else` clause of the attachment content-type loop that was added
to download, cache, and optionally inject text from non-image/audio files.
"""

import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


# ---------------------------------------------------------------------------
# Discord mock setup (copied from test_discord_free_response.py)
# ---------------------------------------------------------------------------

def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import gateway.platforms.discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fake channel / thread types
# ---------------------------------------------------------------------------

class FakeDMChannel:
    def __init__(self, channel_id: int = 1):
        self.id = channel_id
        self.name = "dm"


class FakeThread:
    def __init__(self, channel_id: int = 10):
        self.id = channel_id
        self.name = "thread"
        self.parent = None
        self.parent_id = None
        self.guild = SimpleNamespace(name="TestServer")
        self.topic = None


class FakeTextChannel:
    def __init__(self, channel_id: int = 200):
        self.id = channel_id
        self.name = "general"
        self.guild = SimpleNamespace(name="TestServer")
        self.topic = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document cache to tmp_path so tests never write to ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=999))
    a.handle_message = AsyncMock()
    return a


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_attachment(
    *,
    filename: str,
    content_type: str,
    size: int = 1024,
    url: str = "https://cdn.discordapp.com/attachments/fake/file",
) -> SimpleNamespace:
    return SimpleNamespace(
        filename=filename,
        content_type=content_type,
        size=size,
        url=url,
    )


def make_message(attachments: list, content: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        id=123,
        content=content,
        attachments=attachments,
        mentions=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=FakeDMChannel(),
        author=SimpleNamespace(id=42, display_name="Tester", name="Tester"),
    )


def _mock_aiohttp_download(raw_bytes: bytes):
    """Return a patch context manager that makes aiohttp return raw_bytes."""
    resp = AsyncMock()
    resp.status = 200
    resp.read = AsyncMock(return_value=raw_bytes)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.get = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    return patch("aiohttp.ClientSession", return_value=session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIncomingDocumentHandling:

    @pytest.mark.asyncio
    async def test_pdf_document_cached(self, adapter):
        """A PDF attachment should be downloaded, cached, typed as DOCUMENT."""
        pdf_bytes = b"%PDF-1.4 fake content"

        with _mock_aiohttp_download(pdf_bytes):
            msg = make_message([make_attachment(filename="report.pdf", content_type="application/pdf")])
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT
        assert len(event.media_urls) == 1
        assert os.path.exists(event.media_urls[0])
        assert event.media_types == ["application/pdf"]
        assert "[Content of" not in (event.text or "")

    @pytest.mark.asyncio
    async def test_txt_content_injected(self, adapter):
        """.txt file under 100KB should have its content injected into event.text."""
        file_content = b"Hello from a text file"

        with _mock_aiohttp_download(file_content):
            msg = make_message(
                attachments=[make_attachment(filename="notes.txt", content_type="text/plain")],
                content="summarize this",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert "[Content of notes.txt]:" in event.text
        assert "Hello from a text file" in event.text
        assert "summarize this" in event.text
        # injection prepended before caption
        assert event.text.index("[Content of") < event.text.index("summarize this")

    @pytest.mark.asyncio
    async def test_md_content_injected(self, adapter):
        """.md file under 100KB should have its content injected."""
        file_content = b"# Title\nSome markdown content"

        with _mock_aiohttp_download(file_content):
            msg = make_message(
                attachments=[make_attachment(filename="readme.md", content_type="text/markdown")],
                content="",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert "[Content of readme.md]:" in event.text
        assert "# Title" in event.text

    @pytest.mark.asyncio
    async def test_log_content_injected(self, adapter):
        """.log file under 100KB should be treated as text/plain and injected."""
        file_content = b"BLE trace line 1\nBLE trace line 2"

        with _mock_aiohttp_download(file_content):
            msg = make_message(
                attachments=[make_attachment(filename="btsnoop_hci.log", content_type="text/plain")],
                content="please inspect this",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert "[Content of btsnoop_hci.log]:" in event.text
        assert "BLE trace line 1" in event.text
        assert "please inspect this" in event.text

    @pytest.mark.asyncio
    async def test_oversized_document_skipped(self, adapter):
        """A document over 32MB should be skipped — media_urls stays empty."""
        msg = make_message([
            make_attachment(
                filename="huge.pdf",
                content_type="application/pdf",
                size=33 * 1024 * 1024,
            )
        ])
        await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls == []
        # handler must still be called
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_mid_sized_zip_under_32mb_is_cached(self, adapter):
        """A 25MB .zip should be accepted now that Discord documents allow up to 32MB."""
        msg = make_message([
            make_attachment(
                filename="bugreport.zip",
                content_type="application/zip",
                size=25 * 1024 * 1024,
            )
        ])

        with _mock_aiohttp_download(b"PK\x03\x04test"):
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert event.media_types == ["application/zip"]

    @pytest.mark.asyncio
    async def test_zip_document_cached(self, adapter):
        """A .zip file should be cached as a supported document."""
        msg = make_message([
            make_attachment(filename="archive.zip", content_type="application/zip")
        ])

        with _mock_aiohttp_download(b"PK\x03\x04test"):
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert event.media_types == ["application/zip"]
        assert event.message_type == MessageType.DOCUMENT

    @pytest.mark.asyncio
    async def test_download_error_handled(self, adapter):
        """If the HTTP download raises, the handler should not crash."""
        resp = AsyncMock()
        resp.__aenter__ = AsyncMock(side_effect=RuntimeError("connection reset"))
        resp.__aexit__ = AsyncMock(return_value=False)

        session = AsyncMock()
        session.get = MagicMock(return_value=resp)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session):
            msg = make_message([
                make_attachment(filename="report.pdf", content_type="application/pdf")
            ])
            await adapter._handle_message(msg)

        # Must still deliver an event
        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls == []

    @pytest.mark.asyncio
    async def test_large_txt_cached_not_injected(self, adapter):
        """.txt over 100KB should be cached but NOT injected into event.text."""
        large_content = b"x" * (200 * 1024)

        with _mock_aiohttp_download(large_content):
            msg = make_message(
                attachments=[make_attachment(filename="big.txt", content_type="text/plain", size=len(large_content))],
                content="",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert os.path.exists(event.media_urls[0])
        assert "[Content of" not in (event.text or "")

    @pytest.mark.asyncio
    async def test_multiple_text_files_both_injected(self, adapter):
        """Two text file attachments should both be injected into event.text in order."""
        content1 = b"First file content"
        content2 = b"Second file content"

        call_count = 0
        responses = [content1, content2]

        def make_session(_responses):
            idx = 0

            class FakeSession:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_):
                    pass

                def get(self, url, **kwargs):
                    nonlocal idx
                    data = _responses[idx % len(_responses)]
                    idx += 1

                    resp = AsyncMock()
                    resp.status = 200
                    resp.read = AsyncMock(return_value=data)
                    resp.__aenter__ = AsyncMock(return_value=resp)
                    resp.__aexit__ = AsyncMock(return_value=False)
                    return resp

            return FakeSession()

        with patch("aiohttp.ClientSession", return_value=make_session([content1, content2])):
            msg = make_message(
                attachments=[
                    make_attachment(filename="file1.txt", content_type="text/plain"),
                    make_attachment(filename="file2.txt", content_type="text/plain"),
                ],
                content="",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert "[Content of file1.txt]:" in event.text
        assert "First file content" in event.text
        assert "[Content of file2.txt]:" in event.text
        assert "Second file content" in event.text
        assert event.text.index("file1") < event.text.index("file2")

    @pytest.mark.asyncio
    async def test_image_attachment_unaffected(self, adapter):
        """Image attachments should still go through the image path, not the document path."""
        with patch(
            "gateway.platforms.discord.cache_image_from_url",
            new_callable=AsyncMock,
            return_value="/tmp/cached_image.png",
        ):
            msg = make_message([
                make_attachment(filename="photo.png", content_type="image/png")
            ])
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls == ["/tmp/cached_image.png"]
        assert event.media_types == ["image/png"]


# ---------------------------------------------------------------------------
# Tests: Discord auto-generated message.txt from pasted code/long lines
#
# When a user pastes a large block of code or long text into a Discord message,
# Discord automatically converts it to a `message.txt` attachment.  These tests
# verify that Hermes reads and injects the attachment content so the agent has
# the pasted material available as inline text.
# ---------------------------------------------------------------------------

def make_server_message(attachments: list, content: str = "", mentions=None) -> SimpleNamespace:
    """Build a fake message arriving in a server text channel."""
    return SimpleNamespace(
        id=456,
        content=content,
        attachments=attachments,
        mentions=list(mentions or []),
        reference=None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        channel=FakeTextChannel(),
        author=SimpleNamespace(id=42, display_name="Tester", name="Tester"),
    )


@pytest.fixture
def server_adapter(monkeypatch):
    """Adapter wired to a server channel context with auto-thread disabled."""
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    config = PlatformConfig(enabled=True, token="fake-token")
    a = DiscordAdapter(config)
    bot_user = SimpleNamespace(id=999)
    a._client = SimpleNamespace(user=bot_user)
    a._text_batch_delay_seconds = 0
    a.handle_message = AsyncMock()
    return a, bot_user


class TestDiscordMessageTxtAutoAttachment:
    """Regression tests for GitHub issue #12511.

    Discord converts large pastes to a `message.txt` attachment.  The gateway
    must read and inject that content even when message.content is empty after
    @mention stripping.
    """

    @pytest.mark.asyncio
    async def test_message_txt_content_injected_in_free_response_channel(
        self, server_adapter, monkeypatch
    ):
        """In a free-response channel (no mention required) a bare message.txt
        paste should have its content injected into event.text."""
        adapter, _ = server_adapter
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

        code = b"def hello():\n    print('world')\n"
        msg = make_server_message(
            attachments=[make_attachment(filename="message.txt", content_type="text/plain", size=len(code))],
            content="",
        )

        with _mock_aiohttp_download(code):
            await adapter._handle_message(msg)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert "[Content of message.txt]:" in event.text
        assert "def hello():" in event.text

    @pytest.mark.asyncio
    async def test_message_txt_injected_when_mention_stripped_to_empty(
        self, server_adapter, monkeypatch
    ):
        """Server channel: user @mentions the bot AND pastes code.  After the
        @mention is stripped from message.content the remaining content is empty,
        but the message.txt attachment content must still appear in event.text."""
        adapter, bot_user = server_adapter
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")

        code = b"SELECT * FROM users WHERE id = 1;\n"
        msg = make_server_message(
            attachments=[make_attachment(filename="message.txt", content_type="text/plain", size=len(code))],
            content=f"<@{bot_user.id}>",
            mentions=[bot_user],
        )

        with _mock_aiohttp_download(code):
            await adapter._handle_message(msg)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert "[Content of message.txt]:" in event.text
        assert "SELECT * FROM users" in event.text

    @pytest.mark.asyncio
    async def test_message_txt_with_caption_injected_before_caption(
        self, server_adapter, monkeypatch
    ):
        """When the user adds a caption alongside the pasted code, the attachment
        content is injected before the caption so the agent sees code then question."""
        adapter, bot_user = server_adapter
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")

        code = b"fn main() { println!(\"hello\"); }\n"
        msg = make_server_message(
            attachments=[make_attachment(filename="message.txt", content_type="text/plain", size=len(code))],
            content=f"<@{bot_user.id}> what does this Rust code do?",
            mentions=[bot_user],
        )

        with _mock_aiohttp_download(code):
            await adapter._handle_message(msg)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert "[Content of message.txt]:" in event.text
        assert "fn main()" in event.text
        assert "what does this Rust code do?" in event.text
        assert event.text.index("[Content of message.txt]") < event.text.index("what does this Rust code do?")

    @pytest.mark.asyncio
    async def test_message_txt_without_mention_dropped_in_require_mention_channel(
        self, server_adapter, monkeypatch
    ):
        """In a server channel with require_mention=true, a bare message.txt paste
        with no @mention is still silently dropped (expected gating behaviour)."""
        adapter, _ = server_adapter
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")

        code = b"some pasted text\n"
        msg = make_server_message(
            attachments=[make_attachment(filename="message.txt", content_type="text/plain", size=len(code))],
            content="",
            mentions=[],
        )

        with _mock_aiohttp_download(code):
            await adapter._handle_message(msg)

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_message_txt_no_content_type_still_injected(
        self, server_adapter, monkeypatch
    ):
        """When Discord omits content_type for the auto-generated attachment
        (content_type=None), the gateway must still detect the .txt extension
        from the filename and inject the content."""
        adapter, _ = server_adapter
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

        code = b"console.log('hello world');\n"
        att = SimpleNamespace(
            filename="message.txt",
            content_type=None,
            size=len(code),
            url="https://cdn.discordapp.com/attachments/fake/message.txt",
        )
        msg = make_server_message(attachments=[att], content="")

        with _mock_aiohttp_download(code):
            await adapter._handle_message(msg)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert "[Content of message.txt]:" in event.text
        assert "console.log" in event.text
