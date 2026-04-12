"""Tests for stale response suppression when a session is interrupted.

When a new message arrives while the agent is processing (or just finished
processing) a previous message, the old response should be suppressed so
the user doesn't see a duplicate/stale reply.  (#8221)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    PlatformConfig,
    Platform,
    SendResult,
)


class _StubAdapter(BasePlatformAdapter):
    """Minimal adapter wired for interrupt-suppression tests."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent_messages: list[str] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent_messages.append(content)
        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _make_event(text: str, chat_id: str = "123") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=MagicMock(
            chat_id=chat_id,
            platform=Platform.TELEGRAM,
            thread_id=None,
            user_id="u1",
        ),
        message_id="m1",
    )


class TestStaleResponseSuppression:
    """Verify _process_message_background suppresses stale responses."""

    @pytest.mark.asyncio
    async def test_stale_response_suppressed_when_interrupt_pending(self):
        """If the handler returns a response but the interrupt event is set
        and a pending message exists, the response must be suppressed."""
        adapter = _StubAdapter()
        session_key = "telegram:user:123"
        event_a = _make_event("message A")

        async def fake_handler(event):
            # Simulate interrupt arriving during handler execution:
            # store a pending message and set the interrupt event.
            adapter._pending_messages[session_key] = _make_event("message B")
            adapter._active_sessions[session_key].set()
            return "stale response for A"

        adapter.set_message_handler(fake_handler)

        # _process_message_background will process the pending message
        # recursively, so we also need a handler for that round.
        call_count = 0
        original_handler = fake_handler

        async def counting_handler(event):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return await original_handler(event)
            # Second call (pending message) — return a normal response
            return "response for B"

        adapter.set_message_handler(counting_handler)

        await adapter._process_message_background(event_a, session_key)

        # The stale "response for A" must NOT have been sent.
        assert "stale response for A" not in adapter.sent_messages
        # The response for B should have been sent.
        assert "response for B" in adapter.sent_messages

    @pytest.mark.asyncio
    async def test_response_sent_when_no_interrupt(self):
        """Normal case: no interrupt → response is sent."""
        adapter = _StubAdapter()
        session_key = "telegram:user:456"
        event = _make_event("hello", chat_id="456")

        async def handler(ev):
            return "normal response"

        adapter.set_message_handler(handler)
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter._process_message_background(event, session_key)

        assert "normal response" in adapter.sent_messages

    @pytest.mark.asyncio
    async def test_response_sent_when_photo_burst_queued(self):
        """Photo bursts queue without setting interrupt — response must
        still be sent."""
        adapter = _StubAdapter()
        session_key = "telegram:user:789"
        event = _make_event("describe this", chat_id="789")

        async def handler(ev):
            # Photo burst queued (no interrupt set)
            photo_event = MessageEvent(
                text="",
                message_type=MessageType.PHOTO,
                source=MagicMock(
                    chat_id="789",
                    platform=Platform.TELEGRAM,
                    thread_id=None,
                ),
                message_id="p1",
            )
            adapter._pending_messages[session_key] = photo_event
            # interrupt_event is NOT set (photo bursts don't interrupt)
            return "here is the description"

        adapter.set_message_handler(handler)
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter._process_message_background(event, session_key)

        assert "here is the description" in adapter.sent_messages

    @pytest.mark.asyncio
    async def test_interrupt_event_set_but_no_pending_sends_response(self):
        """If the interrupt event is set but _run_agent already consumed
        the pending message, the response should still be sent (it's the
        response for the NEW message from the recursive call)."""
        adapter = _StubAdapter()
        session_key = "telegram:user:111"
        event = _make_event("message A", chat_id="111")

        async def handler(ev):
            # Simulate _run_agent handling the interrupt: it sets the
            # interrupt event during processing but consumes the pending
            # message.  The response returned is for the NEW message.
            adapter._active_sessions[session_key].set()
            # No pending message left — _run_agent consumed it
            return "response for new message"

        adapter.set_message_handler(handler)
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter._process_message_background(event, session_key)

        assert "response for new message" in adapter.sent_messages
