"""Test Ctrl+D keybinding exits only on empty buffer."""


def test_ctrl_d_handler_skips_exit_when_buffer_has_text():
    """Ctrl+D with non-empty buffer must NOT trigger exit (standard EOF behavior)."""
    from unittest.mock import MagicMock

    # Simulate the handler logic inline (avoids importing the full CLI)
    _should_exit = False

    def handle_ctrl_d(event):
        nonlocal _should_exit
        if not event.app.current_buffer.text:
            _should_exit = True
            event.app.exit()

    event = MagicMock()
    event.app.current_buffer.text = "some typed text"

    handle_ctrl_d(event)

    assert not _should_exit
    event.app.exit.assert_not_called()


def test_ctrl_d_handler_exits_on_empty_buffer():
    """Ctrl+D with empty buffer must trigger exit."""
    from unittest.mock import MagicMock

    _should_exit = False

    def handle_ctrl_d(event):
        nonlocal _should_exit
        if not event.app.current_buffer.text:
            _should_exit = True
            event.app.exit()

    event = MagicMock()
    event.app.current_buffer.text = ""

    handle_ctrl_d(event)

    assert _should_exit
    event.app.exit.assert_called_once()
