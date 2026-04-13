import json
from types import SimpleNamespace


def _make_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test-model", usage=usage)


def _tool_call_chunks(arguments):
    tc_delta = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="read_file", arguments=json.dumps(arguments)),
    )
    yield _make_chunk(tool_calls=[tc_delta])
    yield _make_chunk(finish_reason="tool_calls")


def _done_chunks():
    yield _make_chunk(content="done")
    yield _make_chunk(finish_reason="stop")


class _FakeChatCompletions:
    """Shared completions that return streaming chunks.

    First streaming call returns a tool-call response; subsequent calls
    return a plain text ``"done"`` response.  The counter is shared across
    all ``_FakeClient`` instances so it survives the per-request clients
    that ``run_agent`` creates internally.
    """

    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _tool_call_chunks({"path": "README.md"})
        return _done_chunks()


class _FakeClient:
    _shared_completions = None

    def __init__(self):
        self.chat = SimpleNamespace(completions=self._shared_completions)
        self._client = SimpleNamespace(is_closed=False)

    def close(self):
        self._client.is_closed = True


def test_tool_call_validation_accepts_dict_arguments(monkeypatch):
    from run_agent import AIAgent

    shared = _FakeChatCompletions()
    _FakeClient._shared_completions = shared

    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [{"function": {"name": "read_file"}}],
    )
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True, "args": args}),
    )

    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="cli",
        max_iterations=3,
        quiet_mode=True,
        skip_memory=True,
    )
    result = agent.run_conversation("read the file")

    assert result["final_response"] == "done"
