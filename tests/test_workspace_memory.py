"""User-owned strong memory stays current and cannot be edited by file tools."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from octop_memory import Memory, MemoryService

from octop_harness.backends.workspace import BackendWorkspace
from octop_harness.builtin.tools.memory_tools import build_memory_tools
from octop_harness.middleware.workspace_memory import WorkspaceMemoryMiddleware


def _workspace(tmp_path: Path, prefix: str = "") -> BackendWorkspace:
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    return BackendWorkspace(backend, tmp_path, system_files_path=prefix)


@pytest.mark.parametrize("prefix", ["", ".octop"])
def test_refreshes_cached_memory_after_user_edits_and_removes_file(tmp_path: Path, prefix: str) -> None:
    ws = _workspace(tmp_path, prefix)
    ws.write_text("MEMORY.md", "original strong memory", force=True)
    mw = WorkspaceMemoryMiddleware(ws)
    first = mw.before_agent({}, MagicMock())
    assert first is not None
    ws.write_text("MEMORY.md", "updated strong memory", force=True)
    second = mw.before_agent(first, MagicMock())
    assert second is not None
    assert list(second["memory_contents"].values()) == ["updated strong memory"]
    ws.delete("MEMORY.md")
    assert mw.before_agent(second, MagicMock()) == {"memory_contents": {}}


@pytest.mark.asyncio
async def test_loads_new_strong_memory_even_when_other_bootstrap_files_are_suppressed(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    mw = WorkspaceMemoryMiddleware(ws, memory_files=())
    assert await mw.abefore_agent({}, MagicMock()) == {"memory_contents": {}}
    content = "用户手动维护的强记忆\n" * 1000
    ws.write_text("MEMORY.md", content, force=True)
    ws.write_text("AGENTS.md", "bootstrap instructions", force=True)
    loaded = await mw.abefore_agent({"memory_contents": {}}, MagicMock())
    assert loaded is not None
    assert list(loaded["memory_contents"].values()) == [content]
    request = ModelRequest(
        model=MagicMock(),
        tools=[],
        system_message=SystemMessage(content="base"),
        messages=[HumanMessage(content="hi")],
        state=loaded,
        runtime=MagicMock(),
    )
    system = mw.modify_request(request).system_message
    assert system is not None
    text = "\n".join(str(block.get("text", "")) for block in system.content_blocks if isinstance(block, dict))
    assert content.rstrip() in text
    assert "memory_save" in str(system.content)
    assert "Do not edit" in str(system.content)


@pytest.mark.parametrize("tool_name", ["write_file", "edit_file", "append_file", "delete"])
@pytest.mark.parametrize("path", ["MEMORY.md", "/MEMORY.md", "native"])
def test_file_tools_cannot_change_strong_memory(tmp_path: Path, tool_name: str, path: str) -> None:
    ws = _workspace(tmp_path, ".octop")
    mw = WorkspaceMemoryMiddleware(ws)
    if path == "native":
        path = ws.resolve_path("MEMORY.md")
    request = MagicMock(tool_call={"name": tool_name, "args": {"file_path": path}, "id": "call-1"})
    handler = MagicMock()
    result = mw.wrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "memory_save" in str(result.content)
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_async_guard_allows_normal_workspace_edits(tmp_path: Path) -> None:
    mw = WorkspaceMemoryMiddleware(_workspace(tmp_path))
    request = MagicMock(tool_call={"name": "edit_file", "args": {"file_path": "notes.md"}, "id": "call-1"})
    calls: list[Any] = []

    async def handler(value: Any) -> ToolMessage:
        calls.append(value)
        return ToolMessage(content="edited", tool_call_id="call-1")

    assert (await mw.awrap_tool_call(request, handler)).content == "edited"
    assert calls == [request]


def test_real_graph_saves_regular_memory_and_refreshes_manual_context_between_turns(tmp_path: Path) -> None:
    class RecordingModel(FakeMessagesListChatModel):
        requests: list[Any] = []

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
            self.requests.append(deepcopy(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    ws = _workspace(tmp_path)
    ws.write_text("MEMORY.md", "first manual agreement", force=True)
    memory = Memory(namespace="strong_test", backend_config={"db_path": str(tmp_path / "memory.sqlite")})
    service = MemoryService(memory)
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "memory_save",
                        "args": {"content": "Aurora uses PostgreSQL"},
                        "id": "save-1",
                    }
                ],
            ),
            AIMessage(content="Saved"),
            AIMessage(content="Updated context received"),
        ]
    )
    graph = create_deep_agent(
        model=model,
        backend=ws.backend,
        tools=build_memory_tools(service),
        memory=ws.memory_paths(),
        middleware=[WorkspaceMemoryMiddleware(ws)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "strong-memory-test"}}
    graph.invoke({"messages": [HumanMessage(content="Remember Aurora uses PostgreSQL")]}, config=config)
    assert memory.search_atoms("PostgreSQL")
    assert ws.read_text("MEMORY.md") == "first manual agreement"
    ws.write_text("MEMORY.md", "second manual agreement", force=True)
    graph.invoke({"messages": [HumanMessage(content="Continue")]}, config=config)
    last_system = model.requests[-1][0]
    assert "second manual agreement" in str(last_system.content)
    assert "first manual agreement" not in str(last_system.content)
    assert "Do not edit" in str(last_system.content)
    assert "As you learn from your interactions" not in str(last_system.content)
