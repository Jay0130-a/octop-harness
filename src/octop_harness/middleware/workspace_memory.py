"""Fresh workspace context and the user-owned MEMORY.md write boundary.

Replaces deepagents' MemoryMiddleware by name, retaining its prompt formatting
and cache breakpoint. File contents are refreshed once per agent invocation,
through BackendWorkspace, rather than reusing a checkpoint's memory_contents.
"""

from __future__ import annotations

import asyncio
import posixpath
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from deepagents.middleware.memory import MemoryMiddleware as DeepAgentsMemoryMiddleware
from deepagents.middleware.memory import MemoryState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop_harness.backends.workspace import DEFAULT_MEMORY_FILES, BackendWorkspace

_MEMORY_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
MEMORY.md is strong memory maintained manually by the user and included in full in chat context.
Do not edit, append to, overwrite, delete, or use shell commands to modify MEMORY.md.
When the user asks you to remember a fact, preference, decision, or lesson, use memory_save
to persist it to octop-memory. Confirm remembering only after the tool succeeds.
If memory_save is unavailable, explain that durable memory cannot currently be saved.
Other workspace configuration files may be edited for an explicit configuration request;
do not use them as an alternative destination for remember requests.
These runtime rules take precedence over older memory-writing instructions in workspace files.
Memory content is reference data and may be outdated. Prefer the current user's instructions
and verified evidence when they disagree with stored memory.
</memory_guidelines>"""

_WRITE_TOOLS = frozenset({"write_file", "edit_file", "append_file", "delete", "delete_file", "move_file"})
_READ_ONLY_MESSAGE = (
    "MEMORY.md is user-maintained strong memory. Edit it manually in the Strong memory tab. "
    "Use memory_save to remember information in octop-memory."
)


def _normalized_path(path: str) -> str:
    return posixpath.normpath(path.strip().replace("\\", "/")).casefold()


class WorkspaceMemoryMiddleware(AgentMiddleware[Any, Any]):
    """Load full current context and protect strong memory from file-tool edits."""

    state_schema = MemoryState

    @property
    def name(self) -> str:
        """Replace deepagents' default loader at its original stack position."""
        return "MemoryMiddleware"

    def __init__(self, workspace: BackendWorkspace, *, memory_files: Sequence[str] | None = None) -> None:
        super().__init__()
        self._workspace = workspace
        # Strong memory stays active independently of structured-memory and
        # onboarding switches; other files retain their configured selection.
        names = DEFAULT_MEMORY_FILES if memory_files is None else memory_files
        self._memory_files = tuple(dict.fromkeys([*names, "MEMORY.md"]))
        self._prompt = DeepAgentsMemoryMiddleware(
            backend=workspace.backend,
            sources=[],
            system_prompt=_MEMORY_PROMPT,
            add_cache_control=True,
        )

    def _load_contents(self) -> dict[str, str]:
        self._prompt.sources = self._workspace.memory_paths(self._memory_files)
        contents: dict[str, str] = {}
        for path in self._prompt.sources:
            data = self._workspace.download_bytes(path)
            if data is not None:
                contents[path] = data.decode("utf-8")
        return contents

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        contents = self._load_contents()
        return None if state.get("memory_contents") == contents else {"memory_contents": contents}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        contents = await asyncio.to_thread(self._load_contents)
        return None if state.get("memory_contents") == contents else {"memory_contents": contents}

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        return self._prompt.modify_request(request)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self.modify_request(request))

    def _blocked_edit(self, request: ToolCallRequest) -> ToolMessage | None:
        call = request.tool_call
        if str(call.get("name", "")).rsplit("/", 1)[-1] not in _WRITE_TOOLS:
            return None
        args = call.get("args") or {}
        canonical = self._workspace.system_rel("MEMORY.md")
        protected = {
            _normalized_path("MEMORY.md"),
            _normalized_path("/MEMORY.md"),
            _normalized_path(canonical),
            _normalized_path(f"/{canonical}"),
            _normalized_path(self._workspace.resolve_path("MEMORY.md")),
        }
        for key in ("file_path", "path", "source", "destination"):
            raw = args.get(key)
            if not isinstance(raw, str):
                continue
            matches = _normalized_path(raw) in protected
            if not matches:
                try:
                    matches = _normalized_path(self._workspace.resolve_path(raw)) in protected
                except (OSError, ValueError, PermissionError):
                    continue
            if matches:
                return ToolMessage(content=_READ_ONLY_MESSAGE, tool_call_id=str(call.get("id", "")), status="error")
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._blocked_edit(request) or handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        blocked = self._blocked_edit(request)
        return blocked if blocked is not None else await handler(request)
