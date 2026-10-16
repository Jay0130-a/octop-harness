"""Strong-memory prompt policy and the user-owned MEMORY.md write boundary.

File loading, session snapshots and prompt formatting belong to deepagents'
MemoryMiddleware. This module only supplies its prompt and guards file writes.
"""

from __future__ import annotations

import posixpath
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop_harness.backends.workspace import BackendWorkspace

STRONG_MEMORY_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
MEMORY.md is strong memory maintained manually by the user and included in full in chat context.
Its snapshot is loaded once at session start; manual edits apply to new sessions.
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


class StrongMemoryWriteGuard(AgentMiddleware[Any, Any]):
    """Protect user-maintained MEMORY.md from agent file-tool edits."""

    def __init__(self, workspace: BackendWorkspace) -> None:
        super().__init__()
        self._workspace = workspace

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
