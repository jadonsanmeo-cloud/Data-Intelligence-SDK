"""Base engine contract."""

from __future__ import annotations

from typing import Protocol

from data_intelligence.core.types import DataHubContext, EngineOutput, ExecutionSpec, UserContext


class Engine(Protocol):
    """Executable unit selected by the engine registry."""

    @property
    def name(self) -> str:
        """Stable engine identifier."""

    def can_handle(self, spec: ExecutionSpec) -> bool:
        """Return whether this engine is suitable for a spec."""

    def run(
        self,
        spec: ExecutionSpec,
        datahub: DataHubContext,
        user_context: UserContext | None = None,
    ) -> EngineOutput:
        """Execute the spec and return raw engine output."""
