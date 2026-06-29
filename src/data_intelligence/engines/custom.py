"""Custom engine placeholder."""

from __future__ import annotations

from data_intelligence.core.types import DataHubContext, EngineOutput, ExecutionSpec, UserContext


class CustomEngine:
    """Placeholder for project-specific engine integrations."""

    name = "custom"

    def can_handle(self, spec: ExecutionSpec) -> bool:
        return spec.engine_hint == self.name or spec.intent == self.name

    def run(
        self,
        spec: ExecutionSpec,
        datahub: DataHubContext,
        user_context: UserContext | None = None,
    ) -> EngineOutput:
        raise NotImplementedError("Custom execution is not part of the base scaffold.")
