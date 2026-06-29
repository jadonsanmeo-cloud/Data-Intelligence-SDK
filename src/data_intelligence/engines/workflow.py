"""Workflow engine placeholder."""

from __future__ import annotations

from data_intelligence.core.types import DataHubContext, EngineOutput, ExecutionSpec, UserContext


class WorkflowEngine:
    """Placeholder for multi-step or multi-engine workflows."""

    name = "workflow"

    def can_handle(self, spec: ExecutionSpec) -> bool:
        return spec.engine_hint == self.name or spec.intent.task_type == "workflow"

    def run(
        self,
        spec: ExecutionSpec,
        datahub: DataHubContext,
        user_context: UserContext | None = None,
    ) -> EngineOutput:
        raise NotImplementedError("Workflow execution is not part of the base scaffold.")
