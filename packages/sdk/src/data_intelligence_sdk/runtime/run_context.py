"""Runtime context passed to engines so they can record structured trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from axiom_model_client import (
    ConsumerModelResolution,
    ConsumerModelResolutions,
    ModelContext,
    ResolvedTasks,
    TaskResolution,
)

from data_intelligence_sdk.core.types import (
    EngineOutput,
    EngineStep,
    EngineTrace,
    EvidenceBundle,
    MethodCall,
    TraceStatus,
)

EventRecorder = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ModelRunContext:
    """Immutable organization and Model Service context for one AXIOM run.

    The service token is request-scoped authentication material. It is kept out
    of the wire ``ModelContext`` and never becomes part of an inference body.
    Resolutions are copied into a read-only mapping so a prepared run cannot
    accidentally switch organizations, workspaces, or Model Service revisions.
    """

    organization_id: str
    workspace_id: str | None
    consumer_service: str
    service_token: str = field(repr=False)
    run_id: str
    trace_id: str | None = None
    user_id: str | None = None
    config_revision: int | None = None
    resolutions: Mapping[str, TaskResolution] = field(default_factory=dict)
    profile_resolutions: Mapping[str, ConsumerModelResolution] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("organization_id", self.organization_id),
            ("consumer_service", self.consumer_service),
            ("run_id", self.run_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.service_token, str):
            raise ValueError("service_token must be a string")
        if self.workspace_id is not None and not self.workspace_id.strip():
            raise ValueError("workspace_id must be non-empty or None")
        if self.user_id is not None and not self.user_id.strip():
            raise ValueError("user_id must be non-empty or None")
        if self.trace_id is not None and not self.trace_id.strip():
            raise ValueError("trace_id must be non-empty or None")
        if self.config_revision is not None and (
            type(self.config_revision) is not int or self.config_revision < 0
        ):
            raise ValueError("config_revision must be a non-negative integer or None")
        copied = dict(self.resolutions)
        for task_id, resolution in copied.items():
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("resolution task IDs must be non-empty strings")
            if not isinstance(resolution, TaskResolution):
                raise TypeError("resolutions must contain TaskResolution values")
            if resolution.task_id != task_id:
                raise ValueError("resolution task ID does not match its mapping key")
        object.__setattr__(self, "resolutions", MappingProxyType(copied))
        profile_copied = dict(self.profile_resolutions)
        for role, resolution in profile_copied.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError("profile resolution roles must be non-empty strings")
            if not isinstance(resolution, ConsumerModelResolution):
                raise TypeError(
                    "profile_resolutions must contain ConsumerModelResolution values"
                )
            if resolution.role != role:
                raise ValueError(
                    "profile resolution role does not match its mapping key"
                )
        object.__setattr__(
            self, "profile_resolutions", MappingProxyType(profile_copied)
        )

    def to_model_context(self) -> ModelContext:
        """Return the credential-free context accepted by ``axiom_model_client``."""

        return ModelContext(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            consumer_service=self.consumer_service,
            run_id=self.run_id,
            trace_id=self.trace_id,
            user_id=self.user_id,
        )

    def with_resolutions(self, resolved: ResolvedTasks) -> "ModelRunContext":
        """Return a new context after adding immutable run resolutions."""

        if self.config_revision is not None and (
            resolved.config_revision != self.config_revision
        ):
            raise ValueError(
                "Model Service config revision changed during the AXIOM run"
            )
        merged = dict(self.resolutions)
        for resolution in resolved.resolutions:
            if resolution.config_revision != resolved.config_revision:
                raise ValueError("resolution config revision does not match its batch")
            previous = merged.get(resolution.task_id)
            if previous is not None and previous != resolution:
                raise ValueError(
                    f"task resolution changed during the AXIOM run: {resolution.task_id}"
                )
            merged[resolution.task_id] = resolution
        return ModelRunContext(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            consumer_service=self.consumer_service,
            service_token=self.service_token,
            run_id=self.run_id,
            trace_id=self.trace_id,
            user_id=self.user_id,
            config_revision=(
                self.config_revision
                if self.config_revision is not None
                else resolved.config_revision
            ),
            resolutions=merged,
            profile_resolutions=self.profile_resolutions,
        )

    def with_profile_resolutions(
        self, resolved: ConsumerModelResolutions
    ) -> "ModelRunContext":
        """Return a new context after adding immutable role resolutions."""

        if self.config_revision is not None and (
            resolved.config_revision != self.config_revision
        ):
            raise ValueError(
                "Model Service config revision changed during the AXIOM run"
            )
        merged = dict(self.profile_resolutions)
        for resolution in resolved.resolutions:
            if resolution.consumer_id != resolved.consumer_id:
                raise ValueError("profile resolution consumer does not match its batch")
            if resolution.config_revision != resolved.config_revision:
                raise ValueError(
                    "profile resolution config revision does not match its batch"
                )
            previous = merged.get(resolution.role)
            if previous is not None and previous != resolution:
                raise ValueError(
                    f"profile resolution changed during the AXIOM run: {resolution.role}"
                )
            merged[resolution.role] = resolution
        return ModelRunContext(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            consumer_service=self.consumer_service,
            service_token=self.service_token,
            run_id=self.run_id,
            trace_id=self.trace_id,
            user_id=self.user_id,
            config_revision=(
                self.config_revision
                if self.config_revision is not None
                else resolved.config_revision
            ),
            resolutions=self.resolutions,
            profile_resolutions=merged,
        )


# This alias keeps the AXIOM-specific name discoverable for callers while the
# more explicit ModelRunContext remains the canonical type in SDK internals.
AxiomRunContext = ModelRunContext


class EngineRunContext:
    """Collects structured execution trace while an engine runs.

    Engines should use this context to record steps, Method Hub calls, artifact
    references, and log references instead of inventing a trace format.
    """

    def __init__(
        self,
        event_recorder: EventRecorder | None = None,
    ) -> None:
        self.trace = EngineTrace()
        self._event_recorder = event_recorder

    def _record_event(
        self,
        *,
        phase: str,
        event_type: str,
        status: TraceStatus,
        payload: dict[str, Any],
    ) -> None:
        if self._event_recorder is not None:
            self._event_recorder(
                phase=phase,
                event_type=event_type,
                status=status,
                payload=payload,
            )

    def record_step(
        self,
        name: str,
        *,
        status: TraceStatus = "completed",
        description: str | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        artifact_refs: list[str] | None = None,
        log_refs: list[str] | None = None,
    ) -> EngineStep:
        step = EngineStep(
            name=name,
            status=status,
            description=description,
            inputs=inputs or {},
            outputs=outputs or {},
            artifact_refs=artifact_refs or [],
            log_refs=log_refs or [],
        )
        self.trace.steps.append(step)
        plan = step.outputs.get("plan")
        if plan is not None:
            self._record_event(
                phase="planning",
                event_type="plan.created",
                status=status,
                payload={"step_name": name, "plan": plan},
            )
        self._record_event(
            phase="engine",
            event_type="engine.step",
            status=status,
            payload=asdict(step),
        )
        return step

    def record_method_call(
        self,
        method_name: str,
        *,
        status: TraceStatus = "completed",
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        artifact_refs: list[str] | None = None,
        log_refs: list[str] | None = None,
    ) -> MethodCall:
        method_call = MethodCall(
            method_name=method_name,
            status=status,
            inputs=inputs or {},
            outputs=outputs or {},
            artifact_refs=artifact_refs or [],
            log_refs=log_refs or [],
        )
        self.trace.method_calls.append(method_call)
        self._record_event(
            phase="tool",
            event_type="tool.called",
            status=status,
            payload=asdict(method_call),
        )
        return method_call

    def add_artifact_ref(self, artifact_ref: str) -> None:
        self.trace.artifact_refs.append(artifact_ref)

    def add_log_ref(self, log_ref: str) -> None:
        self.trace.log_refs.append(log_ref)

    def build_output(
        self,
        *,
        engine_name: str,
        answer: str | None = None,
        result: Any = None,
        evidence: EvidenceBundle | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EngineOutput:
        return EngineOutput(
            engine_name=engine_name,
            answer=answer,
            result=result,
            evidence=evidence,
            trace=self.trace,
            metadata=metadata or {},
        )
