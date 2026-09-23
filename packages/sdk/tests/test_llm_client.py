from __future__ import annotations

import json

import pytest

from axiom_model_client import ResolvedTasks, TaskResolution
from data_intelligence_sdk.runtime.llm_client import (
    ModelServiceLLMClient,
    ModelServiceProfileLLMClient,
)
from data_intelligence_sdk.runtime.run_context import ModelRunContext


class _RecordingModelClient:
    def __init__(self, *, resolved_revision: int | None = None) -> None:
        self.resolved_revision = resolved_revision
        self.resolve_calls: list[tuple[object, list[str], int | None]] = []
        self.complete_for_model_calls: list[tuple[object, str, dict[str, object]]] = []
        self.complete_calls: list[tuple[object, str, dict[str, object]]] = []

    def resolve_tasks(self, context, task_ids, *, config_revision=None):
        self.resolve_calls.append((context, list(task_ids), config_revision))
        revision = (
            self.resolved_revision
            if self.resolved_revision is not None
            else (config_revision if config_revision is not None else 1)
        )
        return ResolvedTasks(
            config_revision=revision,
            resolutions=tuple(
                TaskResolution(
                    resolution_id=f"resolution-{task_id}",
                    task_id=task_id,
                    provider_id="provider-from-model-service",
                    model_id="model-from-model-service",
                    config_revision=revision,
                    assignment_revision=3,
                )
                for task_id in task_ids
            ),
        )

    def complete(self, context, resolution_id, **payload):
        self.complete_calls.append((context, resolution_id, payload))
        return {"output": {"content": json.dumps({"engine_name": "general"})}}

    def complete_for_model(self, context, model_resource_id, **payload):
        self.complete_for_model_calls.append((context, model_resource_id, payload))
        return {"output": {"content": json.dumps({"engine_name": "selected"})}}


def _context(*, config_revision: int | None = None) -> ModelRunContext:
    return ModelRunContext(
        organization_id="org-a",
        workspace_id="workspace-a",
        consumer_service="data-intelligence-api",
        service_token="service-token",
        run_id="run-a",
        trace_id="trace-a",
        user_id="user-a",
        config_revision=config_revision,
    )


@pytest.mark.parametrize(
    ("stage", "task_id"),
    [
        ("datahub-clusterer", "data.cluster"),
        ("data-selector", "data.select"),
        ("cluster-spec-selector", "spec.select_cluster"),
        ("spec-builder", "spec.generate"),
        ("markdown-spec-builder", "spec.generate_markdown"),
        ("engine-selector", "execution.select_engine"),
        ("chat.answer", "chat.answer"),
        ("chat.synthesize", "chat.synthesize"),
        ("report.generate", "report.generate"),
    ],
)
def test_model_service_client_resolves_each_stage_to_its_task(
    stage: str,
    task_id: str,
) -> None:
    transport = _RecordingModelClient()
    client = ModelServiceLLMClient(
        "http://model-service/api/v2",
        context=_context(config_revision=9),
        model_client=transport,
    )

    result = client.complete_json([{"role": "user", "content": "hello"}], stage=stage)

    assert result == {"engine_name": "general"}
    assert [call[1] for call in transport.resolve_calls] == [[task_id]]
    assert transport.resolve_calls[0][2] == 9
    assert transport.complete_calls[0][1] == f"resolution-{task_id}"
    assert "provider_id" not in transport.complete_calls[0][2]
    assert "model_id" not in transport.complete_calls[0][2]
    assert "api_key" not in transport.complete_calls[0][2]


def test_model_service_client_reuses_resolution_and_original_revision() -> None:
    transport = _RecordingModelClient()
    client = ModelServiceLLMClient(
        "http://model-service/api/v2",
        context=_context(config_revision=12),
        model_client=transport,
    )

    client.complete_json([], stage="spec-builder")
    client.complete_text([], stage="spec-builder")

    assert len(transport.resolve_calls) == 1
    assert transport.resolve_calls[0][2] == 12
    assert [call[1] for call in transport.complete_calls] == [
        "resolution-spec.generate",
        "resolution-spec.generate",
    ]
    assert client.context.config_revision == 12
    assert client.context.resolutions["spec.generate"].resolution_id == (
        "resolution-spec.generate"
    )


def test_profile_client_uses_selected_model_resource_instead_of_default_profile() -> (
    None
):
    transport = _RecordingModelClient()
    client = ModelServiceProfileLLMClient(
        "http://model-service/api/v2",
        context=_context(),
        model_client=transport,
        model_resource_id="model-resource-selected",
    )

    result = client.complete_json(
        [{"role": "user", "content": "hello"}],
        stage="chat.answer",
    )

    assert result == {"engine_name": "selected"}
    assert transport.complete_calls == []
    assert transport.complete_for_model_calls[0][1] == "model-resource-selected"


def test_model_run_context_is_immutable_and_does_not_expose_provider_credentials() -> (
    None
):
    context = _context(config_revision=2)

    with pytest.raises(AttributeError):
        context.config_revision = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        context.resolutions["chat.answer"] = object()  # type: ignore[index]

    wire_context = context.to_model_context()
    assert wire_context.organization_id == "org-a"
    assert wire_context.workspace_id == "workspace-a"
    assert not hasattr(wire_context, "api_key")


def test_model_service_client_rejects_unknown_stage_and_provider_aliases() -> None:
    transport = _RecordingModelClient()
    client = ModelServiceLLMClient(
        "http://model-service/api/v2",
        context=_context(),
        model_client=transport,
    )

    with pytest.raises(ValueError, match="Unknown AXIOM model stage"):
        client.complete_json([], stage="unregistered-stage")
    with pytest.raises(ValueError, match="provider/model aliases"):
        client.complete_json([], stage="chat.answer", model="provider-alias")


def test_axiom_client_constructor_does_not_accept_provider_api_key() -> None:
    with pytest.raises(TypeError):
        ModelServiceLLMClient(  # type: ignore[call-arg]
            "http://model-service/api/v2",
            context=_context(),
            api_key="provider-key",
        )


def test_model_service_revision_mismatch_fails_closed() -> None:
    transport = _RecordingModelClient(resolved_revision=8)
    client = ModelServiceLLMClient(
        "http://model-service/api/v2",
        context=_context(config_revision=7),
        model_client=transport,
    )

    with pytest.raises(ValueError, match="config revision"):
        client.complete_json([], stage="chat.answer")
    assert transport.complete_calls == []
