from __future__ import annotations

import json

from axiom_model_client import (
    ConsumerModelResolution,
    ConsumerModelResolutions,
)

from data_intelligence_sdk.runtime.llm_client import ModelServiceProfileLLMClient
from data_intelligence_sdk.runtime.run_context import ModelRunContext


class _RecordingProfileClient:
    def __init__(self) -> None:
        self.resolve_calls: list[tuple[object, str, list[str], int | None]] = []
        self.complete_calls: list[tuple[object, str, dict[str, object]]] = []

    def resolve_consumer_profile(
        self,
        context,
        consumer_id,
        roles,
        *,
        config_revision=None,
    ):
        self.resolve_calls.append((context, consumer_id, list(roles), config_revision))
        revision = config_revision if config_revision is not None else 4
        return ConsumerModelResolutions(
            run_id=context.run_id,
            consumer_id="data-intelligence",
            config_revision=revision,
            resolutions=tuple(
                ConsumerModelResolution(
                    resolution_id=f"profile-{role}",
                    consumer_id="data-intelligence",
                    role=role,
                    provider_id="platform-provider",
                    model_id=f"model-{role}",
                    config_revision=revision,
                    assignment_revision=2,
                )
                for role in roles
            ),
        )

    def complete(self, context, resolution_id, **payload):
        self.complete_calls.append((context, resolution_id, payload))
        return {"output": {"content": json.dumps({"ok": True})}}


def _context(*, config_revision: int | None = None) -> ModelRunContext:
    return ModelRunContext(
        organization_id="org-a",
        workspace_id="workspace-a",
        consumer_service="data-intelligence-api",
        service_token="service-token",
        run_id="run-a",
        config_revision=config_revision,
    )


def test_profile_llm_client_resolves_canonical_consumer_once_and_pins_revision() -> (
    None
):
    transport = _RecordingProfileClient()
    client = ModelServiceProfileLLMClient(
        "http://model-service/api/v2",
        context=_context(config_revision=7),
        model_client=transport,
        consumer_id="sdk",
    )

    assert client.complete_json([], stage="spec-builder") == {"ok": True}
    assert client.complete_text([], stage="spec-builder") == json.dumps({"ok": True})

    assert len(transport.resolve_calls) == 1
    assert transport.resolve_calls[0][1:] == ("data-intelligence", ["llm"], 7)
    assert [call[1] for call in transport.complete_calls] == [
        "profile-llm",
        "profile-llm",
    ]
    assert client.context.config_revision == 7
    assert client.context.profile_resolutions["llm"].resolution_id == "profile-llm"
