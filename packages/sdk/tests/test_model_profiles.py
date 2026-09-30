from __future__ import annotations

import json

import pytest

from axiom_model_client import (
    ConsumerModelResolution,
    ConsumerModelResolutions,
)
from langchain_core.messages import HumanMessage

from data_intelligence_sdk.runtime.llm_client import (
    ModelServiceChatModel,
    ModelServiceProfileLLMClient,
)
from data_intelligence_sdk.runtime.run_context import ModelRunContext


class _RecordingProfileClient:
    def __init__(self) -> None:
        self.resolve_calls: list[tuple[object, str, list[str], int | None]] = []
        self.complete_calls: list[tuple[object, str, dict[str, object]]] = []
        self.complete_for_model_calls: list[tuple[object, str, dict[str, object]]] = []
        self.stream_calls: list[tuple[object, str, dict[str, object]]] = []
        self.stream_for_model_calls: list[tuple[object, str, dict[str, object]]] = []

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

    def complete_for_model(self, context, model_resource_id, **payload):
        self.complete_for_model_calls.append((context, model_resource_id, payload))
        return {"output": {"content": json.dumps({"ok": True})}}

    def stream(self, context, resolution_id, **payload):
        self.stream_calls.append((context, resolution_id, payload))
        yield {"type": "text.delta", "text": "The answer "}
        yield {"type": "text.delta", "text": "streams."}
        yield {"type": "response.completed"}

    def stream_for_model(self, context, model_resource_id, **payload):
        self.stream_for_model_calls.append((context, model_resource_id, payload))
        yield {"type": "text.delta", "text": "The answer "}
        yield {"type": "text.delta", "text": "streams."}
        yield {"type": "response.completed"}


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


def test_model_service_chat_model_streams_profile_text_deltas() -> None:
    transport = _RecordingProfileClient()
    profile_client = ModelServiceProfileLLMClient(
        "http://model-service/api/v2",
        context=_context(config_revision=7),
        model_client=transport,
    )
    model = ModelServiceChatModel(profile_client)

    chunks = list(model.stream([HumanMessage(content="hello")]))

    assert [chunk.text for chunk in chunks if chunk.text] == [
        "The answer ",
        "streams.",
    ]
    assert transport.resolve_calls[0][3] == 7
    assert transport.stream_calls[0][1] == "profile-llm"
    assert transport.complete_calls == []


def test_model_service_chat_model_streams_selected_model_resource() -> None:
    transport = _RecordingProfileClient()
    profile_client = ModelServiceProfileLLMClient(
        "http://model-service/api/v2",
        context=_context(),
        model_client=transport,
        model_resource_id="model-resource-selected",
    )
    model = ModelServiceChatModel(profile_client)

    chunks = list(model.stream([HumanMessage(content="hello")]))

    assert [chunk.text for chunk in chunks if chunk.text] == [
        "The answer ",
        "streams.",
    ]
    assert transport.stream_for_model_calls[0][1] == "model-resource-selected"
    assert transport.resolve_calls == []


def test_model_service_chat_model_preserves_streamed_tool_call_fragments() -> None:
    class ToolCallProfileClient(_RecordingProfileClient):
        def stream(self, context, resolution_id, **payload):
            self.stream_calls.append((context, resolution_id, payload))
            yield {
                "type": "tool_call.delta",
                "index": 0,
                "call_id": "call-1",
                "name": "ask_user",
                "arguments": '{"question":',
            }
            yield {
                "type": "tool_call.delta",
                "index": 0,
                "arguments": '"choose"}',
            }
            yield {"type": "response.completed"}

    transport = ToolCallProfileClient()
    profile_client = ModelServiceProfileLLMClient(
        "http://model-service/api/v2",
        context=_context(),
        model_client=transport,
    )
    model = ModelServiceChatModel(profile_client)

    chunks = list(model.stream([HumanMessage(content="help")]))

    tool_chunks = [chunk for chunk in chunks if chunk.tool_call_chunks]

    assert len(tool_chunks) == 2
    assert [chunk.tool_call_chunks[0]["args"] for chunk in tool_chunks] == [
        '{"question":',
        '"choose"}',
    ]
    assert tool_chunks[0].tool_call_chunks[0]["name"] == "ask_user"
    combined = tool_chunks[0] + tool_chunks[1]
    assert combined.tool_calls[0]["name"] == "ask_user"
    assert combined.tool_calls[0]["args"] == {"question": "choose"}
    assert combined.tool_calls[0]["id"] == "call-1"


def test_model_service_chat_model_surfaces_safe_stream_failure_code() -> None:
    class FailedProfileClient(_RecordingProfileClient):
        def stream(self, context, resolution_id, **payload):
            yield {
                "type": "response.failed",
                "code": "provider_http_429",
                "message": "sensitive upstream response",
            }

    model = ModelServiceChatModel(
        ModelServiceProfileLLMClient(
            "http://model-service/api/v2",
            context=_context(),
            model_client=FailedProfileClient(),
        )
    )

    with pytest.raises(RuntimeError, match="provider_http_429") as error:
        list(model.stream([HumanMessage(content="hello")]))

    assert "sensitive upstream response" not in str(error.value)
