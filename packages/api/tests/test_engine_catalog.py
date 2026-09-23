from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from data_intelligence_api.infrastructure.workflow.pipeline_factory import (
    create_example_pipeline,
)
from data_intelligence_sdk.core.types import EngineInput, EngineOutput
from data_intelligence_sdk.registry.engine_selector import LLMEngineSelector
from data_intelligence_sdk.runtime.config import OpenRouterSettings
from data_intelligence_sdk.runtime.llm_client import (
    ModelServiceChatModel,
    ModelServiceProfileLLMClient,
)


class _GeneralLLM:
    def invoke(self, messages: list[object]) -> object:
        del messages
        raise AssertionError("Catalog construction must not invoke the general LLM.")


class _StaticSelector:
    def select(self, request: object, engines: object) -> str:
        del request, engines
        return "general"


class _ReportEngine:
    name = "report"
    description = "Structured report generation test engine."

    def run(self, input: EngineInput) -> EngineOutput:
        del input
        return EngineOutput(engine_name=self.name, answer="Report")

    def run_markdown(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("Catalog construction must not run the report engine.")


def test_default_catalog_exposes_only_public_engine_names() -> None:
    pipeline = create_example_pipeline(
        llm=_GeneralLLM(),
        engine_selector=_StaticSelector(),  # type: ignore[arg-type]
        markdown_report_engine=_ReportEngine(),
        configure_default_sandbox=False,
    )

    assert [item.name for item in pipeline.engine_registry.descriptors()] == [
        "general",
        "reason",
        "report",
    ]


def test_engine_selector_uses_configured_routing_model_not_request_model(
    tmp_path,
) -> None:
    configured_model = "qwen/qwen3.7-flash"
    request_model = "cohere/north-mini-code:free"
    created_models: list[str | None] = []

    class _ConfigManager:
        def artifact_settings(self):
            return SimpleNamespace(root=str(tmp_path / "artifacts"))

        def openrouter_settings(self):
            return OpenRouterSettings(
                model=configured_model,
                api_key="test-key",
                base_url="https://example.test/v1",
            )

    class _RecordingLLMClient:
        def __init__(self, *, model=None, **kwargs):
            del kwargs
            self.model = model
            created_models.append(model)

    with patch(
        "data_intelligence_api.infrastructure.workflow.pipeline_factory."
        "OpenAICompatibleLLMClient",
        _RecordingLLMClient,
    ):
        pipeline = create_example_pipeline(
            llm=_GeneralLLM(),
            model=request_model,
            config_manager=_ConfigManager(),
            use_llm_spec_builder=True,
            configure_default_sandbox=False,
        )

    selector = pipeline.engine_registry._selector
    assert isinstance(selector, LLMEngineSelector)
    assert selector.llm_client.model == configured_model
    assert created_models == [request_model, configured_model]


def test_model_service_mode_uses_consumer_profile_for_general_and_routing(
    monkeypatch,
):
    monkeypatch.setenv("MODEL_SERVICE_URL", "http://model-service/api/v2")
    monkeypatch.setenv("MODEL_SERVICE_TOKEN", "service-token")
    monkeypatch.setenv("MODEL_SERVICE_CONSUMER_ID", "sdk")

    pipeline = create_example_pipeline(
        default_organization_id="org-a",
        workspace_id="workspace-a",
        configure_default_sandbox=False,
    )

    general = pipeline.engine_registry._engines["general"]
    selector = pipeline.engine_registry._selector
    assert isinstance(general.llm, ModelServiceChatModel)
    assert isinstance(general.llm._profile_client, ModelServiceProfileLLMClient)
    assert general.llm._profile_client.consumer_id == "data-intelligence"
    assert isinstance(selector, LLMEngineSelector)
    assert isinstance(selector.llm_client, ModelServiceProfileLLMClient)


def test_model_service_mode_passes_selected_model_resource_to_profile(
    monkeypatch,
):
    monkeypatch.setenv("MODEL_SERVICE_URL", "http://model-service/api/v2")
    monkeypatch.setenv("MODEL_SERVICE_TOKEN", "service-token")

    pipeline = create_example_pipeline(
        default_organization_id="org-a",
        workspace_id="workspace-a",
        model="model-resource-selected",
        configure_default_sandbox=False,
    )

    general = pipeline.engine_registry._engines["general"]
    assert general.llm._profile_client.model_resource_id == "model-resource-selected"


def test_model_service_mode_allows_empty_token_when_url_is_configured(monkeypatch):
    monkeypatch.setenv("MODEL_SERVICE_URL", "http://model-service/api/v2")
    monkeypatch.delenv("MODEL_SERVICE_TOKEN", raising=False)

    pipeline = create_example_pipeline(
        default_organization_id="org-a",
        workspace_id="workspace-a",
        configure_default_sandbox=False,
    )

    general = pipeline.engine_registry._engines["general"]
    assert general.llm._profile_client.context.service_token == ""


def test_model_service_mode_propagates_user_id_to_request_context(monkeypatch):
    monkeypatch.setenv("MODEL_SERVICE_URL", "http://model-service/api/v2")
    monkeypatch.delenv("MODEL_SERVICE_TOKEN", raising=False)

    pipeline = create_example_pipeline(
        default_organization_id="org-a",
        workspace_id="workspace-a",
        user_id="user-a",
        configure_default_sandbox=False,
    )

    context = pipeline.engine_registry._engines["general"].llm._profile_client.context
    assert context.to_model_context().headers()["X-User-ID"] == "user-a"


def test_model_resource_id_fails_fast_without_model_service(monkeypatch):
    monkeypatch.delenv("MODEL_SERVICE_URL", raising=False)
    monkeypatch.delenv("MODEL_SERVICE_TOKEN", raising=False)

    with pytest.raises(ValueError, match="MODEL_SERVICE_URL"):
        create_example_pipeline(
            model="c111ebf7-4c40-4ea0-87c2-09e2f6032edc",
            configure_default_sandbox=False,
        )
