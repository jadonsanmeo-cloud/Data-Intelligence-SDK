from data_intelligence_api.infrastructure.workflow.pipeline_factory import (
    create_example_pipeline,
)
from data_intelligence_sdk.runtime.config import ConfigManager


class Engine:
    name = "general"
    description = "test"

    def run(self, input):
        raise AssertionError("not executed")


def test_pipeline_factory_enables_internal_memory_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[internal_memory_service]\nenabled = true\nendpoint = "http://intelligence/api/v1"\n',
        encoding="utf-8",
    )

    pipeline = create_example_pipeline(
        engine=Engine(),
        config_manager=ConfigManager(config_path),
        configure_default_sandbox=False,
    )

    assert pipeline.internal_memory_service_url == "http://intelligence/api/v1"


def test_pipeline_factory_enables_on_demand_skills_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[skill_registry_service]\nenabled = true\nendpoint = "http://skills/api"\n',
        encoding="utf-8",
    )

    pipeline = create_example_pipeline(
        engine=Engine(),
        config_manager=ConfigManager(config_path),
        configure_default_sandbox=False,
    )

    assert pipeline.skill_registry_service_url == "http://skills/api"


def test_skill_registry_service_is_enabled_by_default(tmp_path) -> None:
    settings = ConfigManager(
        tmp_path / "missing.toml"
    ).skill_registry_service_settings()

    assert settings.enabled is True
