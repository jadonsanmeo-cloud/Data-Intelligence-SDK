from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext
from data_intelligence_sdk.runtime.skills import WorkspaceSkill
from data_intelligence_sdk.tools.skills import create_skill_tools


class _SkillRegistryClient:
    def load_skill(
        self, skill_name: str, *, scope: str | None = None
    ) -> WorkspaceSkill:
        assert skill_name == "report-validation"
        assert scope == "workspace"
        return WorkspaceSkill(
            skill_id="skill-1",
            name=skill_name,
            description="Validate financial reports.",
            body="# Steps\nCheck totals.",
            scope="workspace",
        )


def test_load_skill_tool_returns_prompt_and_context() -> None:
    tools = create_skill_tools(
        EngineRuntimeContext(skill_registry_client=_SkillRegistryClient())
    )
    load_skill = next(tool for tool in tools if tool.name == "load_skill")

    result = load_skill.invoke(
        {"skill_name": "report-validation", "scope": "workspace"}
    )

    assert "Validate financial reports." in result
    assert "Check totals." in result
    assert "workspace" in result
