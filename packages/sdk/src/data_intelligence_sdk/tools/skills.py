from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool, tool

from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext


def create_skill_tools(runtime: EngineRuntimeContext) -> list[BaseTool]:
    """Create request-scoped skill loading tools for one agent run."""

    client = runtime.skill_registry_client
    if client is None:
        return []

    @tool
    def load_skill(
        skill_name: str,
        scope: Literal["global", "organization", "workspace", "personal"] | None = None,
    ) -> str:
        """Load an accessible skill by name when its reusable workflow is relevant."""

        skill = client.load_skill(skill_name, scope=scope)
        return (
            f"Skill: {skill.name}\nScope: {skill.scope}\n"
            f"Description: {skill.description}\n\n{skill.body}"
        )

    return [load_skill]
