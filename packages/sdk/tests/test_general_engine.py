from types import SimpleNamespace

from data_intelligence_sdk.core.types import ExecutionSpec, UserQuery
from data_intelligence_sdk.engines.general import GeneralPurposeEngine
from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext
from data_intelligence_sdk.runtime.mcp_client import MCPToolDefinition
from data_intelligence_sdk.runtime.selected_files import SelectedFilesScope
from data_intelligence_sdk.runtime.skills import WorkspaceSkill


def test_general_engine_prompt_keeps_method_hub_outside_sandbox() -> None:
    engine = GeneralPurposeEngine(llm=object())
    prompt = engine._system_prompt(
        ExecutionSpec(intent="general", objective="Find the latest revenue."),
        EngineRuntimeContext(
            mcp_client=object(),
            mcp_tools=(MCPToolDefinition(name="document_retrieve_context"),),
        ),
        UserQuery(text="Find the latest revenue."),
    )

    assert "call the matching method hub tool directly" in prompt.lower()
    assert "axiom_method_hub" not in prompt


def test_general_engine_prompt_explains_current_workspace_scope() -> None:
    engine = GeneralPurposeEngine(llm=object())
    prompt = engine._system_prompt(
        ExecutionSpec(intent="general", objective="Find the latest revenue."),
        EngineRuntimeContext(
            mcp_client=object(),
            mcp_tools=(MCPToolDefinition(name="corpus_bm25_search"),),
            workspace_id="workspace-1",
        ),
        UserQuery(text="Find the latest revenue."),
    )

    assert "workspace-1" in prompt
    assert "do not ask the user for a workspace_id" in prompt.lower()


def test_general_engine_prompt_does_not_eagerly_inject_workspace_skill_bodies() -> None:
    engine = GeneralPurposeEngine(llm=object())
    prompt = engine._system_prompt(
        ExecutionSpec(intent="general", objective="Validate the report."),
        EngineRuntimeContext(
            workspace_skills=(
                WorkspaceSkill(
                    skill_id="report-validation",
                    name="Report validation",
                    description="Validate reports before publishing.",
                    body="# Report validation\nCheck source totals first.",
                ),
            ),
        ),
        UserQuery(text="Validate the report."),
    )

    assert "Enabled workspace skills" not in prompt
    assert "Check source totals first." not in prompt


def test_general_engine_prompt_separates_memory_from_experience_skills() -> None:
    engine = GeneralPurposeEngine(llm=object())
    prompt = engine._system_prompt(
        ExecutionSpec(intent="general", objective="Remember this workflow."),
        EngineRuntimeContext(skill_registry_client=object()),
        UserQuery(text="Remember this workflow."),
    )

    assert "load_skill" in prompt
    assert "skill-creator" in prompt
    assert "scope `global`" in prompt
    assert "materialize_skill" in prompt
    assert "Use `memory` for facts and preferences" in prompt
    assert "User-owned skills are available across the user's workspaces" in prompt
    assert "Organization admins may apply a user-owned skill to everyone" in prompt
    assert "Never create organization-scoped skills" in prompt


def test_general_agent_binds_materialize_skill_from_method_hub(monkeypatch) -> None:
    captured = {}

    def agent_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "data_intelligence_sdk.engines.general.create_execute_python_tool",
        lambda runtime: type("Tool", (), {"name": "execute_python"})(),
    )
    monkeypatch.setattr(
        "data_intelligence_sdk.engines.general.DeepAgentSandboxBackend",
        lambda sandbox: sandbox,
    )
    engine = GeneralPurposeEngine(llm=object(), agent_factory=agent_factory)
    engine._register_minimal_profile = lambda: None
    runtime = EngineRuntimeContext(
        mcp_client=object(),
        mcp_tools=(
            MCPToolDefinition(
                name="materialize_skill",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "skill_markdown": {"type": "string"},
                        "scope": {"type": "string"},
                    },
                    "required": ["workspace_id", "skill_markdown"],
                },
            ),
        ),
        workspace_id="workspace-1",
        sandbox=object(),
    )

    engine._build_agent(
        SimpleNamespace(
            spec=ExecutionSpec(intent="general", objective="Save a workflow."),
            runtime=runtime,
            query=UserQuery(text="Remember this workflow."),
        )
    )

    assert "materialize_skill" in {tool.name for tool in captured["tools"]}


def test_general_engine_builds_llm_messages_from_conversation_history() -> None:
    engine = GeneralPurposeEngine(llm=object())
    messages = engine._conversation_messages(
        UserQuery(
            text="Who am I?",
            metadata={
                "history": [
                    {"role": "user", "content": "My name is Anh."},
                    {"role": "assistant", "content": "Nice to meet you, Anh."},
                ]
            },
        )
    )

    assert messages == [
        {"role": "user", "content": "My name is Anh."},
        {"role": "assistant", "content": "Nice to meet you, Anh."},
        {"role": "user", "content": "Who am I?"},
    ]


def test_general_engine_prompt_explains_selected_workspace_files() -> None:
    engine = GeneralPurposeEngine(llm=object())
    prompt = engine._system_prompt(
        ExecutionSpec(intent="general", objective="What is this file about?"),
        EngineRuntimeContext(
            mcp_client=object(),
            mcp_tools=(MCPToolDefinition(name="corpus_retrieve_context"),),
            selected_files_scope=SelectedFilesScope(document_ids=("document-1",)),
        ),
        UserQuery(text="What is this file about?"),
    )

    assert "document-1" in prompt
    assert "use the retrieval tools" in prompt.lower()
    assert "do not search the local filesystem" in prompt.lower()
    assert "do not ask the user for a local path" in prompt.lower()


def test_general_engine_prompt_explains_staged_selected_file_inputs() -> None:
    engine = GeneralPurposeEngine(llm=object())
    prompt = engine._system_prompt(
        ExecutionSpec(intent="general", objective="What is this file about?"),
        EngineRuntimeContext(
            selected_files_scope=SelectedFilesScope(document_ids=("document-1",)),
            execution_files=(
                {
                    "filename": "Thư_giới_thiệu.pdf",
                    "sandbox_path": "/workspace/runs/resp-1/inputs/Thư_giới_thiệu.pdf",
                },
            ),
        ),
        UserQuery(text="What is this file about?"),
    )

    assert "/workspace/runs/resp-1/inputs/Thư_giới_thiệu.pdf" in prompt
    assert "execute_python" in prompt
    assert "already staged" in prompt.lower()


def test_general_engine_prompt_prioritizes_direct_file_inputs_over_retrieval() -> None:
    engine = GeneralPurposeEngine(llm=object())
    prompt = engine._system_prompt(
        ExecutionSpec(intent="general", objective="Who is this letter about?"),
        EngineRuntimeContext(
            mcp_client=object(),
            mcp_tools=(MCPToolDefinition(name="corpus_retrieve_context"),),
            selected_files_scope=SelectedFilesScope(document_ids=("document-1",)),
            execution_files=(
                {
                    "filename": "letter.pdf",
                    "sandbox_path": "/workspace/runs/resp-1/inputs/letter.pdf",
                },
            ),
        ),
        UserQuery(text="Who is this letter about?"),
    )

    assert prompt.index("Direct input files") < prompt.index("Method Hub is enabled")
    assert "must call `execute_python`" in prompt.lower()
