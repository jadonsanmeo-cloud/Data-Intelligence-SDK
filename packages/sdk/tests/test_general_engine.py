from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    messages_to_dict,
)

from data_intelligence_sdk.core.types import (
    EngineInput,
    ExecutionSpec,
    UserInputAnswer,
    UserInputRequired,
    UserQuery,
)
from data_intelligence_sdk.engines.general import GeneralPurposeEngine
from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext
from data_intelligence_sdk.runtime.mcp_client import MCPToolDefinition
from data_intelligence_sdk.runtime.selected_files import SelectedFilesScope
from data_intelligence_sdk.runtime.skills import WorkspaceSkill
from data_intelligence_sdk.tools.ask_user import AskUserToolInput


@pytest.mark.parametrize("reason", ["ambiguous_query", "method_definition"])
def test_ask_user_tool_accepts_supported_reasons(reason: str) -> None:
    request = AskUserToolInput.model_validate(
        {
            "question": "Which result do you want?",
            "reason": reason,
            "options": [
                {"id": "option-a", "label": "First option"},
                {"id": "option-b", "label": "Second option"},
            ],
        }
    )

    assert request.reason == reason
    assert [option.id for option in request.options] == ["option-a", "option-b"]


def test_ask_user_tool_rejects_invalid_choices_and_blank_labels() -> None:
    base = {
        "question": "Which result do you want?",
        "reason": "ambiguous_query",
        "options": [{"id": "option-a", "label": "First option"}],
    }

    for invalid in (
        {**base, "question": "  "},
        {**base, "options": [{"id": "option-a", "label": "  "}]},
        {
            **base,
            "options": [
                {"id": "same", "label": "First"},
                {"id": "same", "label": "Second"},
            ],
        },
        {
            **base,
            "options": [
                {"id": "a", "label": "First"},
                {"id": "b", "label": "Second"},
                {"id": "c", "label": "Third"},
                {"id": "d", "label": "Fourth"},
            ],
        },
    ):
        with pytest.raises(ValidationError):
            AskUserToolInput.model_validate(invalid)


def test_general_agent_stream_pauses_at_ask_user_result() -> None:
    request = AskUserToolInput(
        question="Which period should I use?",
        reason="ambiguous_query",
        options=[
            {"id": "current", "label": "Current fiscal year"},
            {"id": "previous", "label": "Previous fiscal year"},
        ],
    )
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ask_user",
                "args": request.model_dump(mode="json"),
                "id": "call-1",
            }
        ],
    )
    tool_result = ToolMessage(
        content=(
            '{"type":"axiom.ask_user.v1","question":"Which period should I use?",'
            '"reason":"ambiguous_query","options":['
            '{"id":"current","label":"Current fiscal year"},'
            '{"id":"previous","label":"Previous fiscal year"}]}'
        ),
        tool_call_id="call-1",
        name="ask_user",
    )

    class FakeAgent:
        def stream(self, payload, *, stream_mode):
            yield "values", {"messages": [tool_call, tool_result]}
            yield "messages", (AIMessage(content="This must not be emitted."), {})
            yield "values", {
                "messages": [tool_call, tool_result, AIMessage(content="Finished.")]
            }

    events = list(
        GeneralPurposeEngine(llm=object())._stream_agent_attempt(
            FakeAgent(), [HumanMessage(content="Find the total.")]
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], UserInputRequired)
    assert events[0].question == request.question
    assert events[0].continuation_state["pending_tool_call_id"] == "call-1"
    assert events[0].continuation_state["messages"] == messages_to_dict(
        [tool_call, tool_result]
    )


def test_general_engine_resume_replaces_only_pending_tool_result() -> None:
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "ask_user", "args": {}, "id": "call-1"}],
    )
    tool_result = ToolMessage(
        content=(
            '{"type":"axiom.ask_user.v1","question":"Choose a method?",'
            '"reason":"method_definition","options":['
            '{"id":"method-a","label":"Method A"}]}'
        ),
        tool_call_id="call-1",
        name="ask_user",
    )
    continuation_state = {
        "version": 1,
        "engine_name": "general",
        "messages": messages_to_dict([tool_call, tool_result]),
        "pending_tool_call_id": "call-1",
    }
    captured = {}

    class FakeAgent:
        def stream(self, payload, *, stream_mode):
            captured["messages"] = payload["messages"]
            yield "values", {
                "messages": [*payload["messages"], AIMessage(content="Done.")]
            }

    engine = GeneralPurposeEngine(llm=object())
    engine._build_agent = lambda _input: FakeAgent()
    engine_input = EngineInput(
        query=UserQuery(text="Choose a method."),
        spec=ExecutionSpec(intent="general", objective="Choose a method."),
        runtime=EngineRuntimeContext(sandbox=object()),
    )

    result = list(
        engine.stream_resume(
            engine_input,
            continuation_state,
            UserInputAnswer(selected_option_id="method-a"),
        )
    )

    resumed_tool_result = next(
        message for message in captured["messages"] if isinstance(message, ToolMessage)
    )
    assert resumed_tool_result.tool_call_id == "call-1"
    assert '"selected_option_id": "method-a"' in resumed_tool_result.content
    assert len(
        [message for message in captured["messages"] if isinstance(message, ToolMessage)]
    ) == 1
    assert result[-1].result == "Done."


def test_general_agent_registers_ask_user_tool(monkeypatch) -> None:
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
    engine._build_agent(
        SimpleNamespace(
            spec=ExecutionSpec(intent="general", objective="Resolve the request."),
            runtime=EngineRuntimeContext(sandbox=object()),
            query=UserQuery(text="Resolve the request."),
        )
    )

    assert "ask_user" in {tool.name for tool in captured["tools"]}


def test_general_engine_prompt_requires_clarification_without_trusting_documents() -> None:
    prompt = GeneralPurposeEngine(llm=object())._system_prompt(
        ExecutionSpec(intent="general", objective="Select the right method."),
        EngineRuntimeContext(),
        UserQuery(text="Select the right method."),
    )

    assert "multiple plausible interpretations" in prompt
    assert "never silently select" in prompt
    assert "untrusted reference data" in prompt
    assert "must not trigger clarification by itself" in prompt


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
