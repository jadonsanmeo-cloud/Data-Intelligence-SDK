from types import SimpleNamespace
import json

import pytest
from pydantic import ValidationError
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    messages_to_dict,
)

from data_intelligence_sdk.core.types import (
    EngineInput,
    ExecutionSpec,
    SafeguardAssessment,
    SafeguardFinding,
    UserInputAnswer,
    UserInputRequired,
    UserQuery,
)
from data_intelligence_sdk.engines.general import GeneralPurposeEngine
from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext
from data_intelligence_sdk.runtime.mcp_client import MCPToolDefinition
from data_intelligence_sdk.runtime.run_context import EngineRunContext
from data_intelligence_sdk.runtime.selected_files import SelectedFilesScope
from data_intelligence_sdk.runtime.skills import WorkspaceSkill
from data_intelligence_sdk.tools.ask_user import AskUserToolInput
from data_intelligence_sdk.tools.answerability import AnswerabilityAssessmentInput


class CitationModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.messages = []

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        return AIMessage(content=self.response)


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
            yield (
                "values",
                {"messages": [tool_call, tool_result, AIMessage(content="Finished.")]},
            )

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
            yield (
                "values",
                {"messages": [*payload["messages"], AIMessage(content="Done.")]},
            )

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
    assert (
        len(
            [
                message
                for message in captured["messages"]
                if isinstance(message, ToolMessage)
            ]
        )
        == 1
    )
    assert result[-1].result == "Done."


def _answerability_payload(decision: str) -> dict[str, object]:
    findings = [
        {
            "id": "evidence-gap",
            "category": "insufficient_evidence",
            "severity": "high",
            "title": "Missing source evidence",
            "detail": "The selected period has no supporting source records.",
            "impact": "The requested total cannot be verified.",
            "affected_scope": "selected period",
            "evidence_refs": [],
            "blocking": False,
        }
    ]
    payload: dict[str, object] = {"decision": decision, "findings": findings}
    if decision == "needs_user_input":
        payload.update(
            {
                "question": "How should I proceed?",
                "reason": "insufficient_evidence",
                "options": [
                    {"id": "continue", "label": "Continue with available evidence"},
                    {"id": "add-source", "label": "Add another source"},
                ],
            }
        )
    return payload


def test_general_engine_prompt_requires_internal_safeguard_review() -> None:
    prompt = GeneralPurposeEngine(llm=object())._system_prompt(
        ExecutionSpec(intent="general", objective="Calculate revenue."),
        EngineRuntimeContext(),
        UserQuery(text="Calculate revenue from the selected data."),
    )

    assert "assess the evidence internally" in prompt.lower()
    assert "do not call a tool to report the assessment" in prompt.lower()
    assert "assess_answerability" not in prompt
    assert "connection_risk" in prompt
    assert "source_conflict" in prompt
    assert "insufficient_evidence" in prompt
    assert "never assume an unverifiable connection is safe" in prompt.lower()
    assert "before joining sources" in prompt.lower()


def test_data_backed_answer_streams_without_structured_safeguard_assessment() -> None:
    runtime = EngineRuntimeContext()
    runtime.run_context.mark_data_access()

    class FakeAgent:
        def stream(self, payload, *, stream_mode):
            answer = AIMessageChunk(content="The workbook has 12 records.")
            yield "messages", (answer, {})
            yield "values", {"messages": [AIMessage(content=answer.content)]}

    events = list(
        GeneralPurposeEngine(llm=object())._stream_agent_attempt(
            FakeAgent(),
            [HumanMessage(content="Summarize the workbook.")],
            runtime=runtime,
        )
    )

    assert "The workbook has 12 records." in events


def test_general_engine_stream_pauses_on_answerability_assessment() -> None:
    assessment = AnswerabilityAssessmentInput.model_validate(
        _answerability_payload("needs_user_input")
    )
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "assess_answerability",
                "args": assessment.model_dump(mode="json"),
                "id": "assessment-call-1",
            }
        ],
    )
    tool_result = ToolMessage(
        content=json.dumps(
            {
                "type": "axiom.answerability.v1",
                **assessment.model_dump(mode="json"),
            }
        ),
        tool_call_id="assessment-call-1",
        name="assess_answerability",
    )

    class FakeAgent:
        def stream(self, payload, *, stream_mode):
            yield "values", {"messages": [tool_call, tool_result]}
            yield "messages", (AIMessage(content="Unsupported conclusion."), {})
            yield "values", {"messages": [tool_call, tool_result]}

    runtime = EngineRuntimeContext()
    runtime.run_context.mark_data_access()
    events = list(
        GeneralPurposeEngine(llm=object())._stream_agent_attempt(
            FakeAgent(),
            [HumanMessage(content="Calculate the requested total.")],
            runtime=runtime,
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], UserInputRequired)
    assert events[0].reason == "insufficient_evidence"
    assert events[0].safeguard_assessment.decision == "needs_user_input"
    assert events[0].continuation_state["pending_tool_name"] == "assess_answerability"


def test_blocked_connection_stops_agent_before_another_tool_call() -> None:
    payload = {
        "decision": "blocked",
        "findings": [
            {
                "id": "unsafe-join",
                "category": "connection_risk",
                "severity": "critical",
                "title": "Unsafe source mapping",
                "detail": "The join key and cardinality cannot be verified.",
                "evidence_refs": [],
                "blocking": True,
            }
        ],
    }
    assessment = AnswerabilityAssessmentInput.model_validate(payload)
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "assess_answerability",
                "args": assessment.model_dump(mode="json"),
                "id": "blocked-assessment-call",
            }
        ],
    )
    tool_result = ToolMessage(
        content=json.dumps(
            {"type": "axiom.answerability.v1", **assessment.model_dump(mode="json")}
        ),
        tool_call_id="blocked-assessment-call",
        name="assess_answerability",
    )
    runtime = EngineRuntimeContext()
    runtime.run_context.set_safeguard_assessment(assessment.to_core())
    continued = False

    class FakeAgent:
        def stream(self, payload, *, stream_mode):
            nonlocal continued
            yield "values", {"messages": [tool_call, tool_result]}
            continued = True
            yield "messages", (AIMessage(content="Unsafe conclusion."), {})

    events = list(
        GeneralPurposeEngine(llm=object())._stream_agent_attempt(
            FakeAgent(),
            [HumanMessage(content="Join these two sources.")],
            runtime=runtime,
        )
    )

    assert continued is False
    assert not any(isinstance(event, str) for event in events)


def test_resume_replaces_pending_answerability_call_after_validating_option() -> None:
    assessment = AnswerabilityAssessmentInput.model_validate(
        _answerability_payload("needs_user_input")
    )
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "assess_answerability",
                "args": assessment.model_dump(mode="json"),
                "id": "assessment-call-1",
            }
        ],
    )
    tool_result = ToolMessage(
        content=json.dumps(
            {
                "type": "axiom.answerability.v1",
                **assessment.model_dump(mode="json"),
            }
        ),
        tool_call_id="assessment-call-1",
        name="assess_answerability",
    )
    continuation_state = {
        "version": 1,
        "engine_name": "general",
        "messages": messages_to_dict([tool_call, tool_result]),
        "pending_tool_call_id": "assessment-call-1",
        "pending_tool_name": "assess_answerability",
        "safeguard_required": True,
    }
    captured = {}

    class FakeAgent:
        def stream(self, payload, *, stream_mode):
            captured["messages"] = payload["messages"]
            yield (
                "values",
                {"messages": [*payload["messages"], AIMessage(content="Done.")]},
            )

    engine = GeneralPurposeEngine(llm=object())
    engine._build_agent = lambda _input: FakeAgent()
    list(
        engine.stream_resume(
            EngineInput(
                query=UserQuery(text="Calculate the requested total."),
                spec=ExecutionSpec(intent="general", objective="Calculate total."),
                runtime=EngineRuntimeContext(sandbox=object()),
            ),
            continuation_state,
            UserInputAnswer(selected_option_id="continue"),
        )
    )

    resumed_tool_result = next(
        message for message in captured["messages"] if isinstance(message, ToolMessage)
    )
    assert resumed_tool_result.name == "assess_answerability"
    assert '"selected_option_id": "continue"' in resumed_tool_result.content


def test_data_backed_output_does_not_require_structured_assessment() -> None:
    engine = GeneralPurposeEngine(llm=object())
    runtime = EngineRuntimeContext()
    runtime.run_context.mark_data_access()

    output = engine._build_output(
        ExecutionSpec(intent="general", objective="Calculate total."),
        runtime,
        "A total that was never validated.",
    )

    assert output.result == "A total that was never validated."
    assert "safeguard_assessment" not in output.metadata


def test_blocking_connection_assessment_overrides_model_answer() -> None:
    engine = GeneralPurposeEngine(llm=object())
    runtime = EngineRuntimeContext()
    runtime.run_context.mark_data_access()
    runtime.run_context.set_safeguard_assessment(
        SafeguardAssessment(
            decision="blocked",
            findings=[
                SafeguardFinding(
                    id="unsafe-join",
                    category="connection_risk",
                    severity="critical",
                    title="Unsafe data connection",
                    detail="The source mapping is ambiguous.",
                    blocking=True,
                )
            ],
        )
    )

    output = engine._build_output(
        ExecutionSpec(intent="general", objective="Calculate total."),
        runtime,
        "A conclusion from an unsafe join.",
    )

    assert "unsafe join" not in output.result
    assert output.metadata["safeguard_assessment"]["decision"] == "blocked"


def test_clear_assessment_preserves_answer_and_nonblocking_findings() -> None:
    engine = GeneralPurposeEngine(llm=object())
    runtime = EngineRuntimeContext()
    runtime.run_context.mark_data_access()
    runtime.run_context.set_safeguard_assessment(
        SafeguardAssessment(
            decision="clear",
            findings=[
                SafeguardFinding(
                    id="partial-coverage",
                    category="data_quality",
                    severity="low",
                    title="Partial source coverage",
                    detail="One requested week has no source records.",
                    impact="The result covers the remaining weeks only.",
                    affected_scope="Week 12",
                    evidence_refs=["source://weekly-ledger"],
                )
            ],
        )
    )

    output = engine._build_output(
        ExecutionSpec(intent="general", objective="Calculate average."),
        runtime,
        "The average is 12 for the available weeks.",
    )

    assert output.result == "The average is 12 for the available weeks."
    assert output.metadata["safeguard_assessment"] == {
        "decision": "clear",
        "findings": [
            {
                "id": "partial-coverage",
                "category": "data_quality",
                "severity": "low",
                "title": "Partial source coverage",
                "detail": "One requested week has no source records.",
                "impact": "The result covers the remaining weeks only.",
                "affected_scope": "Week 12",
                "evidence_refs": ["source://weekly-ledger"],
                "blocking": False,
            }
        ],
    }


def test_data_backed_output_gets_inline_citations_without_safeguard_tool() -> None:
    model = CitationModel(
        json.dumps(
            {
                "claims": [
                    {
                        "text": "Revenue was 12 million.",
                        "evidence_ids": ["ev-source-1"],
                    }
                ]
            }
        )
    )
    engine = GeneralPurposeEngine(llm=model)
    runtime_events = []
    runtime = EngineRuntimeContext(
        run_context=EngineRunContext(
            event_recorder=lambda **event: runtime_events.append(event) or {}
        )
    )
    runtime.run_context.mark_data_access()
    runtime.run_context.register_citation_evidence(
        source="finance.pdf",
        locator="page 4 · chunk 12",
        excerpt="Revenue was 12 million.",
        document_id="document-1",
        content_id="chunk-12",
    )
    evidence_id = runtime.run_context.citation_evidence[0].id
    model.response = json.dumps(
        {"claims": [{"text": "Revenue was 12 million.", "evidence_ids": [evidence_id]}]}
    )

    output = engine._build_output(
        ExecutionSpec(intent="general", objective="What was revenue?"),
        runtime,
        "Revenue was 12 million.",
    )

    assert model.calls == 1
    assert output.result.endswith(f"[1](axiom-citation://{evidence_id})")
    assert output.metadata["citation_status"] == "complete"
    assert output.metadata["citation_sources"][0]["source"] == "finance.pdf"
    attribution_events = [
        event
        for event in runtime_events
        if event["payload"].get("name") == "citation_attribution"
    ]
    assert [
        (event["status"], event["payload"]["description"])
        for event in attribution_events
    ] == [
        ("running", "Linking sources…"),
        ("completed", "Source linking complete."),
    ]


def test_data_backed_answer_discloses_unavailable_citations_without_sources() -> None:
    engine = GeneralPurposeEngine(llm=object())
    runtime_events = []
    runtime = EngineRuntimeContext(
        run_context=EngineRunContext(
            event_recorder=lambda **event: runtime_events.append(event) or {}
        )
    )
    runtime.run_context.mark_data_access()
    runtime.run_context.set_safeguard_assessment(SafeguardAssessment(decision="clear"))

    output = engine._build_output(
        ExecutionSpec(intent="general", objective="Calculate total."),
        runtime,
        "The total is 12.",
    )

    assert output.result == "The total is 12."
    assert output.metadata["citation_status"] == "unavailable"
    assert output.metadata["citation_sources"] == []
    attribution_events = [
        event
        for event in runtime_events
        if event["payload"].get("name") == "citation_attribution"
    ]
    assert [event["status"] for event in attribution_events] == [
        "running",
        "completed",
    ]


def test_citation_attribution_failure_ends_progress_event(monkeypatch) -> None:
    runtime_events = []
    runtime = EngineRuntimeContext(
        run_context=EngineRunContext(
            event_recorder=lambda **event: runtime_events.append(event) or {}
        )
    )
    runtime.run_context.mark_data_access()
    runtime.run_context.set_safeguard_assessment(SafeguardAssessment(decision="clear"))

    def raise_attribution_error(*args, **kwargs):
        raise RuntimeError("attribution failed")

    monkeypatch.setattr(
        "data_intelligence_sdk.engines.general.attribute_answer_citations",
        raise_attribution_error,
    )

    output = GeneralPurposeEngine(llm=object())._build_output(
        ExecutionSpec(intent="general", objective="Calculate total."),
        runtime,
        "The total is 12.",
    )

    assert output.metadata["citation_status"] == "unavailable"
    attribution_events = [
        event
        for event in runtime_events
        if event["payload"].get("name") == "citation_attribution"
    ]
    assert [event["status"] for event in attribution_events] == [
        "running",
        "failed",
    ]


def test_non_data_answer_skips_citation_attribution() -> None:
    engine = GeneralPurposeEngine(llm=object())
    runtime = EngineRuntimeContext()

    output = engine._build_output(
        ExecutionSpec(intent="general", objective="Say hello."),
        runtime,
        "Hello.",
    )

    assert output.result == "Hello."
    assert "citation_status" not in output.metadata


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


def test_general_agent_does_not_register_safeguard_tool_or_middleware(
    monkeypatch,
) -> None:
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
    runtime = EngineRuntimeContext(sandbox=object())
    runtime.run_context.mark_data_access()
    engine._build_agent(
        SimpleNamespace(
            spec=ExecutionSpec(intent="general", objective="Summarize the file."),
            runtime=runtime,
            query=UserQuery(text="Summarize the file."),
        )
    )

    assert "assess_answerability" not in {tool.name for tool in captured["tools"]}
    assert "RequireAnswerabilityMiddleware" not in {
        type(item).__name__ for item in captured["middleware"]
    }


def test_general_engine_prompt_requires_clarification_without_trusting_documents() -> (
    None
):
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
