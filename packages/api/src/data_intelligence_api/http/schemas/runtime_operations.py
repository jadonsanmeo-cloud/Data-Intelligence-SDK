"""Schemas for stateless Data Intelligence runtime operations."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from data_intelligence_api.http.schemas.runtime_inputs import (
    SelectedFilesRequest,
    ExecutionContextRequest,
    ExecutionFileRequest,
    ReportHistoryMessage,
    RuntimeOptionsRequest,
    UploadedFileRequest,
)


class CitationSourcePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    source: str = Field(min_length=1, max_length=300)
    locator: str = Field(min_length=1, max_length=500)
    excerpt: str = Field(min_length=1, max_length=2000)
    document_id: str | None = Field(default=None, max_length=256)
    content_id: str | None = Field(default=None, max_length=256)


class CitationMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_sources: list[CitationSourcePayload] = Field(
        default_factory=list, max_length=50
    )
    uncited_claims: list[str] = Field(default_factory=list, max_length=50)
    citation_status: Literal["complete", "partial", "unavailable"]

    @field_validator("uncited_claims")
    @classmethod
    def validate_uncited_claims(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 500 for value in values):
            raise ValueError("uncited claims must contain 1 to 500 characters")
        return values

    @model_validator(mode="after")
    def validate_source_coverage(self) -> "CitationMetadataPayload":
        source_ids = [source.id for source in self.citation_sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("citation source IDs must be unique")
        if self.citation_status == "complete" and (
            not source_ids or self.uncited_claims
        ):
            raise ValueError("complete citation metadata must cover all claims")
        if self.citation_status == "partial" and (
            not source_ids or not self.uncited_claims
        ):
            raise ValueError("partial citation metadata needs cited and uncited claims")
        if self.citation_status == "unavailable" and source_ids:
            raise ValueError("unavailable citation metadata cannot include sources")
        return self


_CITATION_METADATA_KEYS = frozenset(
    {"citation_sources", "uncited_claims", "citation_status"}
)
_CITATION_MARKER = re.compile(r"\[[^\]]+\]\(axiom-citation://([^\s)]+)\)")


def normalize_citation_metadata(
    metadata: dict[str, Any], answer: str
) -> dict[str, Any]:
    normalized = dict(metadata)
    if not _CITATION_METADATA_KEYS.intersection(normalized):
        return normalized

    try:
        citations = CitationMetadataPayload.model_validate(
            {
                key: normalized[key]
                for key in _CITATION_METADATA_KEYS
                if key in normalized
            }
        )
        marker_ids = _CITATION_MARKER.findall(answer)
        source_ids = [source.id for source in citations.citation_sources]
        if answer.count("axiom-citation://") != len(marker_ids):
            raise ValueError("answer contains a malformed citation marker")
        if set(marker_ids) != set(source_ids):
            raise ValueError("citation markers and sources do not match")
    except (KeyError, TypeError, ValueError, ValidationError):
        normalized.update(
            citation_sources=[],
            uncited_claims=[],
            citation_status="unavailable",
        )
        return normalized

    normalized.update(citations.model_dump(mode="json"))
    return normalized


class OperationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    operation_id: str = Field(min_length=1, max_length=128)
    attempt: int = Field(default=1, ge=1)
    response_id: str = Field(min_length=1, max_length=128)
    trace_id: str | None = Field(default=None, max_length=128)


class RuntimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    session_id: str = Field(min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    language: str = Field(default="auto", min_length=1, max_length=32)
    history: list[ReportHistoryMessage] = Field(default_factory=list, max_length=200)
    organization_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    uploaded_files: list[UploadedFileRequest] = Field(default_factory=list)
    runtime_options: RuntimeOptionsRequest = Field(
        default_factory=RuntimeOptionsRequest
    )
    execution_context: ExecutionContextRequest | None = None
    execution_files: list[ExecutionFileRequest] = Field(default_factory=list)
    primary_source_id: str | None = Field(default=None, max_length=2048)
    primary_source_ids: list[str] = Field(default_factory=list, max_length=100)
    all_inputs_primary: bool = False
    discover_workspace_files: bool | None = None
    workspace_discovery_instruction: str | None = Field(
        default=None,
        max_length=20_000,
    )
    selected_files: SelectedFilesRequest | None = None
    internal_memory_context: dict[str, Any] | None = None


class PrepareSpecRequest(OperationEnvelope):
    runtime_input: RuntimeInput
    memory_scope: dict[str, Any] | None = None
    memory_context: dict[str, Any] | None = None


class PrepareSpecResponse(OperationEnvelope):
    prepared_execution: dict[str, Any]
    spec_markdown: str = Field(min_length=1)
    intent: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviseSpecRequest(OperationEnvelope):
    runtime_input: RuntimeInput
    prepared_execution: dict[str, Any]
    current_spec_markdown: str = Field(min_length=1)
    revised_spec_markdown: str = Field(min_length=1)
    memory_scope: dict[str, Any] | None = None


class ReviseSpecResponse(OperationEnvelope):
    spec_markdown: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThinkingExecutionRequest(OperationEnvelope):
    runtime_input: RuntimeInput
    prepared_execution: dict[str, Any]
    spec_markdown: str = Field(min_length=1)
    memory_scope: dict[str, Any] | None = None
    memory_context: dict[str, Any] | None = None


class InstantExecutionRequest(OperationEnvelope):
    """Execute immediately without building or confirming a spec."""

    runtime_input: RuntimeInput
    memory_scope: dict[str, Any] | None = None
    memory_context: dict[str, Any] | None = None


class AskUserContinuationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    engine_name: Literal["general"]
    messages: list[dict[str, Any]] = Field(min_length=1)
    pending_tool_call_id: str = Field(min_length=1, max_length=256)
    pending_tool_name: Literal["ask_user", "assess_answerability"] = "ask_user"
    safeguard_required: bool = False


class ResumeExecutionRequest(OperationEnvelope):
    execution_mode: Literal["instant", "thinking"]
    runtime_input: RuntimeInput
    prepared_execution: dict[str, Any] | None = None
    spec_markdown: str | None = Field(default=None, min_length=1)
    continuation_state: AskUserContinuationRequest
    selected_option_id: str | None = Field(default=None, min_length=1, max_length=128)
    other_text: str | None = Field(default=None, min_length=1, max_length=2000)
    memory_scope: dict[str, Any] | None = None
    memory_context: dict[str, Any] | None = None

    @field_validator("selected_option_id", "other_text")
    @classmethod
    def strip_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("user input answer must not be blank")
        return value

    @model_validator(mode="after")
    def validate_resume_context(self) -> ResumeExecutionRequest:
        has_option = self.selected_option_id is not None
        has_other = self.other_text is not None
        if has_option == has_other:
            raise ValueError("provide exactly one selected option or other text")
        if self.execution_mode == "thinking":
            if self.prepared_execution is None or self.spec_markdown is None:
                raise ValueError(
                    "thinking resume requires prepared_execution and spec_markdown"
                )
        elif self.prepared_execution is not None or self.spec_markdown is not None:
            raise ValueError("instant resume cannot include thinking execution data")
        return self


class RuntimeErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool
    operation_id: str
    trace_id: str | None = None
