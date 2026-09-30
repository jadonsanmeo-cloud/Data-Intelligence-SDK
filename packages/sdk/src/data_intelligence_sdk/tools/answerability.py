"""Structured answerability assessments for data-backed agent responses."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data_intelligence_sdk.core.types import (
    SafeguardAssessment,
    SafeguardCategory,
    SafeguardDecision,
    SafeguardFinding,
    SafeguardSeverity,
    UserInputOption,
)

SafeguardInputReason = Literal[
    "data_quality_issue", "source_conflict", "insufficient_evidence"
]


class SafeguardFindingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    category: SafeguardCategory
    severity: SafeguardSeverity
    title: str = Field(min_length=1, max_length=300)
    detail: str = Field(min_length=1, max_length=2000)
    impact: str | None = Field(default=None, max_length=1000)
    affected_scope: str | None = Field(default=None, max_length=500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    blocking: bool = False

    @field_validator("id", "title", "detail", "impact", "affected_scope")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("evidence_refs")
    @classmethod
    def normalize_evidence_refs(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("evidence references must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("evidence references must be unique")
        return normalized

    def to_core(self) -> SafeguardFinding:
        return SafeguardFinding(
            id=self.id,
            category=self.category,
            severity=self.severity,
            title=self.title,
            detail=self.detail,
            impact=self.impact,
            affected_scope=self.affected_scope,
            evidence_refs=list(self.evidence_refs),
            blocking=self.blocking,
        )


class AnswerabilityOptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=1000)
    source: str | None = Field(default=None, max_length=500)

    @field_validator("id", "label", "description", "source")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    def to_core(self) -> UserInputOption:
        return UserInputOption(
            id=self.id,
            label=self.label,
            description=self.description,
            source=self.source,
        )


class AnswerabilityAssessmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: SafeguardDecision
    findings: list[SafeguardFindingInput] = Field(default_factory=list, max_length=20)
    question: str | None = Field(default=None, max_length=2000)
    reason: SafeguardInputReason | None = None
    options: list[AnswerabilityOptionInput] = Field(default_factory=list, max_length=3)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_decision(self) -> AnswerabilityAssessmentInput:
        finding_ids = [finding.id for finding in self.findings]
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("finding IDs must be unique")
        option_ids = [option.id for option in self.options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("option IDs must be unique")

        blocking_findings = [finding for finding in self.findings if finding.blocking]
        if any(finding.category != "connection_risk" for finding in blocking_findings):
            raise ValueError("only connection risk findings may block execution")

        if self.decision == "clear":
            if blocking_findings:
                raise ValueError("clear assessment cannot contain blocking findings")
            if self.question or self.reason or self.options:
                raise ValueError("clear assessment cannot request user input")
        elif self.decision == "needs_user_input":
            if blocking_findings:
                raise ValueError("blocking connection findings cannot be overridden")
            if not self.findings or not self.question or self.reason is None:
                raise ValueError(
                    "user input requires findings, a question, and a reason"
                )
            if not self.options:
                raise ValueError("user input requires one to three safe options")
            matching_category = {
                "data_quality_issue": "data_quality",
                "source_conflict": "source_conflict",
                "insufficient_evidence": "insufficient_evidence",
            }[self.reason]
            if not any(
                finding.category == matching_category for finding in self.findings
            ):
                raise ValueError("user-input reason must match a finding category")
        elif self.decision == "blocked":
            if not any(
                finding.category == "connection_risk" and finding.blocking
                for finding in self.findings
            ):
                raise ValueError(
                    "blocked assessment requires a blocking connection finding"
                )
            if self.question or self.reason or self.options:
                raise ValueError("blocked assessment cannot offer a bypass choice")
        elif self.decision == "abstained":
            if not any(
                finding.category == "insufficient_evidence" for finding in self.findings
            ):
                raise ValueError("abstention requires an insufficient-evidence finding")
            if self.question or self.reason or self.options:
                raise ValueError("abstention cannot include pending user input")
        return self

    def to_core(self) -> SafeguardAssessment:
        return SafeguardAssessment(
            decision=self.decision,
            findings=[finding.to_core() for finding in self.findings],
        )

    def user_input_options(self) -> list[UserInputOption]:
        return [option.to_core() for option in self.options]


def parse_answerability_result(content: Any) -> AnswerabilityAssessmentInput | None:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
        if (
            not isinstance(payload, dict)
            or payload.get("type") != "axiom.answerability.v1"
        ):
            return None
        return AnswerabilityAssessmentInput.model_validate(
            {key: value for key, value in payload.items() if key != "type"}
        )
    except (json.JSONDecodeError, ValueError):
        return None
