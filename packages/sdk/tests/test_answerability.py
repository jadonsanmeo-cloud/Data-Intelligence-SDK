import json

import pytest
from pydantic import ValidationError

from data_intelligence_sdk.tools.answerability import (
    AnswerabilityAssessmentInput,
    parse_answerability_result,
)
from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext


def _finding(
    category: str,
    *,
    finding_id: str = "finding-1",
    blocking: bool = False,
) -> dict[str, object]:
    return {
        "id": finding_id,
        "category": category,
        "severity": "high",
        "title": "Finding title",
        "detail": "Finding details",
        "impact": "Affects the requested result",
        "affected_scope": "requested period",
        "evidence_refs": ["source://record/1"],
        "blocking": blocking,
    }


@pytest.mark.parametrize(
    ("decision", "finding", "reason"),
    [
        ("clear", _finding("data_quality"), None),
        ("clear", _finding("connection_risk"), None),
        ("clear", _finding("source_conflict"), None),
        ("clear", _finding("insufficient_evidence"), None),
        (
            "needs_user_input",
            _finding("data_quality"),
            "data_quality_issue",
        ),
        (
            "needs_user_input",
            _finding("source_conflict"),
            "source_conflict",
        ),
        (
            "needs_user_input",
            _finding("insufficient_evidence"),
            "insufficient_evidence",
        ),
        (
            "blocked",
            _finding("connection_risk", blocking=True),
            None,
        ),
        (
            "abstained",
            _finding("insufficient_evidence"),
            None,
        ),
    ],
)
def test_assessment_accepts_supported_decision_and_categories(
    decision: str,
    finding: dict[str, object],
    reason: str | None,
) -> None:
    payload: dict[str, object] = {
        "decision": decision,
        "findings": [finding],
    }
    if decision == "needs_user_input":
        payload.update(
            {
                "question": "How should I proceed?",
                "reason": reason,
                "options": [
                    {"id": "continue", "label": "Continue with available evidence"},
                    {"id": "add_source", "label": "Add another source"},
                ],
            }
        )

    assessment = AnswerabilityAssessmentInput.model_validate(payload)

    assert assessment.decision == decision
    assert assessment.findings[0].category == finding["category"]


def test_clear_assessment_accepts_no_findings() -> None:
    assessment = AnswerabilityAssessmentInput.model_validate(
        {"decision": "clear", "findings": []}
    )

    assert assessment.findings == []


@pytest.mark.parametrize(
    "payload",
    [
        {
            "decision": "needs_user_input",
            "findings": [_finding("data_quality")],
            "reason": "data_quality_issue",
            "options": [{"id": "continue", "label": "Continue"}],
        },
        {
            "decision": "needs_user_input",
            "findings": [_finding("data_quality")],
            "question": "What should I do?",
            "reason": "data_quality_issue",
            "options": [
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "c", "label": "C"},
                {"id": "d", "label": "D"},
            ],
        },
        {
            "decision": "blocked",
            "findings": [_finding("connection_risk")],
        },
        {
            "decision": "blocked",
            "findings": [_finding("data_quality", blocking=True)],
        },
        {
            "decision": "clear",
            "findings": [_finding("connection_risk", blocking=True)],
        },
        {
            "decision": "needs_user_input",
            "findings": [_finding("connection_risk", blocking=True)],
            "question": "What should I do?",
            "reason": "data_quality_issue",
            "options": [{"id": "continue", "label": "Continue"}],
        },
        {
            "decision": "abstained",
            "findings": [_finding("source_conflict")],
        },
        {
            "decision": "clear",
            "findings": [
                _finding("data_quality", finding_id="duplicate"),
                _finding("source_conflict", finding_id="duplicate"),
            ],
        },
    ],
)
def test_assessment_rejects_invalid_decision_combinations(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AnswerabilityAssessmentInput.model_validate(payload)


def test_parse_answerability_result_only_accepts_versioned_marker() -> None:
    assessment = AnswerabilityAssessmentInput.model_validate(
        {"decision": "clear", "findings": []}
    )
    marker = json.dumps(
        {
            "type": "axiom.answerability.v1",
            **assessment.model_dump(mode="json"),
        }
    )

    assert parse_answerability_result(marker) == assessment
    assert parse_answerability_result('{"type":"unknown"}') is None
    assert parse_answerability_result(None) is None


def test_new_data_access_invalidates_a_prior_assessment() -> None:
    runtime = EngineRuntimeContext()
    assessment = AnswerabilityAssessmentInput.model_validate(
        {"decision": "clear", "findings": []}
    )
    runtime.run_context.set_safeguard_assessment(assessment.to_core())

    runtime.run_context.mark_data_access()

    assert runtime.run_context.data_accessed is True
    assert runtime.run_context.safeguard_assessment is None
