"""Validate AI-assisted citation assignments against run-scoped evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_intelligence_sdk.core.types import CitationEvidence


class CitationClaimInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1200)
    evidence_ids: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("claim text must not be blank")
        return normalized

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("evidence IDs must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("evidence IDs must be unique")
        return normalized


class CitationAssignmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[CitationClaimInput] = Field(default_factory=list, max_length=50)


def parse_citation_assignments(content: Any) -> CitationAssignmentInput | None:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
        return CitationAssignmentInput.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def link_citations(
    answer: str,
    assignments: CitationAssignmentInput,
    evidence: list[CitationEvidence] | tuple[CitationEvidence, ...],
) -> tuple[str, dict[str, Any]]:
    evidence_by_id = {item.id: item for item in evidence}
    matches: list[tuple[int, int, CitationClaimInput, list[str]]] = []
    unverified: list[tuple[int, int, str]] = []

    for claim in assignments.claims:
        first = answer.find(claim.text)
        second = answer.find(claim.text, first + 1) if first >= 0 else -1
        if first < 0:
            continue
        if second >= 0:
            position = first
            while position >= 0:
                unverified.append((position, position + len(claim.text), claim.text))
                position = answer.find(claim.text, position + len(claim.text))
            continue
        last = first + len(claim.text)
        valid_ids = [
            evidence_id
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_by_id
        ]
        if not claim.evidence_ids or len(valid_ids) != len(claim.evidence_ids):
            unverified.append((first, last, claim.text))
            continue
        matches.append((first, last, claim, valid_ids))

    overlapping: set[tuple[int, int]] = set()
    ordered_matches = sorted(matches, key=lambda match: (match[0], match[1]))
    for index, current in enumerate(ordered_matches):
        for following in ordered_matches[index + 1 :]:
            if following[0] < current[1]:
                overlapping.add((current[0], current[1]))
                overlapping.add((following[0], following[1]))

    resolved_matches = [
        match for match in ordered_matches if (match[0], match[1]) not in overlapping
    ]
    for start, end, claim, _ in ordered_matches:
        if (start, end) in overlapping:
            unverified.append((start, end, claim.text))

    source_order: list[str] = []
    for _, _, _, evidence_ids in resolved_matches:
        for evidence_id in evidence_ids:
            if evidence_id not in source_order:
                source_order.append(evidence_id)
    source_indexes = {
        evidence_id: index for index, evidence_id in enumerate(source_order, start=1)
    }

    insertions: list[tuple[int, str]] = []
    for _, end, _, evidence_ids in resolved_matches:
        markers = " ".join(
            f"[{source_indexes[evidence_id]}](axiom-citation://{evidence_id})"
            for evidence_id in evidence_ids
        )
        insertions.append((end, f" {markers}"))
    for _, end, _ in unverified:
        insertions.append((end, " [Unverified]"))

    linked_answer = answer
    for position, marker in sorted(insertions, key=lambda item: item[0], reverse=True):
        linked_answer = linked_answer[:position] + marker + linked_answer[position:]

    cited_sources = [
        asdict(evidence_by_id[evidence_id]) for evidence_id in source_order
    ]
    unverified_claims = [claim[:500] for _, _, claim in sorted(unverified)]
    has_verified_claims = bool(resolved_matches)
    has_unverified_claims = bool(unverified)
    if not assignments.claims or not evidence_by_id:
        status = "unavailable"
    elif has_verified_claims and has_unverified_claims:
        status = "partial"
    elif has_verified_claims:
        status = "complete"
    else:
        status = "unavailable"
    return linked_answer, {
        "citation_sources": cited_sources,
        "uncited_claims": unverified_claims,
        "citation_status": status,
    }


def attribute_answer_citations(
    model: Any,
    answer: str,
    evidence: list[CitationEvidence] | tuple[CitationEvidence, ...],
) -> tuple[str, dict[str, Any]]:
    if not evidence:
        return answer, {
            "citation_sources": [],
            "uncited_claims": [],
            "citation_status": "unavailable",
        }
    eligible_evidence = evidence[:50]
    prompt_payload = {
        "answer": answer,
        "evidence": [asdict(item) for item in eligible_evidence],
    }
    result = model.invoke(
        [
            SystemMessage(
                content=(
                    "Link every material factual claim in the answer to evidence "
                    "that directly supports it. Return JSON only in the shape "
                    '{"claims":[{"text":"exact answer substring",'
                    '"evidence_ids":["provided-id"]}]}. Include each claim as an '
                    "exact, unique substring copied from the answer. Use only evidence "
                    "IDs in the supplied evidence. Do not cite topical matches or "
                    "invent IDs; use an empty evidence_ids list when no source directly "
                    "supports a claim. Do not rewrite the answer."
                )
            ),
            HumanMessage(
                content=json.dumps(prompt_payload, ensure_ascii=False, default=str)
            ),
        ]
    )
    content = getattr(result, "content", result)
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    assignments = parse_citation_assignments(content)
    if assignments is None:
        return answer, {
            "citation_sources": [],
            "uncited_claims": [],
            "citation_status": "unavailable",
        }
    return link_citations(answer, assignments, eligible_evidence)
