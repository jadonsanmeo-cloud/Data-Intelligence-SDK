import json
from types import SimpleNamespace

from data_intelligence_sdk.core.types import CitationEvidence
from data_intelligence_sdk.tools.citation_attribution import (
    CitationAssignmentInput,
    attribute_answer_citations,
    link_citations,
    parse_citation_assignments,
)


def _evidence(identifier: str, source: str = "finance.pdf") -> CitationEvidence:
    return CitationEvidence(
        id=identifier,
        source=source,
        locator="page 4 · chunk 12",
        excerpt="Revenue was 12 million.",
        document_id="document-1",
        content_id="chunk-12",
    )


def test_parses_structured_claim_assignments() -> None:
    parsed = parse_citation_assignments(
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

    assert parsed is not None
    assert parsed.claims[0].text == "Revenue was 12 million."
    assert parsed.claims[0].evidence_ids == ["ev-source-1"]


def test_valid_assignments_insert_inline_markers_and_source_payload() -> None:
    answer = "Revenue was 12 million."
    assignments = CitationAssignmentInput(
        claims=[
            {"text": answer, "evidence_ids": ["ev-source-1"]},
        ]
    )

    linked_answer, payload = link_citations(
        answer, assignments, [_evidence("ev-source-1")]
    )

    assert linked_answer == "Revenue was 12 million. [1](axiom-citation://ev-source-1)"
    assert payload["citation_status"] == "complete"
    assert payload["uncited_claims"] == []
    assert payload["citation_sources"][0]["id"] == "ev-source-1"
    assert payload["citation_sources"][0]["excerpt"] == "Revenue was 12 million."


def test_unknown_evidence_ids_make_the_claim_unverified() -> None:
    answer = "Revenue was 12 million."
    assignments = CitationAssignmentInput(
        claims=[{"text": answer, "evidence_ids": ["ev-not-in-run"]}]
    )

    linked_answer, payload = link_citations(
        answer, assignments, [_evidence("ev-source-1")]
    )

    assert linked_answer == "Revenue was 12 million. [Unverified]"
    assert payload["citation_status"] == "unavailable"
    assert payload["citation_sources"] == []
    assert payload["uncited_claims"] == [answer]


def test_duplicate_claim_text_is_not_ambiguously_linked() -> None:
    answer = "Revenue was 12 million. Revenue was 12 million."
    assignments = CitationAssignmentInput(
        claims=[{"text": "Revenue was 12 million.", "evidence_ids": ["ev-source-1"]}]
    )

    linked_answer, payload = link_citations(
        answer, assignments, [_evidence("ev-source-1")]
    )

    assert linked_answer == (
        "Revenue was 12 million. [Unverified] Revenue was 12 million. [Unverified]"
    )
    assert payload["citation_status"] == "unavailable"
    assert payload["citation_sources"] == []


def test_multiple_claims_preserve_text_and_deduplicate_source_list() -> None:
    answer = "Revenue was 12 million. Revenue increased 12 million."
    assignments = CitationAssignmentInput(
        claims=[
            {"text": "Revenue was 12 million.", "evidence_ids": ["ev-source-1"]},
            {"text": "Revenue increased 12 million.", "evidence_ids": ["ev-source-1"]},
        ]
    )

    linked_answer, payload = link_citations(
        answer, assignments, [_evidence("ev-source-1")]
    )

    assert linked_answer == (
        "Revenue was 12 million. [1](axiom-citation://ev-source-1) "
        "Revenue increased 12 million. [1](axiom-citation://ev-source-1)"
    )
    assert payload["citation_status"] == "complete"
    assert [source["id"] for source in payload["citation_sources"]] == ["ev-source-1"]


def test_rejects_evidence_ids_not_in_the_model_visible_subset() -> None:
    answer = "Revenue was 12 million."
    evidence = [_evidence(f"ev-source-{index}") for index in range(51)]

    class FakeModel:
        def invoke(self, messages: list[object]) -> SimpleNamespace:
            del messages
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "claims": [
                            {
                                "text": answer,
                                "evidence_ids": ["ev-source-50"],
                            }
                        ]
                    }
                )
            )

    linked_answer, payload = attribute_answer_citations(FakeModel(), answer, evidence)

    assert linked_answer == f"{answer} [Unverified]"
    assert payload["citation_status"] == "unavailable"
    assert payload["citation_sources"] == []
