"""Default execution spec builder."""

from __future__ import annotations

from data_intelligence_sdk.core.types import (
    CapabilityRequirement,
    DataCorpusPackage,
    ExecutionSpec,
    Intent,
    SessionContext,
    UserContext,
    UserQuery,
)


class BasicSpecBuilder:
    """Build an execution spec from the user query and corpus package."""

    def build(
        self,
        query: UserQuery,
        intent: Intent,
        corpus_package: DataCorpusPackage,
        session_context: SessionContext | None = None,
        user_context: UserContext | None = None,
    ) -> ExecutionSpec:
        del session_context, user_context
        capability_names = [
            "inspect_data",
            "filter_data",
            "aggregate_data",
            "answer_question",
        ]
        if any(
            str(source).lower().endswith(".csv") for source in corpus_package.sources
        ):
            capability_names.append("answer_csv_question")
        return ExecutionSpec(
            intent=intent,
            objective=query.text,
            data_requirements=list(corpus_package.sources),
            capability_requirements=[
                CapabilityRequirement(name=name) for name in capability_names
            ],
        )
