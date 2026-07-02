"""Default intent analyzer implementation."""

from __future__ import annotations

from data_intelligence_sdk.core.types import (
    DataCorpusPackage,
    Intent,
    SessionContext,
    UserContext,
    UserQuery,
)


class BasicIntentAnalyzer:
    """Classify common queries with simple keyword rules."""

    def analyze(
        self,
        query: UserQuery,
        corpus_package: DataCorpusPackage,
        session_context: SessionContext | None = None,
        user_context: UserContext | None = None,
    ) -> Intent:
        del session_context, user_context
        text = query.text.lower()
        if any(term in text for term in ("sql", "select ", "from ", "where ")):
            return "sql"
        if any(term in text for term in ("python", "dataframe", "pandas", "calculate")):
            return "python"
        if any(
            term in text
            for term in (
                "document",
                "documents",
                "search",
                "retrieve",
                "retrieval",
                "rag",
            )
        ):
            return "rag"
        if any(
            str(source).lower().endswith(".csv") for source in corpus_package.sources
        ) or any(
            term in text
            for term in (
                "csv",
                "file",
                "row",
                "rows",
                "count",
                "sum",
                "total",
                "average",
                "revenue",
            )
        ):
            return "custom"
        return "unknown"
