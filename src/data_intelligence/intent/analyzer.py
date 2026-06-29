"""Intent analysis contract."""

from __future__ import annotations

from typing import Protocol

from data_intelligence.core.types import DataHubContext, Intent, UserContext, UserQuery


class IntentAnalyzer(Protocol):
    """Infers task intent from the user query and available data context."""

    def analyze(
        self,
        query: UserQuery,
        datahub: DataHubContext,
        user_context: UserContext | None = None,
    ) -> Intent:
        """Return the interpreted user intent."""
