"""Execution spec drafting contract."""

from __future__ import annotations

from typing import Protocol

from data_intelligence.core.types import DataHubContext, ExecutionSpec, Intent, UserContext


class SpecBuilder(Protocol):
    """Builds a draft execution spec from intent and data context."""

    def build(
        self,
        intent: Intent,
        datahub: DataHubContext,
        user_context: UserContext | None = None,
    ) -> ExecutionSpec:
        """Return a draft execution spec."""
