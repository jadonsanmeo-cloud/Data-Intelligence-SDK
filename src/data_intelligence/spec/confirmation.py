"""Human-in-the-loop spec confirmation boundary."""

from __future__ import annotations

from typing import Protocol

from data_intelligence.core.types import ExecutionSpec, UserContext


class SpecConfirmation(Protocol):
    """Confirms or revises a draft spec before engine selection."""

    def confirm(
        self,
        spec: ExecutionSpec,
        user_context: UserContext | None = None,
    ) -> ExecutionSpec:
        """Return a confirmed spec."""
