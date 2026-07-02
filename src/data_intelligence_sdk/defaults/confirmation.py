"""Default spec confirmation implementation."""

from __future__ import annotations

from data_intelligence_sdk.core.types import ExecutionSpec, SessionContext, UserContext


class AutoSpecConfirmation:
    """Confirm specs without interactive UI."""

    def confirm(
        self,
        spec: ExecutionSpec,
        session_context: SessionContext | None = None,
        user_context: UserContext | None = None,
    ) -> ExecutionSpec:
        del session_context, user_context
        spec.confirmed = True
        return spec
