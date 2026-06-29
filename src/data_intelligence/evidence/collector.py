"""Evidence collection contract."""

from __future__ import annotations

from typing import Protocol

from data_intelligence.core.types import EngineOutput, EvidenceBundle, ExecutionSpec


class EvidenceCollector(Protocol):
    """Collects evidence, traces, method calls, and artifacts from execution."""

    def collect(self, spec: ExecutionSpec, output: EngineOutput) -> EvidenceBundle:
        """Return evidence for synthesis and audit."""
