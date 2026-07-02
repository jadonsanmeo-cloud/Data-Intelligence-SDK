"""Default response synthesis."""

from __future__ import annotations

from data_intelligence_sdk.core.types import (
    EngineOutput,
    EvidenceBundle,
    ExecutionSpec,
    FinalResponse,
)


class BasicSynthesizer:
    """Turn engine output into a user-facing final response."""

    def synthesize(
        self, spec: ExecutionSpec, output: EngineOutput, evidence: EvidenceBundle
    ) -> FinalResponse:
        del spec
        return FinalResponse(
            answer=str(output.result),
            evidence=evidence,
            metadata={"engine_name": output.engine_name},
        )
