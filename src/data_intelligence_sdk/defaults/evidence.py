"""Default evidence collection from engine traces."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from data_intelligence_sdk.core.types import EngineOutput, EvidenceBundle, ExecutionSpec


def _as_result_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {"result": value}


class TraceEvidenceCollector:
    """Build evidence from engine trace and output metadata."""

    def collect(self, spec: ExecutionSpec, output: EngineOutput) -> EvidenceBundle:
        metadata = output.metadata
        return EvidenceBundle(
            sources=list(metadata.get("sources", spec.data_requirements)),
            observations=list(metadata.get("observations", [])),
            steps=list(output.trace.steps),
            method_calls=list(output.trace.method_calls),
            interface_defs=list(metadata.get("interface_defs", [])),
            sandbox_results=[
                _as_result_dict(result)
                for result in metadata.get("sandbox_results", [])
            ],
            artifact_refs=list(output.trace.artifact_refs),
            log_refs=list(output.trace.log_refs),
        )
