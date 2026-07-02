"""Method hub boundary for engine-accessible capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_intelligence_sdk.core.types import CapabilityRequirement, TrustLevel


@dataclass(slots=True)
class RegisteredMethod:
    """Callable method plus capability and trust metadata."""

    name: str
    method: object
    capability_names: list[str] = field(default_factory=list)
    trust_level: TrustLevel = "builtin"
    metadata: dict[str, Any] = field(default_factory=dict)


class MethodHub:
    """Registry for methods that engines may call during execution."""

    def __init__(self) -> None:
        self._methods: dict[str, RegisteredMethod] = {}

    def register(
        self,
        name: str,
        method: object,
        *,
        capability_names: list[str] | None = None,
        trust_level: TrustLevel = "builtin",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._methods[name] = RegisteredMethod(
            name=name,
            method=method,
            capability_names=capability_names or [],
            trust_level=trust_level,
            metadata=metadata or {},
        )

    def get(self, name: str) -> object:
        return self._methods[name].method

    def get_definition(self, name: str) -> RegisteredMethod:
        return self._methods[name]

    def list_methods(self) -> list[RegisteredMethod]:
        return list(self._methods.values())

    def resolve(self, requirement: CapabilityRequirement) -> RegisteredMethod | None:
        for method in self._methods.values():
            if requirement.name in method.capability_names:
                return method
        return None
