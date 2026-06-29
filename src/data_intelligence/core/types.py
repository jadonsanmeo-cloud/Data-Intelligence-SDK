"""Shared data contracts for the base architecture.

These types are intentionally small. They describe the information that moves
between modules without locking the project into detailed SDK behavior too
early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UserQuery:
    """Raw request from a user."""

    text: str
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataHubContext:
    """Data, metadata, and semantic context available to the system."""

    sources: list[str] = field(default_factory=list)
    schemas: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UserContext:
    """User-level memory and preferences across tasks."""

    user_id: str | None = None
    preferences: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Intent:
    """Interpreted user intent before execution planning."""

    task_type: str
    summary: str
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionSpec:
    """Draft or confirmed execution request for engine selection."""

    intent: Intent
    objective: str
    data_requirements: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    confirmed: bool = False
    engine_hint: str | None = None


@dataclass(slots=True)
class EngineOutput:
    """Raw result returned by an engine."""

    engine_name: str
    result: Any = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    method_calls: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceBundle:
    """Evidence and trace material collected around an engine run."""

    sources: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    method_calls: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FinalResponse:
    """User-facing answer plus supporting trace references."""

    answer: str
    evidence: EvidenceBundle | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
