# Data Intelligence

Base Python package for a data intelligence orchestration system.

The current goal is to capture the architecture boundaries from
`ai-sdk-platform-architecture.svg`, not to provide production-ready SDK
implementations.

## Architecture Flow

```text
User Query + Data Hub Context
  -> Intent Analyzer
  -> Spec Builder
  -> Spec Confirmation
  -> Engine Registry
  -> Selected Engine
  -> Evidence Collector
  -> Synthesizer
  -> Final Response
```

Supporting layers:

- `runtime`: method hub, executor, logger, resource manager, cache.
- `sandbox`: data/workspace/artifacts/logs boundaries.
- `context`: user and session context placeholders.

## Base Design Notes

- `Intent` is a controlled string selected from `SUPPORTED_INTENTS`, not a
  rich object. The spec carries the richer objective and constraints.
- `EngineOutput` is raw engine output plus references to artifacts/logs. It
  does not claim to know the final execution steps or method calls.
- `EvidenceBundle` is where steps, method calls, observations, artifact refs,
  and log refs are synthesized for audit and final response generation.
- `SessionContext` is separate from `UserContext`: session context is short
  lived conversation/task state, while user context is longer-lived preference
  and history.
