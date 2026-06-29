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
