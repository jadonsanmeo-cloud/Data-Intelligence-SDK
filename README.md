## Setup

```
uv venv --python 3.11
```

# Data Intelligence

Base Python package for a data intelligence orchestration system.

The current goal is to capture the architecture boundaries from
`ai-sdk-platform-architecture.svg`, not to provide production-ready SDK
implementations.

## Architecture Flow

```text
User Query + DataCorpusPackage
  -> Intent Analyzer
  -> Spec Builder
  -> Spec Confirmation
  -> Engine Registry
  -> Selected Engine
  -> EngineRuntimeContext
       -> MethodHub
       -> InterfaceRegistry
       -> InterfaceBuilder
       -> SandboxExecutor
       -> Artifacts / Logs / Resources
  -> Engine Execution
       -> reuse trusted interface/method
       -> or create generated interface
       -> validate/run generated interface in sandbox
       -> emit EngineOutput + EngineTrace
  -> Evidence Collector
  -> Synthesizer
  -> FinalResponse(answer + evidence)
```

Supporting layers:

- `runtime`: method hub, interface registry, interface builder boundary, engine runtime context, executor, logger, resource manager, cache.
- `sandbox`: controlled execution, data/workspace/artifacts/logs boundaries.
- `context`: user and session context placeholders.

## Base Design Notes

- `DataCorpusPackage` describes the available data universe for a task. It may contain source refs, schemas, semantic metadata, and policy metadata; it does not necessarily contain raw data.
- `DataHubContext` remains available as a compatibility alias for `DataCorpusPackage` during the transition.
- `Intent` is a controlled string selected from `SUPPORTED_INTENTS`, not a rich object. The spec carries the richer objective, constraints, and capability requirements.
- `ExecutionSpec.capability_requirements` describes what the selected engine/runtime must resolve.
- Engines receive an `EngineRuntimeContext`, which owns an `EngineRunContext` for trace recording and exposes runtime services such as `MethodHub`, `InterfaceRegistry`, `InterfaceBuilder`, and `SandboxExecutor`.
- Engines should request capabilities from runtime services. They should not each reimplement interface discovery, generated interface lifecycle, sandbox policy, trust promotion, artifact/log policy, or evidence construction.
- Generated interfaces start as `generated_unvalidated` and should be validated through sandbox execution before reuse.
- `EngineOutput` contains raw engine output plus `EngineTrace`.
- `EvidenceBundle` uses engine trace, method calls, interface definitions, sandbox results, observations, artifact refs, and log refs for audit and final response generation.
- `SessionContext` is separate from `UserContext`: session context is short-lived conversation/task state, while user context is longer-lived preference and history.

## Base Query-to-Answer Workflow

The SDK now includes a first runnable base workflow for CSV-backed data questions. `GeneralPurposeEngine` is the fallback/general engine, OpenRouter is the first supported LLM provider through LangChain, and concrete CSV capabilities are exposed through MethodHub methods.

```python
from data_intelligence_sdk import DataCorpusPackage, UserQuery
from data_intelligence_sdk.defaults import create_default_pipeline_from_openrouter

pipeline = create_default_pipeline_from_openrouter()
response = pipeline.run(
    UserQuery("What is the total revenue in this file?"),
    DataCorpusPackage(sources=["sales.csv"]),
)
print(response.answer)
```

Configure OpenRouter with:

```text
OPENROUTER_API_KEY
OPENROUTER_MODEL
```

CSV MethodHub methods live in `data_intelligence_sdk.methods` and include `scan_csv`, `filter_csv`, `count_csv`, and `sum_csv`. Default pipeline factories live in `data_intelligence_sdk.defaults` as `create_default_pipeline()` and `create_default_pipeline_from_openrouter()`.

Tests use fake engines or fake LLMs and do not call OpenRouter.
