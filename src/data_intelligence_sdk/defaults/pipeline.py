"""Factories for the default query-to-answer workflow."""

from __future__ import annotations

from data_intelligence_sdk.core.pipeline import DataIntelligencePipeline
from data_intelligence_sdk.defaults.confirmation import AutoSpecConfirmation
from data_intelligence_sdk.defaults.evidence import TraceEvidenceCollector
from data_intelligence_sdk.defaults.intent import BasicIntentAnalyzer
from data_intelligence_sdk.defaults.spec import BasicSpecBuilder
from data_intelligence_sdk.defaults.synthesis import BasicSynthesizer
from data_intelligence_sdk.engines.general import GeneralPurposeEngine
from data_intelligence_sdk.methods import register_csv_methods
from data_intelligence_sdk.registry.engine_registry import InMemoryEngineRegistry
from data_intelligence_sdk.runtime.interfaces import InMemoryInterfaceRegistry
from data_intelligence_sdk.runtime.method_hub import MethodHub


def create_default_pipeline(
    *,
    engine: object | None = None,
    llm: object | None = None,
    method_hub: MethodHub | None = None,
    interface_registry: object | None = None,
    interface_builder: object | None = None,
    sandbox_executor: object | None = None,
) -> DataIntelligencePipeline:
    """Create a default pipeline using explicit engine or injected LLM."""

    method_hub = method_hub or MethodHub()
    if not method_hub.list_methods():
        register_csv_methods(method_hub)
    if engine is None:
        if llm is None:
            raise ValueError("engine or llm is required")
        engine = GeneralPurposeEngine(llm=llm)
    interface_registry = interface_registry or InMemoryInterfaceRegistry()
    registry = InMemoryEngineRegistry(fallback_engine=engine)
    return DataIntelligencePipeline(
        intent_analyzer=BasicIntentAnalyzer(),
        spec_builder=BasicSpecBuilder(),
        spec_confirmation=AutoSpecConfirmation(),
        engine_registry=registry,
        evidence_collector=TraceEvidenceCollector(),
        synthesizer=BasicSynthesizer(),
        method_hub=method_hub,
        interface_registry=interface_registry,
        interface_builder=interface_builder,
        sandbox_executor=sandbox_executor,
    )


def create_default_pipeline_from_openrouter(
    *,
    model: str | None = None,
    api_key: str | None = None,
    allow_method_generation: bool = True,
) -> DataIntelligencePipeline:
    """Create a default pipeline with OpenRouter-backed GeneralPurposeEngine."""

    engine = GeneralPurposeEngine.from_openrouter(
        model=model,
        api_key=api_key,
        allow_method_generation=allow_method_generation,
    )
    return create_default_pipeline(engine=engine)
