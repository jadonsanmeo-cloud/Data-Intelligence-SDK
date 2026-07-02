import unittest

from data_intelligence_sdk.core.types import (
    EngineOutput,
    EngineStep,
    EvidenceBundle,
    ExecutionSpec,
    InterfaceDefinition,
    MethodCall,
    DataCorpusPackage,
    UserQuery,
)
from data_intelligence_sdk.defaults.confirmation import AutoSpecConfirmation
from data_intelligence_sdk.defaults.evidence import TraceEvidenceCollector
from data_intelligence_sdk.defaults.intent import BasicIntentAnalyzer
from data_intelligence_sdk.defaults.spec import BasicSpecBuilder
from data_intelligence_sdk.defaults.synthesis import BasicSynthesizer
from data_intelligence_sdk.sandbox.executor import SandboxRunResult


class DefaultComponentTests(unittest.TestCase):
    def test_basic_intent_analyzer_classifies_common_queries(self) -> None:
        corpus = DataCorpusPackage(sources=["sales.csv"])

        self.assertEqual(
            BasicIntentAnalyzer().analyze(UserQuery("count rows in sales.csv"), corpus),
            "custom",
        )
        self.assertEqual(
            BasicIntentAnalyzer().analyze(UserQuery("run sql query"), corpus), "sql"
        )
        self.assertEqual(
            BasicIntentAnalyzer().analyze(UserQuery("search documents"), corpus), "rag"
        )

    def test_basic_spec_builder_creates_capability_requirements(self) -> None:
        spec = BasicSpecBuilder().build(
            UserQuery("count rows"), "custom", DataCorpusPackage(sources=["sales.csv"])
        )

        self.assertEqual(spec.objective, "count rows")
        self.assertEqual(spec.data_requirements, ["sales.csv"])
        names = [capability.name for capability in spec.capability_requirements]
        self.assertIn("answer_question", names)
        self.assertIn("answer_csv_question", names)

    def test_auto_spec_confirmation_confirms_spec(self) -> None:
        spec = ExecutionSpec(intent="custom", objective="x")

        self.assertTrue(AutoSpecConfirmation().confirm(spec).confirmed)

    def test_trace_evidence_collector_copies_trace_and_metadata(self) -> None:
        interface = InterfaceDefinition(name="generated_reader")
        sandbox_result = SandboxRunResult(status="completed", result={"ok": True})
        output = EngineOutput(
            engine_name="x",
            trace=type(
                "Trace",
                (),
                {
                    "steps": [EngineStep("step")],
                    "method_calls": [MethodCall("method")],
                    "artifact_refs": ["artifact://1"],
                    "log_refs": ["log://1"],
                },
            )(),
            metadata={
                "sources": ["sales.csv"],
                "interface_defs": [interface],
                "sandbox_results": [sandbox_result],
                "observations": [{"summary": "ok"}],
            },
        )

        evidence = TraceEvidenceCollector().collect(
            ExecutionSpec(intent="custom", objective="x"), output
        )

        self.assertEqual(evidence.sources, ["sales.csv"])
        self.assertEqual(evidence.steps[0].name, "step")
        self.assertEqual(evidence.method_calls[0].method_name, "method")
        self.assertEqual(evidence.interface_defs, [interface])
        self.assertEqual(evidence.sandbox_results[0]["status"], "completed")
        self.assertEqual(evidence.observations, [{"summary": "ok"}])
        self.assertEqual(evidence.artifact_refs, ["artifact://1"])
        self.assertEqual(evidence.log_refs, ["log://1"])

    def test_basic_synthesizer_uses_engine_output_result(self) -> None:
        evidence = EvidenceBundle()
        response = BasicSynthesizer().synthesize(
            ExecutionSpec(intent="custom", objective="x"),
            EngineOutput(engine_name="x", result="hello"),
            evidence,
        )

        self.assertEqual(response.answer, "hello")
        self.assertIs(response.evidence, evidence)
        self.assertEqual(response.metadata["engine_name"], "x")


if __name__ == "__main__":
    unittest.main()
