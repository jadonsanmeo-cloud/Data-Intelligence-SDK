import os
import tempfile
import unittest
from pathlib import Path

from data_intelligence_sdk.core.types import (
    CapabilityRequirement,
    DataCorpusPackage,
    ExecutionSpec,
    InterfaceDefinition,
)
from data_intelligence_sdk.engines.general import GeneralPurposeEngine
from data_intelligence_sdk.methods.csv import register_csv_methods
from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext
from data_intelligence_sdk.runtime.interfaces import InMemoryInterfaceRegistry
from data_intelligence_sdk.runtime.method_hub import MethodHub
from data_intelligence_sdk.sandbox.executor import SandboxRunResult


class FakeBuilder:
    def propose(self, requirement, corpus_package):
        return InterfaceDefinition(
            name="generated_method", metadata={"capability_names": [requirement.name]}
        )


class FakeSandbox:
    def validate(self, interface, validation_inputs, resource_policy=None):
        return SandboxRunResult(status="completed", result={"validated": True})

    def run(self, interface, inputs, resource_policy=None):
        return SandboxRunResult(status="completed")


class FakeToolCallingAgent:
    def __init__(self, tool_calls, final_answer="agent answer"):
        self.tool_calls = tool_calls
        self.final_answer = final_answer
        self.prompts = []
        self.tools = []
        self.tool_results = []

    def run(self, *, prompt, tools):
        self.prompts.append(prompt)
        self.tools.append(tools)
        return {"tool_calls": self.tool_calls, "final_answer": self.final_answer}


class FakeSummarizingAgent(FakeToolCallingAgent):
    def run(self, *, prompt, tools):
        self.prompts.append(prompt)
        self.tools.append(tools)
        result = tools["scan_csv"]({"path": self.tool_calls[0]["args"]["path"]})
        self.tool_results.append(result)
        return {
            "tool_calls": [],
            "final_answer": f"columns: {', '.join(result['columns'])}",
        }


class FakeInvokeResponse:
    def __init__(self, content):
        self.content = content


class FakeInvokeModel:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeInvokeResponse(self.content)


class GeneralPurposeEngineTests(unittest.TestCase):
    def test_from_openrouter_requires_api_key(self) -> None:
        old_key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                GeneralPurposeEngine.from_openrouter(model="some/model")
        finally:
            if old_key is not None:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_from_openrouter_requires_model(self) -> None:
        old_model = os.environ.pop("OPENROUTER_MODEL", None)
        try:
            with self.assertRaisesRegex(ValueError, "OPENROUTER_MODEL"):
                GeneralPurposeEngine.from_openrouter(api_key="key")
        finally:
            if old_model is not None:
                os.environ["OPENROUTER_MODEL"] = old_model

    def test_name_and_can_handle_general_specs(self) -> None:
        engine = GeneralPurposeEngine(llm=object())

        self.assertEqual(engine.name, "general_purpose")
        self.assertTrue(
            engine.can_handle(ExecutionSpec(intent="custom", objective="x"))
        )
        self.assertTrue(
            engine.can_handle(ExecutionSpec(intent="unknown", objective="x"))
        )

    def test_engine_lets_agent_call_method_hub_tools_and_records_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sales.csv"
            csv_path.write_text(
                "country,status,revenue\nUS,complete,10\nCA,complete,7\n",
                encoding="utf-8",
            )
            method_hub = MethodHub()
            register_csv_methods(method_hub)
            runtime = EngineRuntimeContext(method_hub=method_hub)
            agent = FakeToolCallingAgent(
                tool_calls=[{"name": "count_csv", "args": {"path": str(csv_path)}}],
                final_answer="There are 2 rows.",
            )

            output = GeneralPurposeEngine(llm=agent).run(
                ExecutionSpec(intent="custom", objective="How many rows are there?"),
                DataCorpusPackage(sources=[str(csv_path)]),
                runtime,
            )

            self.assertEqual(output.result, "There are 2 rows.")
            self.assertIn("How many rows are there?", agent.prompts[0])
            self.assertIn(str(csv_path), agent.prompts[0])
            self.assertIn("count_csv", agent.tools[0])
            self.assertEqual(output.trace.method_calls[0].method_name, "count_csv")
            self.assertEqual(output.trace.method_calls[0].status, "completed")
            self.assertEqual(output.trace.method_calls[0].outputs["result"]["count"], 2)

    def test_engine_exposes_scan_csv_tool_to_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sales.csv"
            csv_path.write_text(
                "country,status,revenue\nUS,complete,10\n",
                encoding="utf-8",
            )
            method_hub = MethodHub()
            register_csv_methods(method_hub)
            runtime = EngineRuntimeContext(method_hub=method_hub)
            agent = FakeToolCallingAgent(
                tool_calls=[{"name": "scan_csv", "args": {"path": str(csv_path)}}],
                final_answer="Columns are country, status, revenue.",
            )

            output = GeneralPurposeEngine(llm=agent).run(
                ExecutionSpec(
                    intent="custom", objective="What columns are in this file?"
                ),
                DataCorpusPackage(sources=[str(csv_path)]),
                runtime,
            )

            self.assertEqual(output.result, "Columns are country, status, revenue.")
            self.assertEqual(output.trace.method_calls[0].method_name, "scan_csv")

    def test_invoke_model_json_tool_request_executes_method_hub_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sales.csv"
            csv_path.write_text(
                "country,status,revenue\nUS,complete,10\n",
                encoding="utf-8",
            )
            method_hub = MethodHub()
            register_csv_methods(method_hub)
            runtime = EngineRuntimeContext(method_hub=method_hub)
            model = FakeInvokeModel(
                '{"tool": "scan_csv", "arguments": {"file_path": "'
                + str(csv_path)
                + '"}}'
            )

            output = GeneralPurposeEngine(llm=model).run(
                ExecutionSpec(
                    intent="custom", objective="What columns are in this file?"
                ),
                DataCorpusPackage(sources=[str(csv_path)]),
                runtime,
            )

            self.assertEqual(
                output.result, "CSV columns: country, status, revenue; rows: 1"
            )
            self.assertEqual(output.trace.method_calls[0].method_name, "scan_csv")
            self.assertEqual(
                output.trace.method_calls[0].inputs, {"path": str(csv_path)}
            )

    def test_invoke_model_json_tool_request_accepts_parameters_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sales.csv"
            csv_path.write_text(
                "country,status,revenue\nUS,complete,10\n",
                encoding="utf-8",
            )
            method_hub = MethodHub()
            register_csv_methods(method_hub)
            runtime = EngineRuntimeContext(method_hub=method_hub)
            model = FakeInvokeModel(
                '{"tool": "scan_csv", "parameters": {"file_path": "'
                + str(csv_path)
                + '"}}'
            )

            output = GeneralPurposeEngine(llm=model).run(
                ExecutionSpec(
                    intent="custom", objective="What columns are in this file?"
                ),
                DataCorpusPackage(sources=[str(csv_path)]),
                runtime,
            )

            self.assertEqual(
                output.result, "CSV columns: country, status, revenue; rows: 1"
            )
            self.assertEqual(
                output.trace.method_calls[0].inputs, {"path": str(csv_path)}
            )

    def test_missing_generation_services_returns_clear_answer(self) -> None:
        output = GeneralPurposeEngine(llm=object()).run(
            ExecutionSpec(
                intent="custom",
                objective="needs unavailable capability",
                capability_requirements=[CapabilityRequirement("missing")],
            ),
            DataCorpusPackage(sources=["sales.csv"]),
            EngineRuntimeContext(),
        )

        self.assertIn("method generation is not configured", str(output.result))
        self.assertEqual(output.trace.steps[-1].name, "method_generation_unavailable")

    def test_generated_method_lifecycle_validates_and_registers_interface(self) -> None:
        registry = InMemoryInterfaceRegistry()
        runtime = EngineRuntimeContext(
            interface_builder=FakeBuilder(),
            sandbox_executor=FakeSandbox(),
            interface_registry=registry,
        )

        output = GeneralPurposeEngine(llm=object()).run(
            ExecutionSpec(
                intent="custom",
                objective="needs generated capability",
                capability_requirements=[CapabilityRequirement("generated_capability")],
            ),
            DataCorpusPackage(sources=["sales.csv"]),
            runtime,
        )

        interface = output.metadata["interface_defs"][0]
        self.assertEqual(interface.trust_level, "generated_validated")
        self.assertEqual(registry.list_available(), [interface])
        self.assertEqual(output.metadata["sandbox_results"][0].status, "completed")


if __name__ == "__main__":
    unittest.main()
