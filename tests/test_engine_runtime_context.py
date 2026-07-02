import unittest

from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext
from data_intelligence_sdk.runtime.interfaces import InMemoryInterfaceRegistry
from data_intelligence_sdk.runtime.method_hub import MethodHub
from data_intelligence_sdk.runtime.resource_manager import ResourceManager
from data_intelligence_sdk.runtime.run_context import EngineRunContext
from data_intelligence_sdk.sandbox.artifacts import ArtifactStore
from data_intelligence_sdk.sandbox.executor import SandboxRunResult
from data_intelligence_sdk.sandbox.logs import LogStore


class EngineRuntimeContextTests(unittest.TestCase):
    def test_runtime_context_defaults_trace_and_method_hub(self) -> None:
        runtime = EngineRuntimeContext()

        self.assertIsInstance(runtime.run_context, EngineRunContext)
        self.assertIsInstance(runtime.method_hub, MethodHub)
        self.assertIsNone(runtime.interface_registry)
        self.assertIsNone(runtime.interface_builder)
        self.assertIsNone(runtime.sandbox_executor)

    def test_runtime_context_accepts_runtime_services(self) -> None:
        run_context = EngineRunContext()
        method_hub = MethodHub()
        interface_registry = InMemoryInterfaceRegistry()
        artifact_store = ArtifactStore()
        log_store = LogStore()
        resource_manager = ResourceManager()

        runtime = EngineRuntimeContext(
            run_context=run_context,
            method_hub=method_hub,
            interface_registry=interface_registry,
            artifact_store=artifact_store,
            log_store=log_store,
            resource_manager=resource_manager,
        )

        self.assertIs(runtime.run_context, run_context)
        self.assertIs(runtime.method_hub, method_hub)
        self.assertIs(runtime.interface_registry, interface_registry)
        self.assertIs(runtime.artifact_store, artifact_store)
        self.assertIs(runtime.log_store, log_store)
        self.assertIs(runtime.resource_manager, resource_manager)

    def test_sandbox_run_result_records_execution_evidence(self) -> None:
        result = SandboxRunResult(
            status="completed",
            result={"rows": 3},
            artifact_refs=["artifact://orders.csv"],
            log_refs=["log://sandbox/1"],
            resource_usage={"cpu_seconds": 1},
            metadata={"interface": "orders_reader"},
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.result, {"rows": 3})
        self.assertEqual(result.artifact_refs, ["artifact://orders.csv"])
        self.assertEqual(result.log_refs, ["log://sandbox/1"])
        self.assertEqual(result.resource_usage, {"cpu_seconds": 1})
        self.assertEqual(result.metadata, {"interface": "orders_reader"})


if __name__ == "__main__":
    unittest.main()
