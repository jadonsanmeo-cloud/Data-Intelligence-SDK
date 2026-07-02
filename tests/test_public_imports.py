import unittest

import data_intelligence_sdk as dis
from data_intelligence_sdk.runtime import (
    EngineRuntimeContext,
    InMemoryInterfaceRegistry,
    MethodHub,
)
from data_intelligence_sdk.sandbox import SandboxRunResult


class PublicImportTests(unittest.TestCase):
    def test_root_package_exports_new_core_contracts(self) -> None:
        self.assertTrue(hasattr(dis, "DataCorpusPackage"))
        self.assertTrue(hasattr(dis, "DataHubContext"))
        self.assertTrue(hasattr(dis, "CapabilityRequirement"))
        self.assertTrue(hasattr(dis, "InterfaceDefinition"))
        self.assertTrue(hasattr(dis, "TrustLevel"))

    def test_runtime_and_sandbox_packages_export_new_contracts(self) -> None:
        runtime = EngineRuntimeContext(
            method_hub=MethodHub(),
            interface_registry=InMemoryInterfaceRegistry(),
        )
        sandbox_result = SandboxRunResult(status="completed")

        self.assertIsInstance(runtime.method_hub, MethodHub)
        self.assertEqual(sandbox_result.status, "completed")


if __name__ == "__main__":
    unittest.main()
