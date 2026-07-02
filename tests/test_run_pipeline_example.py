import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data_intelligence_sdk.core.types import FinalResponse


class FakePipeline:
    def run(self, query, corpus_package):
        self.query = query
        self.corpus_package = corpus_package
        return FinalResponse(answer="openrouter answer")


def load_run_pipeline_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_example", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunPipelineExampleTests(unittest.TestCase):
    def test_main_uses_openrouter_pipeline_without_flag(self) -> None:
        module = load_run_pipeline_module()
        fake_pipeline = FakePipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sales.csv"
            csv_path.write_text(
                "country,status,revenue\nUS,complete,10\n", encoding="utf-8"
            )
            argv = [
                "run_pipeline.py",
                "--csv",
                str(csv_path),
                "--query",
                "What columns are in this file?",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    module,
                    "create_default_pipeline_from_openrouter",
                    return_value=fake_pipeline,
                ) as create_openrouter,
                redirect_stdout(io.StringIO()) as stdout,
            ):
                module.main()

        create_openrouter.assert_called_once_with()
        self.assertEqual(fake_pipeline.query.text, "What columns are in this file?")
        self.assertEqual(fake_pipeline.corpus_package.sources, [str(csv_path)])
        self.assertEqual(stdout.getvalue().strip(), "openrouter answer")

    def test_repeated_source_arguments_build_corpus_sources(self) -> None:
        module = load_run_pipeline_module()
        fake_pipeline = FakePipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            sales_path = Path(temp_dir) / "sales.csv"
            customers_path = Path(temp_dir) / "customers.csv"
            sales_path.write_text("country,status,revenue\nUS,complete,10\n", encoding="utf-8")
            customers_path.write_text("customer_id,country\n1,US\n", encoding="utf-8")
            argv = [
                "run_pipeline.py",
                "--source",
                str(sales_path),
                "--source",
                str(customers_path),
                "--query",
                "What sources are available?",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(module, "create_default_pipeline_from_openrouter", return_value=fake_pipeline),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                module.main()

        self.assertEqual(fake_pipeline.query.text, "What sources are available?")
        self.assertEqual(fake_pipeline.corpus_package.sources, [str(sales_path), str(customers_path)])
        self.assertEqual(fake_pipeline.corpus_package.schemas, {})
        self.assertEqual(stdout.getvalue().strip(), "openrouter answer")

    def test_corpus_json_builds_full_corpus_package(self) -> None:
        module = load_run_pipeline_module()
        fake_pipeline = FakePipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.json"
            corpus_path.write_text(
                json.dumps(
                    {
                        "sources": ["data/sales.csv", "data/customers.csv"],
                        "schemas": {"data/sales.csv": {"columns": ["country", "revenue"]}},
                        "metadata": {"description": "demo corpus"},
                    }
                ),
                encoding="utf-8",
            )
            argv = ["run_pipeline.py", "--corpus", str(corpus_path), "--query", "Summarize this corpus"]

            with (
                patch.object(sys, "argv", argv),
                patch.object(module, "create_default_pipeline_from_openrouter", return_value=fake_pipeline),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                module.main()

        self.assertEqual(fake_pipeline.query.text, "Summarize this corpus")
        self.assertEqual(fake_pipeline.corpus_package.sources, ["data/sales.csv", "data/customers.csv"])
        self.assertEqual(fake_pipeline.corpus_package.schemas, {"data/sales.csv": {"columns": ["country", "revenue"]}})
        self.assertEqual(fake_pipeline.corpus_package.metadata, {"description": "demo corpus"})
        self.assertEqual(stdout.getvalue().strip(), "openrouter answer")

    def test_openrouter_flag_is_not_supported(self) -> None:
        module = load_run_pipeline_module()

        with patch.object(sys, "argv", ["run_pipeline.py", "--openrouter"]):
            with self.assertRaises(SystemExit) as raised:
                module.main()

        self.assertNotEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
