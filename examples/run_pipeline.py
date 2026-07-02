"""Run the default Data Intelligence pipeline against a DataCorpusPackage.

Examples:
    uv run python examples/run_pipeline.py
    uv run python examples/run_pipeline.py --source sales.csv --query "What is the total revenue?"
    uv run python examples/run_pipeline.py --source sales.csv --source customers.csv --query "What sources are available?"
    uv run python examples/run_pipeline.py --corpus examples/corpus.json --query "Summarize this corpus"

Requires OPENROUTER_API_KEY and OPENROUTER_MODEL in the environment or .env.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from data_intelligence_sdk import DataCorpusPackage, UserQuery
from data_intelligence_sdk.defaults import create_default_pipeline_from_openrouter

load_dotenv()


SAMPLE_CSV = """country,status,revenue
US,complete,10.5
US,pending,4.5
CA,complete,7
"""


def _create_sample_csv() -> tuple[tempfile.TemporaryDirectory[str], str]:
    temp_dir = tempfile.TemporaryDirectory()
    csv_path = Path(temp_dir.name) / "sales.csv"
    csv_path.write_text(SAMPLE_CSV, encoding="utf-8")
    return temp_dir, str(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Data Intelligence SDK pipeline."
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Source reference to include in the DataCorpusPackage. Can be repeated.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Deprecated alias for --source for one CSV file.",
    )
    parser.add_argument(
        "--corpus",
        dest="corpus_path",
        help="Path to a JSON file with sources, schemas, and metadata fields.",
    )
    parser.add_argument(
        "--query",
        default="What is the total revenue?",
        help="Question to ask about the data corpus.",
    )
    args = parser.parse_args()

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.corpus_path:
        corpus_payload = json.loads(Path(args.corpus_path).read_text(encoding="utf-8"))
        corpus_package = DataCorpusPackage(
            sources=list(corpus_payload.get("sources", [])),
            schemas=dict(corpus_payload.get("schemas", {})),
            metadata=dict(corpus_payload.get("metadata", {})),
        )
    else:
        sources = list(args.sources or [])
        if args.csv_path:
            sources.append(args.csv_path)
        if not sources:
            temp_dir, sample_csv_path = _create_sample_csv()
            sources.append(sample_csv_path)
        corpus_package = DataCorpusPackage(sources=sources)

    try:
        pipeline = create_default_pipeline_from_openrouter()

        response = pipeline.run(
            UserQuery(args.query),
            corpus_package,
        )

        print(response.answer)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
