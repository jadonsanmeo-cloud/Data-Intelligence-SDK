"""Static DataHub catalog for the currently parsed NAPH dataset.

For this phase, DataHub is not a parser, a storage engine, or a vector store.
It is a list of files that upstream processing has already prepared, plus the
minimal metadata an intent analyzer or engine needs to understand those files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from data_intelligence_sdk.core.types import DataHubContext

FileType = Literal["csv", "pdf"]


@dataclass(slots=True)
class ColumnSchema:
    """Column-level metadata for a CSV file in the DataHub catalog."""

    name: str
    data_type: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataHubFile:
    """One prepared file known to DataHub."""

    file_id: str
    file_type: FileType
    name: str
    link: str
    description: str
    parsed_link: str | None = None
    schema: list[ColumnSchema] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DataHub:
    """A static catalog over already parsed data files."""

    files: list[DataHubFile]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_file(self, file_id: str) -> DataHubFile:
        for item in self.files:
            if item.file_id == file_id:
                return item
        raise KeyError(file_id)

    def list_files(self, file_type: FileType | None = None) -> list[DataHubFile]:
        if file_type is None:
            return list(self.files)
        return [item for item in self.files if item.file_type == file_type]

    def to_context(self) -> DataHubContext:
        return DataHubContext(
            sources=[item.file_id for item in self.files],
            schemas={
                item.file_id: {
                    "columns": [asdict(column) for column in item.schema],
                    "metadata": item.metadata,
                }
                for item in self.files
                if item.schema
            },
            metadata={
                **self.metadata,
                "files": [item.to_dict() for item in self.files],
            },
        )


NAPH_OPEN_DATA_COLUMNS = [
    ColumnSchema("Section", "string", "Section in the 10th NAPH annual report."),
    ColumnSchema("Output_Reference", "string", "Table or chart number in the 10th NAPH annual report."),
    ColumnSchema("Output_Description", "string", "Description of the table, chart, or National Standard."),
    ColumnSchema("Period", "string", "Assessment period. Interpret together with WhenOccurred."),
    ColumnSchema("WhenOccurred", "string", "Point within the Period when the event occurred."),
    ColumnSchema("Organisation_Code", "string", "Pulmonary hypertension centre code."),
    ColumnSchema("Organisation_Name", "string", "Pulmonary hypertension centre short name."),
    ColumnSchema("Identifier", "string", "Metric/value label within the output."),
    ColumnSchema("DataType", "string", "Data type or unit of Value and Standard_Target."),
    ColumnSchema("Value", "string", "Reported metric value."),
    ColumnSchema("Standard_Target", "string", "Target value for a NAPH National Standard."),
    ColumnSchema("Standard_Met", "string", "Whether the reported value met the standard target."),
    ColumnSchema("Value_Denominator", "integer", "Denominator used to calculate percentage values."),
    ColumnSchema("Value_Numerator", "integer", "Numerator used to calculate percentage values."),
    ColumnSchema("OrderId", "integer", "Sort key for intuitive row ordering."),
]

NAPH_SPECIFICATION_COLUMNS = [
    ColumnSchema("Field_Name", "string", "Name of the field in the open data CSV."),
    ColumnSchema("Field_Type", "string", "Declared field type."),
    ColumnSchema("Field_Description", "string", "Human-readable field description."),
    ColumnSchema("Notes", "string", "Additional notes about the field."),
    ColumnSchema("Field_Values", "string", "Allowed values when the field is categorical."),
]


NAPH_DATAHUB = DataHub(
    files=[
        DataHubFile(
            file_id="csv:naph_10ar_open_data_specification",
            file_type="csv",
            name="NAPH 10AR - Open Data - Specification.csv",
            link="data_parsed/NAPH 10AR - Open Data - Specification.csv",
            description=(
                "CSV data dictionary for the NAPH open data file. It describes field names, "
                "declared types, field meanings, notes, and allowed values."
            ),
            schema=NAPH_SPECIFICATION_COLUMNS,
            metadata={"row_count": 15, "column_count": 5, "encoding": "cp1252"},
        ),
        DataHubFile(
            file_id="csv:naph_10ar_open_data",
            file_type="csv",
            name="NAPH 10AR - Open Data.csv",
            link="data_parsed/NAPH 10AR - Open Data.csv",
            description=(
                "Main NAPH open data table containing audit metrics, values, periods, "
                "organisations, identifiers, standards, and numerator/denominator fields."
            ),
            schema=NAPH_OPEN_DATA_COLUMNS,
            metadata={"row_count": 91749, "column_count": 15, "encoding": "utf-8-sig"},
        ),
        DataHubFile(
            file_id="pdf:naph_10ar_main_report",
            file_type="pdf",
            name="NAPH 10AR - Main Report.pdf",
            link="data_parsed/documents/naph_10ar_main_report.md",
            parsed_link="data_parsed/documents/naph_10ar_main_report.md",
            description=(
                "Parsed Markdown for the NAPH 10th annual report. It covers the National "
                "Audit of Pulmonary Hypertension for Great Britain in 2018-19, including "
                "foreword, overview, standards, reference tables, and explanatory notes."
            ),
            metadata={"source_file": "NAPH 10AR - Main Report.pdf", "page_count": 102},
        ),
        DataHubFile(
            file_id="pdf:naph_10ar_supplementary_survival_analysis",
            file_type="pdf",
            name="NAPH 10AR - Supplementary Survival Analysis.pdf",
            link="data_parsed/documents/naph_10ar_supplementary_survival_analysis.md",
            parsed_link="data_parsed/documents/naph_10ar_supplementary_survival_analysis.md",
            description=(
                "Parsed Markdown for the supplementary survival analysis report. It contains "
                "survival analysis outputs and group characteristics related to the NAPH audit."
            ),
            metadata={"source_file": "NAPH 10AR - Supplementary Survival Analysis.pdf", "page_count": 33},
        ),
        DataHubFile(
            file_id="pdf:naph_lap_10ar_v1_0_golden_jubilee_for_web",
            file_type="pdf",
            name="NAPH LAP 10AR v1.0 Golden Jubilee for web.pdf",
            link="data_parsed/documents/naph_lap_10ar_v1_0_golden_jubilee_for_web.md",
            parsed_link="data_parsed/documents/naph_lap_10ar_v1_0_golden_jubilee_for_web.md",
            description="Parsed Markdown for the Golden Jubilee local audit provider report.",
            metadata={"source_file": "NAPH LAP 10AR v1.0 Golden Jubilee for web.pdf", "page_count": 1},
        ),
        DataHubFile(
            file_id="pdf:naph_lap_10ar_v1_0_imperial_for_web",
            file_type="pdf",
            name="NAPH LAP 10AR v1.0 Imperial for web.pdf",
            link="data_parsed/documents/naph_lap_10ar_v1_0_imperial_for_web.md",
            parsed_link="data_parsed/documents/naph_lap_10ar_v1_0_imperial_for_web.md",
            description="Parsed Markdown for the Imperial local audit provider report.",
            metadata={"source_file": "NAPH LAP 10AR v1.0 Imperial for web.pdf", "page_count": 1},
        ),
        DataHubFile(
            file_id="pdf:naph_lap_10ar_v1_0_newcastle_for_web",
            file_type="pdf",
            name="NAPH LAP 10AR v1.0 Newcastle for web.pdf",
            link="data_parsed/documents/naph_lap_10ar_v1_0_newcastle_for_web.md",
            parsed_link="data_parsed/documents/naph_lap_10ar_v1_0_newcastle_for_web.md",
            description="Parsed Markdown for the Newcastle local audit provider report.",
            metadata={"source_file": "NAPH LAP 10AR v1.0 Newcastle for web.pdf", "page_count": 1},
        ),
        DataHubFile(
            file_id="pdf:naph_lap_10ar_v1_0_royal_brompton_for_web",
            file_type="pdf",
            name="NAPH LAP 10AR v1.0 Royal Brompton for web.pdf",
            link="data_parsed/documents/naph_lap_10ar_v1_0_royal_brompton_for_web.md",
            parsed_link="data_parsed/documents/naph_lap_10ar_v1_0_royal_brompton_for_web.md",
            description="Parsed Markdown for the Royal Brompton local audit provider report.",
            metadata={"source_file": "NAPH LAP 10AR v1.0 Royal Brompton for web.pdf", "page_count": 1},
        ),
        DataHubFile(
            file_id="pdf:naph_lap_10ar_v1_0_sheffield_for_web",
            file_type="pdf",
            name="NAPH LAP 10AR v1.0 Sheffield for web.pdf",
            link="data_parsed/documents/naph_lap_10ar_v1_0_sheffield_for_web.md",
            parsed_link="data_parsed/documents/naph_lap_10ar_v1_0_sheffield_for_web.md",
            description="Parsed Markdown for the Sheffield local audit provider report.",
            metadata={"source_file": "NAPH LAP 10AR v1.0 Sheffield for web.pdf", "page_count": 1},
        ),
    ],
    metadata={
        "concept": "static_file_catalog",
        "dataset": "NAPH 10AR parsed data",
        "description": (
            "Catalog of prepared NAPH files. CSV entries expose schema and metadata; "
            "PDF entries point to already parsed Markdown documents."
        ),
        "base_dir": "data_parsed",
    },
)
