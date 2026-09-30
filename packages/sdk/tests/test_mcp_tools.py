import pytest
from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext
from data_intelligence_sdk.runtime.mcp_client import MCPToolDefinition
from data_intelligence_sdk.runtime.selected_files import SelectedFilesScope
from data_intelligence_sdk.core.types import SafeguardAssessment, SafeguardFinding
from data_intelligence_sdk.tools.mcp import (
    SelectedFilesScopeError,
    _call_remote_tool,
    create_mcp_tools,
)


class RecordingMcpClient:
    def __init__(self, result=None) -> None:
        self.arguments = None
        self.result = result if result is not None else {"results": []}

    def call_tool(self, name, arguments):
        self.arguments = (name, arguments)
        return self.result


def test_selected_files_scope_contains_authorized_document_ids() -> None:
    scope = SelectedFilesScope(document_ids=("document-1",))
    runtime = EngineRuntimeContext(selected_files_scope=scope)

    assert runtime.selected_files_scope is scope
    assert scope.document_ids == ("document-1",)


def test_workspace_id_is_injected_for_selected_file_lookup() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(
        mcp_client=client,
        workspace_id="workspace-1",
        selected_files_scope=SelectedFilesScope(document_ids=("document-1",)),
    )

    _call_remote_tool(
        runtime,
        MCPToolDefinition(name="corpus_get_file_ingested_data"),
        {"document_id": "document-1"},
    )

    assert client.arguments == (
        "corpus_get_file_ingested_data",
        {"document_id": "document-1", "workspace_id": "workspace-1"},
    )


def test_blocked_assessment_prevents_remote_data_access() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(mcp_client=client)
    runtime.run_context.set_safeguard_assessment(
        SafeguardAssessment(
            decision="blocked",
            findings=[
                SafeguardFinding(
                    id="unsafe-source",
                    category="connection_risk",
                    severity="critical",
                    title="Unsafe source mapping",
                    detail="The requested join cannot be verified.",
                    blocking=True,
                )
            ],
        )
    )

    with pytest.raises(PermissionError, match="safeguard assessment"):
        _call_remote_tool(
            runtime,
            MCPToolDefinition(name="corpus_retrieve_context"),
            {"query": "revenue"},
        )

    assert client.arguments is None


def test_empty_selected_file_scope_does_not_block_skill_materialization() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(
        mcp_client=client,
        workspace_id="workspace-1",
        selected_files_scope=SelectedFilesScope(document_ids=()),
    )

    _call_remote_tool(
        runtime,
        MCPToolDefinition(name="materialize_skill"),
        {"skill_markdown": "---\nname: reusable-workflow\n---\n# Workflow"},
    )

    assert client.arguments == (
        "materialize_skill",
        {
            "skill_markdown": "---\nname: reusable-workflow\n---\n# Workflow",
            "workspace_id": "workspace-1",
        },
    )
    assert runtime.run_context.data_accessed is False


def test_workspace_scope_is_hidden_from_model_and_injected_for_bm25_search() -> None:
    client = RecordingMcpClient()
    definition = MCPToolDefinition(
        name="corpus_bm25_search",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "workspace_id": {"type": "string"},
            },
            "required": ["query", "workspace_id"],
        },
    )
    runtime = EngineRuntimeContext(
        mcp_client=client,
        mcp_tools=(definition,),
        workspace_id="workspace-1",
    )

    tool = create_mcp_tools(runtime)[0]
    tool.invoke({"query": "revenue"})

    assert tool.args_schema == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    assert client.arguments == (
        "corpus_bm25_search",
        {"query": "revenue", "workspace_id": "workspace-1"},
    )
    assert runtime.run_context.data_accessed is True


def test_selected_files_override_corpus_search_document_filter() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(
        mcp_client=client,
        selected_files_scope=SelectedFilesScope(
            document_ids=("document-1", "document-2")
        ),
    )

    _call_remote_tool(
        runtime,
        MCPToolDefinition(name="corpus_vector_search"),
        {
            "query": "climate impact",
            "workspace_id": "workspace-1",
            "document_ids": ["document-outside-scope"],
        },
    )

    assert client.arguments == (
        "corpus_vector_search",
        {
            "query": "climate impact",
            "workspace_id": "workspace-1",
            "document_ids": ["document-1", "document-2"],
        },
    )


def test_corpus_retrieval_registers_inspectable_source_evidence() -> None:
    client = RecordingMcpClient(
        {
            "chunks": [
                {
                    "document": {
                        "document_id": "document-1",
                        "file_name": "finance.pdf",
                    },
                    "content_id": "chunk-12",
                    "position": {"page": 4, "chunk_index": 12},
                    "text": "Revenue was 12 million.",
                }
            ]
        }
    )
    runtime = EngineRuntimeContext(mcp_client=client, workspace_id="workspace-1")

    _call_remote_tool(
        runtime,
        MCPToolDefinition(name="corpus_retrieve_context"),
        {"query": "revenue"},
    )

    evidence = runtime.run_context.citation_evidence
    assert len(evidence) == 1
    assert evidence[0].source == "finance.pdf"
    assert evidence[0].locator == "page 4 · chunk 12"
    assert evidence[0].excerpt == "Revenue was 12 million."
    assert evidence[0].document_id == "document-1"
    assert evidence[0].content_id == "chunk-12"


def test_corpus_retrieval_deduplicates_chunks_within_run_and_isolates_runs() -> None:
    result = {
        "chunks": [
            {
                "document": {
                    "document_id": "document-1",
                    "file_name": "finance.pdf",
                },
                "content_id": "chunk-12",
                "position": {"page": 4, "chunk_index": 12},
                "text": "Revenue was 12 million.",
            }
        ]
    }
    first_runtime = EngineRuntimeContext(mcp_client=RecordingMcpClient(result))
    second_runtime = EngineRuntimeContext(mcp_client=RecordingMcpClient(result))
    definition = MCPToolDefinition(name="corpus_retrieve_context")

    _call_remote_tool(first_runtime, definition, {"query": "revenue"})
    first_id = first_runtime.run_context.citation_evidence[0].id
    _call_remote_tool(first_runtime, definition, {"query": "monthly revenue"})
    _call_remote_tool(second_runtime, definition, {"query": "revenue"})

    assert len(first_runtime.run_context.citation_evidence) == 1
    assert second_runtime.run_context.citation_evidence[0].id != first_id


def test_unstructured_tool_result_does_not_register_citation_evidence() -> None:
    runtime = EngineRuntimeContext(
        mcp_client=RecordingMcpClient("No source metadata returned.")
    )

    _call_remote_tool(
        runtime,
        MCPToolDefinition(name="corpus_retrieve_context"),
        {"query": "revenue"},
    )

    assert runtime.run_context.citation_evidence == ()


def test_selected_files_scope_forces_every_corpus_search_filter() -> None:
    for tool_name in (
        "corpus_vector_search",
        "corpus_bm25_search",
        "corpus_retrieve_context",
    ):
        client = RecordingMcpClient()
        runtime = EngineRuntimeContext(
            mcp_client=client,
            selected_files_scope=SelectedFilesScope(
                document_ids=("document-1", "document-2")
            ),
        )

        _call_remote_tool(
            runtime,
            MCPToolDefinition(name=tool_name),
            {
                "query": "climate impact",
                "workspace_id": "workspace-1",
                "document_id": "document-outside-scope",
                "document_ids": ["document-outside-scope"],
            },
        )

        assert client.arguments == (
            tool_name,
            {
                "query": "climate impact",
                "workspace_id": "workspace-1",
                "document_ids": ["document-1", "document-2"],
            },
        )


def test_selected_files_scope_infers_single_document_lookup() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(
        mcp_client=client,
        selected_files_scope=SelectedFilesScope(document_ids=("document-1",)),
    )

    _call_remote_tool(
        runtime,
        MCPToolDefinition(name="corpus_get_file_ingested_data"),
        {
            "workspace_id": "workspace-1",
            "file_name": "outside-scope.pdf",
            "bucket": "outside-bucket",
            "object_key": "outside-key",
        },
    )

    assert client.arguments == (
        "corpus_get_file_ingested_data",
        {
            "workspace_id": "workspace-1",
            "document_id": "document-1",
        },
    )


def test_selected_files_scope_rejects_ambiguous_document_lookup() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(
        mcp_client=client,
        selected_files_scope=SelectedFilesScope(
            document_ids=("document-1", "document-2")
        ),
    )

    with pytest.raises(SelectedFilesScopeError, match="document_id"):
        _call_remote_tool(
            runtime,
            MCPToolDefinition(name="corpus_get_file_ingested_data"),
            {"workspace_id": "workspace-1"},
        )

    assert client.arguments is None


def test_selected_files_scope_rejects_out_of_scope_document_lookup() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(
        mcp_client=client,
        selected_files_scope=SelectedFilesScope(document_ids=("document-1",)),
    )

    with pytest.raises(SelectedFilesScopeError, match="outside"):
        _call_remote_tool(
            runtime,
            MCPToolDefinition(name="corpus_get_file_ingested_data"),
            {
                "workspace_id": "workspace-1",
                "document_id": "document-outside-scope",
            },
        )

    assert client.arguments is None
    assert runtime.run_context.trace.method_calls[-1].status == "failed"


def test_selected_files_scope_validates_neighbor_document() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(
        mcp_client=client,
        selected_files_scope=SelectedFilesScope(
            document_ids=("document-1", "document-2")
        ),
    )

    _call_remote_tool(
        runtime,
        MCPToolDefinition(name="get_neighbor_chunk"),
        {
            "workspace_id": "workspace-1",
            "file_id": "document-1",
            "inc": 1,
            "des": 1,
            "chunk_id": 2,
        },
    )

    assert client.arguments == (
        "get_neighbor_chunk",
        {
            "workspace_id": "workspace-1",
            "file_id": "document-1",
            "inc": 1,
            "des": 1,
            "chunk_id": 2,
        },
    )


def test_selected_files_scope_rejects_neighbor_without_document() -> None:
    client = RecordingMcpClient()
    runtime = EngineRuntimeContext(
        mcp_client=client,
        selected_files_scope=SelectedFilesScope(
            document_ids=("document-1", "document-2")
        ),
    )

    with pytest.raises(SelectedFilesScopeError, match="file_id"):
        _call_remote_tool(
            runtime,
            MCPToolDefinition(name="get_neighbor_chunk"),
            {
                "workspace_id": "workspace-1",
                "inc": 1,
                "des": 1,
                "chunk_id": 2,
            },
        )

    assert client.arguments is None
