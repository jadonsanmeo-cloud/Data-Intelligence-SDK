"""Runtime boundaries for engine execution."""

from __future__ import annotations

from data_intelligence_sdk.runtime.config import (
    ArtifactSettings,
    ConfigManager,
    MethodHubSettings,
    OpenRouterSettings,
    SandboxSettings,
    get_config_manager,
)
from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext
from data_intelligence_sdk.runtime.sandbox import (
    EngineSandboxSession,
    SandboxEnvironment,
    SandboxSessionProvider,
)
from data_intelligence_sdk.runtime.interfaces import (
    InterfaceBuilder,
    InterfaceRegistry,
    InMemoryInterfaceRegistry,
)
from data_intelligence_sdk.runtime.logger import (
    ConsoleRuntimeLogger,
    FileRuntimeLogger,
    InMemoryRuntimeLogger,
    RuntimeLogger,
)
from data_intelligence_sdk.runtime.llm_client import (
    AXIOM_STAGE_TASK_MAP,
    LLMClient,
    ModelServiceChatModel,
    ModelServiceLLMClient,
    ModelServiceProfileLLMClient,
    OpenAICompatibleLLMClient,
    axiom_task_for_stage,
    canonical_consumer_id,
)
from data_intelligence_sdk.runtime.mcp_client import (
    MCPClientError,
    MCPMethodClient,
    MCPToolDefinition,
    MCPToolError,
)
from data_intelligence_sdk.runtime.run_context import (
    AxiomRunContext,
    EngineRunContext,
    ModelRunContext,
)
from data_intelligence_sdk.runtime.selected_files import (
    SelectedFilesScope,
    SelectedFilesScopeError,
)
from data_intelligence_sdk.runtime.tracing import langsmith_tracing_enabled
from data_intelligence_sdk.sandbox.artifacts import (
    ArtifactPersistenceError,
    ArtifactStore,
    CodeAttemptArtifact,
    FilesystemArtifactStore,
    RunArtifactSession,
)

__all__ = [
    "ArtifactSettings",
    "ConfigManager",
    "ConsoleRuntimeLogger",
    "ArtifactPersistenceError",
    "ArtifactStore",
    "CodeAttemptArtifact",
    "DeepAgentSandboxBackend",
    "DeepAgentSandboxSession",
    "EngineSandboxSession",
    "EngineRunContext",
    "AxiomRunContext",
    "EngineRuntimeContext",
    "FileRuntimeLogger",
    "FilesystemArtifactStore",
    "InMemoryRuntimeLogger",
    "InterfaceBuilder",
    "InterfaceRegistry",
    "InMemoryInterfaceRegistry",
    "LLMClient",
    "ModelServiceChatModel",
    "ModelRunContext",
    "ModelServiceLLMClient",
    "ModelServiceProfileLLMClient",
    "MethodHubSettings",
    "MCPClientError",
    "MCPMethodClient",
    "MCPToolDefinition",
    "MCPToolError",
    "OpenAICompatibleLLMClient",
    "AXIOM_STAGE_TASK_MAP",
    "axiom_task_for_stage",
    "canonical_consumer_id",
    "OpenRouterSettings",
    "SandboxSettings",
    "SandboxEnvironment",
    "RuntimeLogger",
    "RunArtifactSession",
    "SandboxSessionProvider",
    "SelectedFilesScope",
    "SelectedFilesScopeError",
    "get_config_manager",
    "langsmith_tracing_enabled",
]


def __getattr__(name: str) -> object:
    if name == "DeepAgentSandboxBackend":
        from data_intelligence_sdk.runtime.deep_agent_backend import (
            DeepAgentSandboxBackend,
        )

        return DeepAgentSandboxBackend
    if name == "DeepAgentSandboxSession":
        return EngineSandboxSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
