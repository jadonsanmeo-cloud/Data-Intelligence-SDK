"""Small OpenAI-compatible chat completion client boundary."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Protocol

from axiom_model_client import (
    ConsumerModelResolution,
    ConsumerModelResolutions,
    ModelClient,
    ResolvedTasks,
    TaskResolution,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from data_intelligence_sdk.runtime.config import get_config_manager
from data_intelligence_sdk.runtime.run_context import ModelRunContext
from data_intelligence_sdk.runtime.tracing import traceable_llm_call


class LLMClient(Protocol):
    """Completion boundary used by SDK components."""

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
    ) -> dict[str, Any]:
        """Return a JSON object produced from chat messages."""

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        *,
        stage: str,
    ) -> str:
        """Return raw model text produced from chat messages."""


# Existing SDK stage names stay stable for tracing and callers while AXIOM
# resolves the organization-owned task IDs at the Model Service boundary.
AXIOM_STAGE_TASK_MAP = MappingProxyType(
    {
        "datahub-clusterer": "data.cluster",
        "data-selector": "data.select",
        "cluster-spec-selector": "spec.select_cluster",
        "spec-builder": "spec.generate",
        "markdown-spec-builder": "spec.generate_markdown",
        "engine-selector": "execution.select_engine",
        "chat.answer": "chat.answer",
        "chat.synthesize": "chat.synthesize",
        "report.generate": "report.generate",
        "report.discover_sources": "report.discover_sources",
        "intent.generate": "intent.generate",
        "learning.review": "learning.review",
    }
)

_AXIOM_TASK_IDS = frozenset(AXIOM_STAGE_TASK_MAP.values())
_FORBIDDEN_AXIOM_FIELDS = frozenset(
    {
        "api_key",
        "base_url",
        "model",
        "model_id",
        "model_resource_id",
        "provider",
        "provider_id",
    }
)

_MODEL_PROFILE_ROLES = frozenset({"llm", "vlm", "embedding", "ocr"})
_CONSUMER_ALIASES = {
    "system": "data-intelligence",
    "sdk": "data-intelligence",
    "data-intelligence-sdk": "data-intelligence",
    "data-intelligence-api": "data-intelligence",
    "api": "data-intelligence",
    "genreport": "data-intelligence",
    "gen-report": "data-intelligence",
    "intent-service": "data-intelligence",
    "methods-hub": "data-intelligence",
    "intelligence-service": "data-intelligence",
}


def canonical_consumer_id(value: str) -> str:
    """Normalize the public consumer aliases accepted by Model Service."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("consumer_id must be a non-empty string")
    normalized = value.strip().lower().replace("_", "-")
    return _CONSUMER_ALIASES.get(normalized, normalized)


def axiom_task_for_stage(stage: str) -> str:
    """Return the governed Model Service task for an SDK LLM stage."""

    normalized = stage.strip()
    task_id = AXIOM_STAGE_TASK_MAP.get(normalized)
    if task_id is None and normalized in _AXIOM_TASK_IDS:
        task_id = normalized
    if task_id is None:
        raise ValueError(f"Unknown AXIOM model stage: {stage!r}")
    return task_id


Transport = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]

_TRANSPORT_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.5


class OpenAICompatibleLLMClient:
    """Calls an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        config_path: str | Path | None = None,
        temperature: float = 0,
        timeout: int = 60,
        transport: Transport | None = None,
    ) -> None:
        configured_base_url = ""
        configured_api_key = ""
        configured_model = ""
        if not base_url or not api_key or not model:
            settings = get_config_manager(
                str(config_path) if config_path is not None else None
            ).openrouter_settings()
            configured_base_url = settings.base_url
            configured_api_key = settings.api_key or ""
            configured_model = settings.model or ""

        self.base_url = (base_url or configured_base_url).rstrip("/")
        self.api_key = api_key or configured_api_key
        self.model = model or configured_model
        self.temperature = temperature
        self.timeout = timeout
        self._transport = transport or self._default_transport

        if not self.base_url:
            raise ValueError("base_url is required for OpenAICompatibleLLMClient.")
        if not self.api_key:
            raise ValueError("api_key is required for OpenAICompatibleLLMClient.")
        if not self.model:
            raise ValueError("model is required for OpenAICompatibleLLMClient.")

        self._complete_json_traced: dict[str, Callable[..., dict[str, Any]]] = {}
        self._complete_text_traced: dict[str, Callable[..., str]] = {}

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
    ) -> dict[str, Any]:
        trace_name = self._validate_stage(stage)
        traced_call = self._complete_json_traced.get(trace_name)
        if traced_call is None:
            traced_call = traceable_llm_call(
                self._complete_json_impl,
                name=trace_name,
            )
            self._complete_json_traced[trace_name] = traced_call
        return traced_call(messages)

    def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
    ) -> str:
        trace_name = self._validate_stage(stage)
        traced_call = self._complete_text_traced.get(trace_name)
        if traced_call is None:
            traced_call = traceable_llm_call(
                self._complete_text_impl,
                name=trace_name,
            )
            self._complete_text_traced[trace_name] = traced_call
        return traced_call(messages)

    def _validate_stage(self, stage: str) -> str:
        if not stage.strip():
            raise ValueError("stage must be a non-empty LangSmith trace name.")
        return stage

    def _complete_json_impl(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._transport(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            self.timeout,
        )
        return self._extract_json_object(response)

    def _complete_text_impl(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._transport(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            self.timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "OpenAI-compatible response did not contain message content."
            ) from exc
        if not isinstance(content, str):
            raise ValueError("OpenAI-compatible message content must be text.")
        return content

    def _extract_json_object(self, response: dict[str, Any]) -> dict[str, Any]:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "OpenAI-compatible response did not contain message content."
            ) from exc

        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise ValueError(
                "OpenAI-compatible message content must be a JSON object string."
            )

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object.")
        return parsed

    def _default_transport(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        for attempt in range(1, _TRANSPORT_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                raise ConnectionError(
                    f"OpenAI-compatible request failed with HTTP {exc.code}."
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == _TRANSPORT_ATTEMPTS:
                    raise ConnectionError(
                        "OpenAI-compatible request failed after "
                        f"{_TRANSPORT_ATTEMPTS} attempts: {exc}"
                    ) from exc
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI-compatible response must be a JSON object.")
        return parsed


class ModelServiceLLMClient:
    """Task-routed LLM client for AXIOM mode.

    The adapter deliberately accepts no provider/model configuration. It sends
    an opaque resolution ID returned by Model Service together with request
    messages, so provider credentials and aliases remain outside the SDK.
    """

    def __init__(
        self,
        model_service_url: str,
        *,
        context: ModelRunContext,
        model_client: ModelClient | Any | None = None,
        temperature: float = 0,
        timeout: float = 120,
    ) -> None:
        if not isinstance(context, ModelRunContext):
            raise TypeError("context must be a ModelRunContext")
        self.context = context
        self.temperature = temperature
        self.timeout = timeout
        self._model_client = model_client or ModelClient(
            model_service_url,
            service_token=context.service_token,
            timeout=timeout,
        )

    def resolve_tasks(
        self,
        task_ids: Iterable[str],
        *,
        config_revision: int | None = None,
    ) -> tuple[TaskResolution, ...]:
        """Resolve missing task IDs while pinning the current run revision."""

        requested = list(task_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("task_ids must be a non-empty unique iterable")
        unknown = [task_id for task_id in requested if task_id not in _AXIOM_TASK_IDS]
        if unknown:
            raise ValueError(f"Unknown AXIOM task IDs: {', '.join(unknown)}")
        if (
            config_revision is not None
            and self.context.config_revision is not None
            and config_revision != self.context.config_revision
        ):
            raise ValueError(
                "Model Service config revision changed during the AXIOM run"
            )

        missing = [
            task_id for task_id in requested if task_id not in self.context.resolutions
        ]
        if missing:
            resolved: ResolvedTasks = self._model_client.resolve_tasks(
                self.context.to_model_context(),
                missing,
                config_revision=(
                    self.context.config_revision
                    if self.context.config_revision is not None
                    else config_revision
                ),
            )
            self.context = self.context.with_resolutions(resolved)

        return tuple(self.context.resolutions[task_id] for task_id in requested)

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        stage: str,
        **options: Any,
    ) -> dict[str, Any]:
        """Complete a JSON stage through its immutable task resolution."""

        task_id = axiom_task_for_stage(stage)
        self._validate_options(options)
        resolution = self.resolve_tasks([task_id])[0]
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            **options,
        }
        response = self._model_client.complete(
            self.context.to_model_context(),
            resolution.resolution_id,
            **payload,
        )
        return _extract_json_response(response)

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        *,
        stage: str,
        **options: Any,
    ) -> str:
        """Complete a text stage through its immutable task resolution."""

        task_id = axiom_task_for_stage(stage)
        self._validate_options(options)
        resolution = self.resolve_tasks([task_id])[0]
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            **options,
        }
        response = self._model_client.complete(
            self.context.to_model_context(),
            resolution.resolution_id,
            **payload,
        )
        return _extract_text_response(response)

    @staticmethod
    def _validate_options(options: dict[str, Any]) -> None:
        forbidden = sorted(_FORBIDDEN_AXIOM_FIELDS.intersection(options))
        if forbidden:
            raise ValueError(
                "AXIOM mode does not accept provider/model aliases: "
                + ", ".join(forbidden)
            )


class ModelServiceProfileLLMClient:
    """Role-profile LLM boundary for Data Intelligence consumers.

    An explicit model resource is sent directly to Model Service for a user-
    selected run; otherwise the ``data-intelligence`` consumer profile is
    resolved once. The context keeps any resolved role and Model Service
    revision immutable for the rest of the run.
    ``ModelServiceLLMClient`` remains available for task-routed compatibility
    while callers migrate to this role-based boundary.
    """

    def __init__(
        self,
        model_service_url: str,
        *,
        context: ModelRunContext,
        model_client: ModelClient | Any | None = None,
        consumer_id: str = "data-intelligence",
        model_resource_id: str | None = None,
        temperature: float = 0,
        timeout: float = 120,
    ) -> None:
        if not isinstance(context, ModelRunContext):
            raise TypeError("context must be a ModelRunContext")
        self.context = context
        self.consumer_id = canonical_consumer_id(consumer_id)
        self.model_resource_id = (
            model_resource_id.strip() if model_resource_id else None
        )
        self.temperature = temperature
        self.timeout = timeout
        self._model_client = model_client or ModelClient(
            model_service_url,
            service_token=context.service_token,
            timeout=timeout,
        )

    def resolve_roles(
        self,
        roles: Iterable[str],
        *,
        config_revision: int | None = None,
    ) -> tuple[ConsumerModelResolution, ...]:
        """Resolve missing roles while pinning one run-level config revision."""

        requested = list(roles)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("roles must be a non-empty unique iterable")
        unknown = [role for role in requested if role not in _MODEL_PROFILE_ROLES]
        if unknown:
            raise ValueError(f"Unknown Model Service roles: {', '.join(unknown)}")
        if (
            config_revision is not None
            and self.context.config_revision is not None
            and config_revision != self.context.config_revision
        ):
            raise ValueError(
                "Model Service config revision changed during the AXIOM run"
            )

        missing = [
            role for role in requested if role not in self.context.profile_resolutions
        ]
        if missing:
            resolved: ConsumerModelResolutions = (
                self._model_client.resolve_consumer_profile(
                    self.context.to_model_context(),
                    self.consumer_id,
                    missing,
                    config_revision=(
                        self.context.config_revision
                        if self.context.config_revision is not None
                        else config_revision
                    ),
                )
            )
            if resolved.consumer_id != self.consumer_id:
                raise ValueError(
                    "Model Service returned a profile for a different consumer"
                )
            self.context = self.context.with_profile_resolutions(resolved)

        return tuple(self.context.profile_resolutions[role] for role in requested)

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        stage: str,
        **options: Any,
    ) -> dict[str, Any]:
        """Complete a JSON stage through the run's ``llm`` role."""

        response = self.complete_messages(
            messages,
            stage=stage,
            response_format={"type": "json_object"},
            **options,
        )
        return _extract_json_response(response)

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        *,
        stage: str,
        **options: Any,
    ) -> str:
        """Complete a text stage through the run's ``llm`` role."""

        response = self.complete_messages(messages, stage=stage, **options)
        return _extract_text_response(response)

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        stage: str,
        **options: Any,
    ) -> dict[str, Any]:
        """Send a native chat request through the run's ``llm`` role."""

        self._validate_stage(stage)
        self._validate_options(options)
        payload = {"messages": messages, "temperature": self.temperature, **options}
        if self.model_resource_id is not None:
            return self._model_client.complete_for_model(
                self.context.to_model_context(), self.model_resource_id, **payload
            )
        resolution = self.resolve_roles(["llm"])[0]
        return self._model_client.complete(
            self.context.to_model_context(), resolution.resolution_id, **payload
        )

    def embed(
        self,
        inputs: str | list[str],
        *,
        input_type: str = "document",
        **options: Any,
    ) -> dict[str, Any]:
        """Embed input through the run's ``embedding`` role."""

        self._validate_options(options)
        resolution = self.resolve_roles(["embedding"])[0]
        return self._model_client.embed(
            self.context.to_model_context(),
            resolution.resolution_id,
            input=inputs,
            input_type=input_type,
            **options,
        )

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        *,
        top_n: int | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Legacy rerank hook; organization profiles no longer configure it."""

        self._validate_options(options)
        resolution = self.resolve_roles(["reranker"])[0]
        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
            **options,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        return self._model_client.rerank(
            self.context.to_model_context(), resolution.resolution_id, **payload
        )

    @staticmethod
    def _validate_stage(stage: str) -> None:
        normalized = stage.strip()
        if normalized not in AXIOM_STAGE_TASK_MAP and normalized not in _AXIOM_TASK_IDS:
            raise ValueError(f"Unknown AXIOM model stage: {stage!r}")

    @staticmethod
    def _validate_options(options: dict[str, Any]) -> None:
        forbidden = sorted(_FORBIDDEN_AXIOM_FIELDS.intersection(options))
        if forbidden:
            raise ValueError(
                "AXIOM mode does not accept provider/model aliases: "
                + ", ".join(forbidden)
            )


class ModelServiceChatModel(BaseChatModel):
    """Small LangChain chat boundary backed by ``ModelServiceProfileLLMClient``.

    The adapter keeps provider/model selection out of LangChain while exposing
    the ``BaseChatModel`` protocol expected by Deep Agents.  Tool definitions
    are passed through to Model Service as OpenAI-shaped definitions.
    """

    model_name: str = "model-service"
    model: str = "model-service"
    _profile_client: ModelServiceProfileLLMClient = PrivateAttr()

    def __init__(self, profile_client: ModelServiceProfileLLMClient) -> None:
        super().__init__()
        self._profile_client = profile_client

    @property
    def _llm_type(self) -> str:
        return "axiom-model-service"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        options = dict(kwargs)
        if stop:
            options["stop"] = stop
        response = self._profile_client.complete_messages(
            convert_to_openai_messages(messages),
            stage="chat.answer",
            **options,
        )
        output = response.get("output")
        if not isinstance(output, dict):
            output = {}
        content = output.get("content", "")
        if not isinstance(content, (str, list)):
            content = str(content)
        tool_calls = _langchain_tool_calls(output.get("tool_calls"))
        message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        bound = {
            "tools": [convert_to_openai_tool(tool) for tool in tools],
            **kwargs,
        }
        if tool_choice is not None:
            bound["tool_choice"] = tool_choice
        return self.bind(**bound)

    def _get_ls_params(self, **kwargs):
        del kwargs
        return {"ls_provider": "axiom", "ls_model_name": self.model_name}


def _langchain_tool_calls(value: object) -> list[dict[str, Any]]:
    """Convert Model Service tool-call objects to LangChain's AIMessage shape."""

    if not isinstance(value, list):
        return []
    converted: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments", "{}")
        call_id = item.get("id")
        if not isinstance(name, str) or not name or not isinstance(call_id, str):
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        converted.append(
            {
                "name": name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }
        )
    return converted


def _extract_output_content(response: object) -> object:
    if not isinstance(response, dict):
        raise ValueError("Model Service response must be a JSON object.")
    output = response.get("output")
    if isinstance(output, dict) and "content" in output:
        return output["content"]
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and "content" in message:
                return message["content"]
    raise ValueError("Model Service response did not contain output content.")


def _extract_text_response(response: object) -> str:
    content = _extract_output_content(response)
    if not isinstance(content, str):
        raise ValueError("Model Service text output must be a string.")
    return content


def _extract_json_response(response: object) -> dict[str, Any]:
    content = _extract_output_content(response)
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("Model Service JSON output must be an object string.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Model Service JSON output was invalid.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model Service JSON output must be an object.")
    return parsed
