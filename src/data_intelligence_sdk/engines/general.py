"""General-purpose fallback engine backed by MethodHub tools."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from data_intelligence_sdk.core.types import (
    DataCorpusPackage,
    EngineOutput,
    ExecutionSpec,
    UserContext,
)
from data_intelligence_sdk.runtime.engine_runtime import EngineRuntimeContext


class GeneralPurposeEngine:
    """Fallback engine that answers with available MethodHub capabilities."""

    name = "general_purpose"

    def __init__(self, llm: object, *, allow_method_generation: bool = True) -> None:
        self.llm = llm
        self.allow_method_generation = allow_method_generation

    @classmethod
    def from_openrouter(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        allow_method_generation: bool = True,
    ) -> "GeneralPurposeEngine":
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when no api_key is passed."
            )
        model = model or os.environ.get("OPENROUTER_MODEL")
        if not model:
            raise ValueError("OPENROUTER_MODEL is required when no model is passed.")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=api_key, base_url="https://openrouter.ai/api/v1", model=model
        )
        return cls(llm=llm, allow_method_generation=allow_method_generation)

    def can_handle(self, spec: ExecutionSpec) -> bool:
        return spec.engine_hint == self.name or spec.intent in {"custom", "unknown"}

    def run(
        self,
        spec: ExecutionSpec,
        corpus_package: DataCorpusPackage,
        runtime: EngineRuntimeContext,
        user_context: UserContext | None = None,
    ) -> EngineOutput:
        del user_context
        runtime.run_context.record_step(
            "general_purpose_start", inputs={"objective": spec.objective}
        )
        if not corpus_package.sources:
            return runtime.run_context.build_output(
                engine_name=self.name,
                result="No data source was provided. Add a CSV path to DataCorpusPackage.sources.",
                metadata={"sources": []},
            )

        agent_result = self._run_agent(spec, corpus_package, runtime)
        if agent_result:
            return runtime.run_context.build_output(
                engine_name=self.name,
                result=agent_result,
                metadata={"sources": corpus_package.sources},
            )

        if self.allow_method_generation:
            return self._handle_method_generation(spec, corpus_package, runtime)

        runtime.run_context.record_step("method_generation_disabled", status="skipped")
        return runtime.run_context.build_output(
            engine_name=self.name,
            result="The required capability is not available, and method generation is not configured.",
            metadata={"sources": corpus_package.sources},
        )

    def _run_known_method(
        self, spec: ExecutionSpec, csv_source: str | None, runtime: EngineRuntimeContext
    ) -> str | None:
        if csv_source is None:
            return None
        objective = spec.objective.lower()
        if "count" in objective or "how many" in objective:
            result = self._call_method(runtime, "count_csv", {"path": csv_source})
            return f"Count: {result['count']}"
        if "sum" in objective or "total" in objective:
            column = self._guess_column(objective, default="revenue")
            result = self._call_method(
                runtime, "sum_csv", {"path": csv_source, "column": column}
            )
            return f"Sum of {column}: {result['sum']}"
        if "scan" in objective or "inspect" in objective or "columns" in objective:
            result = self._call_method(runtime, "scan_csv", {"path": csv_source})
            return f"CSV columns: {', '.join(result['columns'])}; rows: {result['row_count']}"
        return None

    def _call_method(
        self, runtime: EngineRuntimeContext, method_name: str, inputs: dict[str, Any]
    ) -> Any:
        try:
            method = runtime.method_hub.get(method_name)
            output = method(**inputs)
            runtime.run_context.record_method_call(
                method_name,
                status="completed",
                inputs=inputs,
                outputs={"result": output},
            )
            return output
        except Exception as exc:
            runtime.run_context.record_method_call(
                method_name, status="failed", inputs=inputs, outputs={"error": str(exc)}
            )
            raise

    def _build_agent_prompt(
        self,
        spec: ExecutionSpec,
        corpus_package: DataCorpusPackage,
        runtime: EngineRuntimeContext,
    ) -> str:
        method_lines = []
        for registered in runtime.method_hub.list_methods():
            description = registered.metadata.get("description", "")
            capabilities = ", ".join(registered.capability_names)
            method_lines.append(
                f"- {registered.name}: {description} Capabilities: {capabilities}"
            )
        sources = (
            "\n".join(f"- {source}" for source in corpus_package.sources) or "- none"
        )
        methods = "\n".join(method_lines) or "- none"
        return (
            "Answer the user query using MethodHub tools for factual data access.\n"
            "Do not guess facts about data files; call tools instead.\n\n"
            f"User objective: {spec.objective}\n\n"
            f"Data sources:\n{sources}\n\n"
            f"Available tools:\n{methods}\n"
        )

    def _build_agent_tools(self, runtime: EngineRuntimeContext) -> dict[str, object]:
        tools = {}
        for registered in runtime.method_hub.list_methods():

            def make_tool(method_name: str):
                def tool(args: dict[str, Any]) -> Any:
                    return self._call_method(runtime, method_name, args)

                return tool

            tools[registered.name] = make_tool(registered.name)
        return tools

    def _run_agent(
        self,
        spec: ExecutionSpec,
        corpus_package: DataCorpusPackage,
        runtime: EngineRuntimeContext,
    ) -> str | None:
        prompt = self._build_agent_prompt(spec, corpus_package, runtime)
        tools = self._build_agent_tools(runtime)

        if hasattr(self.llm, "run"):
            response = self.llm.run(prompt=prompt, tools=tools)
            for call in response.get("tool_calls", []):
                tool_name = call["name"]
                args = call.get("args", {})
                tools[tool_name](args)
            return str(response.get("final_answer", ""))

        return self._run_langchain_agent(prompt, tools)

    def _run_langchain_agent(self, prompt: str, tools: dict[str, object]) -> str | None:
        if hasattr(self.llm, "invoke"):
            response = self.llm.invoke(prompt)
            content = str(getattr(response, "content", response))
            tool_request = self._parse_tool_request(content)
            if tool_request is None:
                return content
            tool_name = tool_request["tool"]
            raw_arguments = tool_request.get(
                "arguments", tool_request.get("parameters", {})
            )
            arguments = self._normalize_tool_arguments(raw_arguments)
            result = tools[tool_name](arguments)
            return self._format_tool_result(tool_name, result)
        return None

    def _parse_tool_request(self, content: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or "tool" not in parsed:
            return None
        return parsed

    def _normalize_tool_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        if "file_path" in normalized and "path" not in normalized:
            normalized["path"] = normalized.pop("file_path")
        return normalized

    def _format_tool_result(self, tool_name: str, result: Any) -> str:
        if tool_name == "scan_csv" and isinstance(result, dict):
            return f"CSV columns: {', '.join(result['columns'])}; rows: {result['row_count']}"
        if tool_name == "count_csv" and isinstance(result, dict):
            return f"Count: {result['count']}"
        if tool_name == "sum_csv" and isinstance(result, dict):
            return f"Sum of {result['column']}: {result['sum']}"
        return str(result)

    def _handle_method_generation(
        self,
        spec: ExecutionSpec,
        corpus_package: DataCorpusPackage,
        runtime: EngineRuntimeContext,
    ) -> EngineOutput:
        if (
            runtime.interface_builder is None
            or runtime.sandbox_executor is None
            or runtime.interface_registry is None
        ):
            runtime.run_context.record_step(
                "method_generation_unavailable",
                status="skipped",
                description="Required interface builder, sandbox executor, or interface registry is missing.",
            )
            return runtime.run_context.build_output(
                engine_name=self.name,
                result="The required capability is not available, and method generation is not configured.",
                metadata={"sources": corpus_package.sources},
            )

        requirement = (
            spec.capability_requirements[0] if spec.capability_requirements else None
        )
        if requirement is None:
            runtime.run_context.record_step(
                "method_generation_unavailable", status="skipped"
            )
            return runtime.run_context.build_output(
                engine_name=self.name,
                result="The required capability is not available, and method generation is not configured.",
                metadata={"sources": corpus_package.sources},
            )

        interface = runtime.interface_builder.propose(requirement, corpus_package)
        if interface.trust_level != "generated_validated":
            interface.trust_level = "generated_unvalidated"
        sandbox_result = runtime.sandbox_executor.validate(interface, {}, None)
        if sandbox_result.status == "completed":
            interface.trust_level = "generated_validated"
            runtime.interface_registry.register(interface)
            runtime.run_context.record_step(
                "method_generation_validated", outputs={"interface": interface.name}
            )
            return runtime.run_context.build_output(
                engine_name=self.name,
                result=f"Generated and validated interface: {interface.name}",
                metadata={
                    "sources": corpus_package.sources,
                    "interface_defs": [interface],
                    "sandbox_results": [sandbox_result],
                },
            )
        runtime.run_context.record_step(
            "method_generation_failed",
            status="failed",
            outputs={"interface": interface.name},
        )
        return runtime.run_context.build_output(
            engine_name=self.name,
            result=f"Generated interface validation failed: {interface.name}",
            metadata={
                "sources": corpus_package.sources,
                "interface_defs": [interface],
                "sandbox_results": [sandbox_result],
            },
        )

    def _guess_column(self, objective: str, *, default: str) -> str:
        tokens = [token.strip(" ?.!,") for token in objective.split()]
        if "revenue" in tokens:
            return "revenue"
        return default


def _to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {"result": value}
