"""Stateless operation endpoints for the Data Intelligence runtime."""

from __future__ import annotations

import asyncio
import logging
import secrets
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from data_intelligence_api.application.runtime_operations import (
    execute_instant,  # noqa: F401 - retained for downstream patch compatibility
    execute_thinking,  # noqa: F401 - retained for downstream patch compatibility
    prepare_spec,
    revise_spec,
    resume_execution,
    select_instant_engine,
    select_thinking_engine,
    stream_instant,
    stream_report_events,
    stream_thinking,
)
from data_intelligence_api.application.workflow import (
    PipelineFactory,
    default_pipeline_factory,
)
from data_intelligence_api.http.schemas.runtime_operations import (
    InstantExecutionRequest,
    normalize_citation_metadata,
    OperationEnvelope,
    PrepareSpecRequest,
    ReviseSpecRequest,
    RuntimeErrorResponse,
    ResumeExecutionRequest,
    ThinkingExecutionRequest,
)
from data_intelligence_api.http.streaming import chunk_text, encode_sse
from data_intelligence_api.infrastructure.config.settings import ApiSettings
from data_intelligence_sdk.core.errors import EngineSelectionError
from data_intelligence_sdk.core.types import FinalResponse, UserInputRequired
from data_intelligence_sdk.runtime.logger import ConsoleRuntimeLogger, RuntimeLogger

logger = logging.getLogger(__name__)


class _StreamingRuntimeLogger(RuntimeLogger):
    def __init__(self) -> None:
        self.events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._console_logger = ConsoleRuntimeLogger()

    def log(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self._console_logger.log(event, payload)
        if event == "pipeline.runtime_event":
            self.enqueue("runtime", payload or {})

    def enqueue(self, kind: str, payload: Any) -> None:
        self._loop.call_soon_threadsafe(self.events.put_nowait, (kind, payload))


def _completed_runtime_payload(result: FinalResponse) -> dict[str, object]:
    metadata = normalize_citation_metadata(dict(result.metadata), result.answer)
    return {
        "output_text": result.answer,
        "evidence": (asdict(result.evidence) if result.evidence is not None else None),
        "metadata": metadata,
    }


async def _stream_execution_events(
    events: Iterator[str | FinalResponse | UserInputRequired],
    *,
    operation_id: str,
    response_id: str,
    runtime_logger: _StreamingRuntimeLogger | None = None,
) -> AsyncIterator[str]:
    runtime_logger = runtime_logger or _StreamingRuntimeLogger()
    result: FinalResponse | None = None
    emitted_delta = False

    stop_worker = threading.Event()

    def produce_events() -> None:
        try:
            for item in events:
                if stop_worker.is_set():
                    break
                runtime_logger.enqueue("execution", item)
        except BaseException as error:
            runtime_logger.enqueue("error", error)
        finally:
            runtime_logger.enqueue("finished", None)

    worker = threading.Thread(target=produce_events, daemon=True)
    worker.start()
    try:
        while True:
            event_kind, event_payload = await runtime_logger.events.get()
            if event_kind == "runtime":
                runtime_event = {
                    "type": "pipeline.runtime_event",
                    "operation_id": operation_id,
                    "response_id": response_id,
                    "payload": event_payload,
                }
                yield encode_sse("pipeline.runtime_event", runtime_event)
                continue
            if event_kind == "error":
                raise event_payload
            if event_kind == "finished":
                break
            if event_kind != "execution":
                continue

            item = event_payload
            if isinstance(item, str):
                if not item:
                    continue
                emitted_delta = True
                yield encode_sse(
                    "runtime.output_text.delta",
                    {
                        "type": "runtime.output_text.delta",
                        "operation_id": operation_id,
                        "response_id": response_id,
                        "payload": {"delta": item},
                    },
                )
            elif isinstance(item, FinalResponse):
                result = item
            elif isinstance(item, UserInputRequired):
                yield encode_sse(
                    "runtime.requires_user_input",
                    {
                        "type": "runtime.requires_user_input",
                        "operation_id": operation_id,
                        "response_id": response_id,
                        "payload": {
                            "question": item.question,
                            "reason": item.reason,
                            "options": [asdict(option) for option in item.options],
                            "safeguard_assessment": (
                                asdict(item.safeguard_assessment)
                                if item.safeguard_assessment is not None
                                else None
                            ),
                            "continuation_state": item.continuation_state,
                        },
                    },
                )
                return

        if result is None:
            raise RuntimeError("Runtime stream returned no completed response.")
        if not emitted_delta:
            for delta in chunk_text(result.answer):
                yield encode_sse(
                    "runtime.output_text.delta",
                    {
                        "type": "runtime.output_text.delta",
                        "operation_id": operation_id,
                        "response_id": response_id,
                        "payload": {"delta": delta},
                    },
                )
        yield encode_sse(
            "runtime.completed",
            {
                "type": "runtime.completed",
                "operation_id": operation_id,
                "response_id": response_id,
                "payload": _completed_runtime_payload(result),
            },
        )
    finally:
        stop_worker.set()


def _authorize_service(
    settings: ApiSettings,
    authorization: str | None,
    consumer_service: str | None,
) -> None:
    if settings.runtime_service_token is None:
        raise HTTPException(
            status_code=503,
            detail="Runtime service authentication is not configured.",
        )
    expected = f"Bearer {settings.runtime_service_token}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=401, detail="Runtime service authentication failed."
        )
    if consumer_service != settings.runtime_consumer_service:
        raise HTTPException(
            status_code=403, detail="Runtime consumer service is not allowed."
        )


def _error_response(
    request: OperationEnvelope,
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
) -> JSONResponse:
    error = RuntimeErrorResponse(
        code=code,
        message=message,
        retryable=retryable,
        operation_id=request.operation_id,
        trace_id=request.trace_id,
    )
    return JSONResponse(status_code=status_code, content=error.model_dump(mode="json"))


def create_runtime_operations_router(
    *,
    settings: ApiSettings,
    pipeline_factory: PipelineFactory,
) -> APIRouter:
    router = APIRouter(tags=["runtime-operations"])

    @router.post("/v1/specs:prepare")
    async def prepare_runtime_spec(
        request: PrepareSpecRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
        consumer_service: str | None = Header(default=None, alias="X-Consumer-Service"),
    ):
        _authorize_service(settings, authorization, consumer_service)
        try:
            return prepare_spec(
                request,
                settings=settings,
                pipeline_factory=pipeline_factory,
            )
        except ValueError:
            return _error_response(
                request,
                status_code=422,
                code="validation_error",
                message="The runtime operation payload is invalid.",
                retryable=False,
            )
        except Exception:
            return _error_response(
                request,
                status_code=500,
                code="internal_error",
                message="The runtime could not prepare the execution spec.",
                retryable=False,
            )

    @router.post("/v1/specs:revise")
    async def revise_runtime_spec(
        request: ReviseSpecRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
        consumer_service: str | None = Header(default=None, alias="X-Consumer-Service"),
    ):
        _authorize_service(settings, authorization, consumer_service)
        try:
            return revise_spec(request, pipeline_factory=pipeline_factory)
        except ValueError:
            return _error_response(
                request,
                status_code=422,
                code="validation_error",
                message="The revised execution spec is invalid.",
                retryable=False,
            )
        except Exception:
            return _error_response(
                request,
                status_code=500,
                code="internal_error",
                message="The runtime could not revise the execution spec.",
                retryable=False,
            )

    @router.post("/v1/execution:thinking")
    async def stream_runtime_execution(
        request: ThinkingExecutionRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
        consumer_service: str | None = Header(default=None, alias="X-Consumer-Service"),
        user_authorization: str | None = Header(
            default=None,
            alias="X-Axiom-User-Authorization",
        ),
    ) -> StreamingResponse:
        _authorize_service(settings, authorization, consumer_service)

        async def event_stream() -> AsyncIterator[str]:
            selection = None
            runtime_logger = _StreamingRuntimeLogger()
            try:
                if pipeline_factory is default_pipeline_factory:
                    selection = select_thinking_engine(
                        request,
                        settings=settings,
                        pipeline_factory=pipeline_factory,
                    )
                    yield encode_sse(
                        "runtime.engine.selected",
                        {
                            "type": "runtime.engine.selected",
                            "operation_id": request.operation_id,
                            "response_id": request.response_id,
                            "payload": {
                                "engine_name": selection.engine.name,
                                "selection_source": selection.selection_source,
                            },
                        },
                    )
                    if selection.engine.name == "report":
                        async for report_event in stream_report_events(
                            request,
                            instruction=request.spec_markdown,
                            settings=settings,
                            user_authorization=user_authorization,
                        ):
                            yield encode_sse(report_event["type"], report_event)
                        return
                async for runtime_event in _stream_execution_events(
                    stream_thinking(
                        request,
                        settings=settings,
                        pipeline_factory=pipeline_factory,
                        selection=selection,
                        user_authorization=user_authorization,
                        logger=runtime_logger,
                    ),
                    operation_id=request.operation_id,
                    response_id=request.response_id,
                    runtime_logger=runtime_logger,
                ):
                    yield runtime_event
            except Exception as exc:
                logger.exception(
                    "Runtime execution failed operation_id=%s response_id=%s",
                    request.operation_id,
                    request.response_id,
                )
                yield encode_sse(
                    "runtime.failed",
                    {
                        "type": "runtime.failed",
                        "operation_id": request.operation_id,
                        "response_id": request.response_id,
                        "payload": {
                            "code": (
                                "engine_selection_failed"
                                if isinstance(exc, EngineSelectionError)
                                else "execution_failed"
                            ),
                            "message": (
                                "The runtime could not select an engine."
                                if isinstance(exc, EngineSelectionError)
                                else "The runtime execution failed."
                            ),
                            "retryable": False,
                        },
                    },
                )
                return

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/v1/execution:instant")
    async def stream_direct_runtime_execution(
        request: InstantExecutionRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
        consumer_service: str | None = Header(default=None, alias="X-Consumer-Service"),
        user_authorization: str | None = Header(
            default=None,
            alias="X-Axiom-User-Authorization",
        ),
    ) -> StreamingResponse:
        _authorize_service(settings, authorization, consumer_service)

        async def event_stream() -> AsyncIterator[str]:
            selection = None
            runtime_logger = _StreamingRuntimeLogger()
            if pipeline_factory is default_pipeline_factory:
                try:
                    selection = select_instant_engine(
                        request,
                        settings=settings,
                        pipeline_factory=pipeline_factory,
                    )
                    yield encode_sse(
                        "runtime.engine.selected",
                        {
                            "type": "runtime.engine.selected",
                            "operation_id": request.operation_id,
                            "response_id": request.response_id,
                            "payload": {
                                "engine_name": selection.engine.name,
                                "selection_source": selection.selection_source,
                            },
                        },
                    )
                    if selection.engine.name != "report":
                        async for runtime_event in _stream_execution_events(
                            stream_instant(
                                request,
                                settings=settings,
                                pipeline_factory=pipeline_factory,
                                selection=selection,
                                user_authorization=user_authorization,
                                logger=runtime_logger,
                            ),
                            operation_id=request.operation_id,
                            response_id=request.response_id,
                            runtime_logger=runtime_logger,
                        ):
                            yield runtime_event
                    else:
                        async for report_event in stream_report_events(
                            request,
                            instruction=request.runtime_input.input,
                            settings=settings,
                            user_authorization=user_authorization,
                        ):
                            yield encode_sse(report_event["type"], report_event)
                except Exception as exc:
                    logger.exception(
                        "Runtime Instant execution failed operation_id=%s response_id=%s",
                        request.operation_id,
                        request.response_id,
                    )
                    yield encode_sse(
                        "runtime.failed",
                        {
                            "type": "runtime.failed",
                            "operation_id": request.operation_id,
                            "response_id": request.response_id,
                            "payload": {
                                "code": (
                                    "engine_selection_failed"
                                    if isinstance(exc, EngineSelectionError)
                                    else "execution_failed"
                                ),
                                "message": (
                                    "The runtime could not select an engine."
                                    if isinstance(exc, EngineSelectionError)
                                    else "The runtime execution failed."
                                ),
                                "retryable": False,
                            },
                        },
                    )
                return
            try:
                async for runtime_event in _stream_execution_events(
                    stream_instant(
                        request,
                        settings=settings,
                        pipeline_factory=pipeline_factory,
                        user_authorization=user_authorization,
                        logger=runtime_logger,
                    ),
                    operation_id=request.operation_id,
                    response_id=request.response_id,
                    runtime_logger=runtime_logger,
                ):
                    yield runtime_event
            except Exception as exc:
                logger.exception(
                    "Runtime direct execution failed operation_id=%s response_id=%s",
                    request.operation_id,
                    request.response_id,
                )
                yield encode_sse(
                    "runtime.failed",
                    {
                        "type": "runtime.failed",
                        "operation_id": request.operation_id,
                        "response_id": request.response_id,
                        "payload": {
                            "code": (
                                "engine_selection_failed"
                                if isinstance(exc, EngineSelectionError)
                                else "execution_failed"
                            ),
                            "message": (
                                "The runtime could not select an engine."
                                if isinstance(exc, EngineSelectionError)
                                else "The runtime execution failed."
                            ),
                            "retryable": False,
                        },
                    },
                )
                return

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/v1/execution:resume")
    async def stream_runtime_resume(
        request: ResumeExecutionRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
        consumer_service: str | None = Header(default=None, alias="X-Consumer-Service"),
        user_authorization: str | None = Header(
            default=None,
            alias="X-Axiom-User-Authorization",
        ),
    ) -> StreamingResponse:
        _authorize_service(settings, authorization, consumer_service)

        async def event_stream() -> AsyncIterator[str]:
            runtime_logger = _StreamingRuntimeLogger()
            try:
                async for runtime_event in _stream_execution_events(
                    resume_execution(
                        request,
                        settings=settings,
                        pipeline_factory=pipeline_factory,
                        user_authorization=user_authorization,
                        logger=runtime_logger,
                    ),
                    operation_id=request.operation_id,
                    response_id=request.response_id,
                    runtime_logger=runtime_logger,
                ):
                    yield runtime_event
            except Exception:
                logger.exception(
                    "Runtime resume failed operation_id=%s response_id=%s",
                    request.operation_id,
                    request.response_id,
                )
                yield encode_sse(
                    "runtime.failed",
                    {
                        "type": "runtime.failed",
                        "operation_id": request.operation_id,
                        "response_id": request.response_id,
                        "payload": {
                            "code": "resume_failed",
                            "message": "The runtime could not resume this response.",
                            "retryable": True,
                        },
                    },
                )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
