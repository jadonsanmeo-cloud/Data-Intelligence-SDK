"""Structured user-clarification tool for the general-purpose agent."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data_intelligence_sdk.core.types import UserInputOption


UserInputReason = Literal["ambiguous_query", "method_definition"]


class AskUserToolOptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=1000)
    source: str | None = Field(default=None, max_length=500)

    @field_validator("id", "label", "description", "source")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class AskUserToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    reason: UserInputReason
    options: list[AskUserToolOptionInput] = Field(min_length=1, max_length=3)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def require_unique_option_ids(self) -> AskUserToolInput:
        option_ids = [option.id for option in self.options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("option IDs must be unique")
        return self

    def user_input_options(self) -> list[UserInputOption]:
        return [
            UserInputOption(
                id=option.id,
                label=option.label,
                description=option.description,
                source=option.source,
            )
            for option in self.options
        ]


def create_ask_user_tool() -> BaseTool:
    @tool(args_schema=AskUserToolInput)
    def ask_user(
        question: str,
        reason: UserInputReason,
        options: list[AskUserToolOptionInput],
    ) -> str:
        """Pause the analysis and ask the user to choose one interpretation."""

        request = AskUserToolInput(
            question=question,
            reason=reason,
            options=options,
        )
        return json.dumps(
            {
                "type": "axiom.ask_user.v1",
                "question": request.question,
                "reason": request.reason,
                "options": [option.model_dump(mode="json") for option in request.options],
            },
            ensure_ascii=False,
        )

    return ask_user


def parse_ask_user_result(content: Any) -> AskUserToolInput | None:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get("type") != "axiom.ask_user.v1":
            return None
        return AskUserToolInput.model_validate(
            {key: value for key, value in payload.items() if key != "type"}
        )
    except (json.JSONDecodeError, ValueError):
        return None
