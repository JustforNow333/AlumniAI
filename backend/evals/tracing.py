from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CALL_TYPE_INTENT = "intent_inference"
CALL_TYPE_CLASSIFIER = "llm_classifier"
CALL_TYPE_FINAL_SYNTHESIS = "final_model_synthesis"
CALL_TYPE_UNKNOWN = "unknown"


@dataclass
class ModelCallTrace:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, model: str | None, instructions: Any) -> None:
        self.calls.append(
            {
                "model": model,
                "call_type": classify_model_call(instructions),
            }
        )

    @property
    def total_model_calls(self) -> int:
        return len(self.calls)

    @property
    def llm_classifier_calls(self) -> int:
        return sum(1 for call in self.calls if call.get("call_type") == CALL_TYPE_CLASSIFIER)

    @property
    def final_model_synthesis_calls(self) -> int:
        return sum(1 for call in self.calls if call.get("call_type") == CALL_TYPE_FINAL_SYNTHESIS)

    @property
    def used_llm_classifier(self) -> bool:
        return self.llm_classifier_calls > 0

    @property
    def used_final_model_synthesis(self) -> bool:
        return self.final_model_synthesis_calls > 0

    @property
    def model_name(self) -> str | None:
        for call in reversed(self.calls):
            if call.get("model"):
                return str(call["model"])
        return None

    def as_case_fields(self) -> dict[str, Any]:
        return {
            "total_model_calls": self.total_model_calls,
            "used_llm_classifier": self.used_llm_classifier,
            "llm_classifier_calls": self.llm_classifier_calls,
            "used_final_model_synthesis": self.used_final_model_synthesis,
            "final_model_synthesis_calls": self.final_model_synthesis_calls,
            "model_name": self.model_name,
            "model_call_types": [call.get("call_type") for call in self.calls],
        }


class TracedAIClient:
    def __init__(self, client: Any, trace: ModelCallTrace):
        self._client = client
        self._trace = trace
        self.responses = _TracedResponses(client.responses, trace)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _TracedResponses:
    def __init__(self, responses: Any, trace: ModelCallTrace):
        self._responses = responses
        self._trace = trace

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._trace.record(
            model=kwargs.get("model"),
            instructions=kwargs.get("instructions"),
        )
        return self._responses.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)


def classify_model_call(instructions: Any) -> str:
    text = str(instructions or "")
    if "You classify whether an employer belongs to a target industry" in text:
        return CALL_TYPE_CLASSIFIER
    if "You present spreadsheet analysis results" in text:
        return CALL_TYPE_FINAL_SYNTHESIS
    if "You infer spreadsheet analysis intent" in text:
        return CALL_TYPE_INTENT
    return CALL_TYPE_UNKNOWN
