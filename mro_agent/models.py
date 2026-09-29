from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Profile(StrictModel):
    object: str = Field(default="", max_length=500)
    task: str = Field(default="", max_length=1000)
    expected_result: str = Field(default="", max_length=1000)
    identifiers: list[str] = Field(default_factory=list, max_length=20)
    questions: list[str] = Field(default_factory=list, max_length=3)


class Claim(StrictModel):
    text: str = Field(max_length=1000)
    evidence_id: str = Field(max_length=100)
    quote: str = Field(min_length=1, max_length=1500)
    category: Literal["work", "deliverable", "difference", "context"] = "context"


class Extraction(StrictModel):
    claims: list[Claim] = Field(default_factory=list, max_length=12)


class Verdict(StrictModel):
    supported: bool


class ScopeItem(StrictModel):
    text: str = Field(max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)


class Synthesis(StrictModel):
    proposed_scope: list[ScopeItem] = Field(default_factory=list, max_length=10)
    questions: list[str] = Field(default_factory=list, max_length=5)


class Followup(StrictModel):
    kind: Literal["question", "correction"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: Any = ""


class ChatRequest(StrictModel):
    model: Literal["mro-assessment-agent"] = "mro-assessment-agent"
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    chat_id: str = Field(min_length=1, max_length=200)
    message_id: str = Field(min_length=1, max_length=200)
    stream: bool = True
    attachments_present: bool = False


class AssessmentRequest(StrictModel):
    request: str = Field(min_length=1, max_length=16000)
    chat_id: str = Field(min_length=1, max_length=200)
    message_id: str = Field(min_length=1, max_length=200)
    attachments_present: bool = False

