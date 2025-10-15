from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_serializer, model_validator

DocSource = Literal["yandex", "internal"]
DocType = Literal["law", "gov_letter", "court", "article", "faq", "news", "forum", "internal"]


class LawRef(BaseModel):
    code: str
    article: str | None = None
    span: str | None = None


class UnifiedDoc(BaseModel):
    title: str = Field(..., min_length=2)
    snippet: str | None = None
    content: str | None = None
    url: HttpUrl
    source: DocSource
    doc_type: DocType = "article"
    score_raw: float = 0.0
    score_rank: float = 0.0
    law_refs: list[LawRef] = []
    # published_at: Optional[datetime] = None  # зарезервировано, пока не используем
    hash: str = ""

    @field_serializer("url")
    def serialize_url(self, v: HttpUrl) -> str:
        return str(v)

    @model_validator(mode="before")
    @classmethod
    def compute_hash(cls, data):
        import hashlib

        if isinstance(data, dict):
            if data.get("hash"):
                return data

            title = data.get("title", "")
            url = str(data.get("url", ""))
            base = (title + "|" + url).lower()
            data["hash"] = hashlib.md5(base.encode("utf-8")).hexdigest()

        return data


class SearchResults(BaseModel):
    docs: list[UnifiedDoc] = []
    meta: dict[str, str] = {}


class QualityReport(BaseModel):
    good: bool
    reasons: list[str] = []
    missing: list[str] = []


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = []
    raw_user_query: str = ""
    yandex_results: SearchResults = SearchResults()
    internal_results: SearchResults = SearchResults()
    merged_results: SearchResults = SearchResults()
    final_answer: str | None = None
    decision: Literal["CHAT", "SEARCH", "TOOL_CALL", "FINAL"] | None = None
    diagnostics: dict[str, str] = {}

    class Config:
        arbitrary_types_allowed = True
