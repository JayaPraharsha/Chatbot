import operator
from typing import Annotated, List, TypedDict

from schemas import SentenceSpan


class InputState(TypedDict):
    user_query: str


class RetrieveExpandOutput(TypedDict):
    expanded_query: str


class RetrieveRerankOutput(TypedDict):
    reranked_ids: List[str]


class ContentExtractorOutput(TypedDict):
    extracted_documents: Annotated[List[SentenceSpan], operator.add]


class OutputState(TypedDict):
    response: str


__all__ = [
    "InputState",
    "OutputState",
    "ContentExtractorOutput",
    "RetrieveRerankOutput",
    "RetrieveExpandOutput",
]
