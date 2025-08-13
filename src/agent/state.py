import operator
from typing import Annotated, List

from typing_extensions import TypedDict


def merge_dicts(a: dict, b: dict) -> dict:
    if a is None:
        a = {}

    if b is None:
        return a

    return {**a, **b}


class AgentState(TypedDict):
    user_query: str
    response: str
    expanded_query: str
    reranked_ids: List[str]
    extracted_documents: Annotated[list, operator.add]
    metrics: Annotated[dict, merge_dicts]


class InputState(TypedDict):
    user_query: str


class ContentExtractionState(TypedDict):
    content_id: str


class OutputState(TypedDict):
    response: str
    metrics: dict
