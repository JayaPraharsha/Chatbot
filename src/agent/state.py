from typing import List, TypedDict

from langchain.schema import Document


class InputState(TypedDict):
    user_query: str


class PreRetrievalNodeOutput(TypedDict):
    expanded_query: str


class RetrievalNodeOutput(TypedDict):
    retrieved_documents: List[Document]


class PostRetrievalNodeOutput(TypedDict):
    reranked_documents: List[Document]
    compressed_context: str


class OverallState(TypedDict):
    final_response: str
