from typing import List

from pydantic import BaseModel, Field


class RerankerSchema(BaseModel):
    ranks: str = Field(
        description=" A string, which looks like a python list of integers, containing the indexes of the documents sorted in descending order of their relevance.",
        examples=["[1, 2, 3, 4, 5,]"],
    )
    justification: list[str] = Field(
        description="Concise statments that provide justification for the given ordered ranks of documents."
    )


class ExpansionSchema(BaseModel):
    expanded_query: str = Field(
        description=(
            "A improved version of the user's original query, "
            "with added relevant keywords, targeted for better keyword and semantic search."
        )
    )
    justification: str = Field(
        description=(
            "A concise rationale explaining why the expanded query differs from the original, "
            "highlighting key terms, context, or intent that informed the expansion."
        )
    )


class SentenceSpan(BaseModel):
    start_idx: int = Field(description="Start index of relevant sentence span (inclusive).")
    end_idx: int = Field(description="End index of relevant sentence span (inclusive).")


class ContentExtractionSchema(BaseModel):
    extracted_contents: List[SentenceSpan] = Field(
        min_items=1,
        description=(
            "A list of extracted sentence ranges or individual sentences from the content.\n"
            "- Each item is either:\n"
            "  • A tuple of two integers (start_idx, end_idx), representing a passage of consecutive relevant sentences from start to end (inclusive),\n"
            "  • Or a tuple where start_idx == end_idx, indicating a single relevant sentence.\n"
            "Sentence indices are 1-based and refer to the position of sentences in the original text."
            "Sentence i is immediately followed by sentence i+1."
        ),
    )
