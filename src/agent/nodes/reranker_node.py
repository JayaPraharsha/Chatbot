from typing import List

from db.access import get_summary_by_id
from llms import reranker_llm
from prompts.reranking import reranker_prompt
from schemas import RerankerSchema
from vector_db import get_vectorstore

reranker_llm = reranker_llm.with_structured_output(RerankerSchema)
reranker_chain = reranker_prompt | reranker_llm


def retrieve_and_rerank(expanded_query: str) -> List:
    vectorstore = get_vectorstore()
    retrieved_docs = vectorstore.similarity_search_with_relevance_scores(expanded_query, k=10)
    results = []
    for doc, score in retrieved_docs:
        results.append(doc.metadata.get("content_id"))

    summaries = [get_summary_by_id(id) for id in results]
    args_map = {}
    args_map["query"] = expanded_query
    for i, summary in enumerate(summaries):
        args_map[f"Document_{i + 1}"] = summary

    response = reranker_chain.invoke(args_map)
    ranks = response.ranks

    import ast

    ranks = ast.literal_eval(ranks)

    llm_ranks_map = {}
    for content_id, rank in zip(results, ranks):
        llm_ranks_map[content_id] = rank

    llm_ranks_sorted = sorted(llm_ranks_map.items(), key=lambda item: item[1])
    top_n = 5
    llm_reranked_ids = []
    for content_id, rank in llm_ranks_sorted[:top_n]:
        llm_reranked_ids.append(content_id)

    return llm_reranked_ids
