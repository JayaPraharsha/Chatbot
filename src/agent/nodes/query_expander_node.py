from agent.db.access import get_summary_by_id
from agent.llms import query_expander_llm
from agent.prompts.query_expansion import query_expansion_prompt
from agent.schemas import ExpansionSchema
from agent.vector_db.vectorstore import get_vectorstore

query_expander_llm = query_expander_llm.with_structured_output(ExpansionSchema)
query_expansion_chain = query_expansion_prompt | query_expander_llm


def retrieve_and_expand_query(query: str):
    vectorstore = get_vectorstore()
    retrieved_docs = vectorstore.similarity_search_with_relevance_scores(query, k=10)

    keywords_map = {
        f"Document_{i + 1}_Keywords": doc.metadata.get("keywords")
        for i, (doc, score) in enumerate(retrieved_docs)
    }
    retrieved_article_ids = [doc.metadata.get("content_id") for (doc, score) in retrieved_docs]

    retrieved_summaries = [get_summary_by_id(article_id) for article_id in retrieved_article_ids]
    summary_map = {
        f"Document_{i + 1}_Summary": summary for i, summary in enumerate(retrieved_summaries)
    }
    summary_map["query"] = query
    args_map = {**summary_map, **keywords_map}
    expansion_result = query_expansion_chain.invoke(args_map).expanded_query

    return expansion_result
