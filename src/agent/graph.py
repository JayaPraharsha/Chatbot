from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from nodes.content_extraction_node import extract_content
from nodes.generation_node import generate_answer
from nodes.query_expander_node import retrieve_and_expand_query
from nodes.reranker_node import retrieve_and_rerank

from agent.state import (
    ContentExtractorOutput,
    InputState,
    OutputState,
    RetrieveExpandOutput,
    RetrieveRerankOutput,
)


def retrieve_and_expand_node(state: InputState) -> RetrieveExpandOutput:
    query = state["user_query"]
    expanded_query = retrieve_and_expand_query(query)
    return {"expanded_query": expanded_query}


def retrieve_and_rerank_node(state: RetrieveExpandOutput) -> RetrieveRerankOutput:
    expanded_query = state["expanded_query"]
    reranked_ids = retrieve_and_rerank(expanded_query)

    return [
        Send("extract_content_node", {"content_id": content_id, "query": expanded_query})
        for content_id in reranked_ids
    ]


def extract_content_node(state: RetrieveRerankOutput) -> ContentExtractorOutput:
    content_id = state["content_id"]
    expanded_query = state["expanded_query"]
    return {"extracted_documents": extract_content(content_id, expanded_query)}


def generate_answer_node(state: ContentExtractorOutput) -> OutputState:
    args = {}
    args["Question"] = state["expanded_query"]

    for i, sentences in enumerate(state["extracted_documents"]):
        args[f"Document_{i}"] = sentences

    return {"response": generate_answer(args)}


def evaluation_node(state: OutputState) -> OutputState:
    pass


def create_research_graph():
    graph = StateGraph(input=InputState, output=OutputState)
    graph.add_node("retrieve_and_expand", retrieve_and_expand_node)
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.add_node("extract_content", extract_content_node)
    graph.add_node("generate_answer", generate_answer_node)
    # graph.add_node("evaluation", evaluation_node)

    graph.add_edge(START, "retrieve_and_expand")
    graph.add_edge("retrieve_and_expand", "retrieve_and_rerank")
    graph.add_edge("retrieve_and_rerank", "extract_content")
    graph.add_edge("extract_content", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph


def create_compiled_graph():
    graph = create_research_graph()
    return graph.compile
