from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from agent.db.db_config import init_db
from agent.nodes.content_extraction_node import extract_content
from agent.nodes.evaluation_node import compute_metrics
from agent.nodes.generation_node import generate_answer
from agent.nodes.query_expander_node import retrieve_and_expand_query
from agent.nodes.reranker_node import retrieve_and_rerank
from agent.state import AgentState, ContentExtractionState, InputState, OutputState
from agent.vector_db.vectorstore import close_client


def retrieve_and_expand_node(state: InputState) -> AgentState:
    init_db()
    query = state["user_query"]
    expanded_query = retrieve_and_expand_query(query)
    return {"expanded_query": expanded_query}


def retrieve_and_rerank_node(state: AgentState) -> AgentState:
    exp_query = state["expanded_query"]
    reranked_ids = retrieve_and_rerank(exp_query)
    close_client()
    return {"reranked_ids": reranked_ids}


def extract_content_func(state: AgentState):
    return [
        Send(
            "extract_content_node",
            {"content_id": content_id, "expanded_query": state["expanded_query"]},
        )
        for content_id in state["reranked_ids"]
    ]


def extract_content_node(state: ContentExtractionState) -> AgentState:
    content_id = state["content_id"]
    expanded_query = state["expanded_query"]
    return {"extracted_documents": [extract_content(content_id, expanded_query)]}


def generate_answer_node(state: AgentState) -> AgentState:
    args = {}
    args["Question"] = state["expanded_query"]

    for i, doc in enumerate(state["extracted_documents"]):
        args[f"Document_{i + 1}"] = doc

    return {"response": generate_answer(args)}


def evaluation_calls_func(state: AgentState):
    return [
        Send("evaluation", {**state, "metric": metric})
        for metric in ["F1", "Bleu", "Rouge", "Factuality", "Context Recall"]
    ]


def evaluation_node(state: AgentState) -> OutputState:
    metric = state["metric"]
    expanded_question = state["expanded_query"]
    initial_user_query = state["user_query"]
    retreived_docs = state["extracted_documents"]
    response = state["response"]
    return {
        "metrics": compute_metrics(
            metric, expanded_question, initial_user_query, response, retreived_docs
        )
    }


def create_research_graph():
    graph = StateGraph(AgentState, input_schema=InputState, output_schema=OutputState)
    graph.add_node("retrieve_and_expand", retrieve_and_expand_node)
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.add_node("extract_content_node", extract_content_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("evaluation", evaluation_node)

    graph.add_edge(START, "retrieve_and_expand")
    graph.add_edge("retrieve_and_expand", "retrieve_and_rerank")
    graph.add_conditional_edges(
        "retrieve_and_rerank", extract_content_func, ["extract_content_node"]
    )
    graph.add_edge("extract_content_node", "generate_answer")
    graph.add_conditional_edges("generate_answer", evaluation_calls_func, ["evaluation"])
    graph.add_edge("evaluation", END)

    return graph


def create_compiled_graph():
    graph = create_research_graph()
    return graph.compile()


def answer_query(query: str):
    graph = create_compiled_graph()
    return graph.invoke({"user_query": query})
