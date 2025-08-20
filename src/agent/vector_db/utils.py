from llama_index.core import VectorStoreIndex

from agent.vector_db.vectorstore import get_vectorstore


def retrieve_top_k(query, k=5):
    vector_store = get_vectorstore()
    retriever = VectorStoreIndex.from_vector_store(vector_store).as_retriever(similarity_top_k=k)
    results = []
    keywords = []
    for obj in retriever.retrieve(query):
        results.append(obj.node.metadata.get("content_id"))
        keywords.append(obj.node.metadata.get("keywords"))

    return results, keywords
