from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

import weaviate
from agent.vector_db import config

_vector_store = None


def get_vectorstore():
    global _vector_store
    if _vector_store is None:
        client = weaviate.connect_to_local()
        _vector_store = WeaviateVectorStore(
            client=client,
            index_name=config.index_name,
            text_key=config.text_key,
            embedding=OpenAIEmbeddings(),
            attributes=config.attributes,
        )
    return _vector_store
