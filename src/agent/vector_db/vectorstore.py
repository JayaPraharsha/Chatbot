import config
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate

import weaviate

_vector_store = None


def get_vectorstore():
    global _vector_store
    if _vector_store is None:
        client = weaviate.connect_to_local()
        _vector_store = Weaviate(
            client=client,
            index_name=config.index_name,
            text_key=config.text_key,
            embedding=OpenAIEmbeddings(),
            attributes=config.attributes,
        )
    return _vector_store
