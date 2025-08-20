import os

from llama_index.vector_stores.weaviate import WeaviateVectorStore

import weaviate
from agent.vector_db import config
from weaviate.classes.init import Auth

_client = None
_vector_store = None


def _make_client() -> weaviate.WeaviateClient:
    """
    Create & return a weaviate v4 client.
    """
    global _client
    if _client is not None:
        return _client

    cloud_url = os.getenv("WEAVIATE_CLOUD_URL")
    cloud_api_key = os.getenv("WEAVIATE_CLOUD_API_KEY")
    if cloud_url and cloud_api_key:
        auth = Auth.api_key(cloud_api_key)
        _client = weaviate.connect_to_weaviate_cloud(cluster_url=cloud_url, auth_credentials=auth)
    # else:
    #     _client = weaviate.connect_to_local()

    if not _client.is_ready():
        raise RuntimeError("Weaviate client not ready")

    return _client


def close_client() -> None:
    """
    Close the global client and clear cached vectorstore.
    """
    global _client, _vector_store
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
    _vector_store = None


def get_vectorstore():
    """
    Return a LlamaIndex WeaviateVectorStore.
    Lazily initializes client before returning.
    """
    index_name: str = os.getenv("WEAVIATE_CLOUD_INDEX_NAME")
    text_key: str = config.text_key
    # attributes = config.attributes

    global _vector_store, _client
    if _vector_store is not None:
        return _vector_store

    client = _make_client()

    _vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=index_name,
        text_key=text_key,
    )
    return _vector_store
