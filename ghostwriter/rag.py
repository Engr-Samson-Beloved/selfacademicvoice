import random

from . import config

_client = None
_collection = None
_all_docs = None
_embedder = None


def get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb

        _client = chromadb.PersistentClient(path=str(config.VECTOR_DB_DIR))
        _collection = _client.get_collection(config.COLLECTION_NAME)
    return _collection


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def get_all_documents():
    global _all_docs
    if _all_docs is None:
        _all_docs = get_collection().get()["documents"]
    return _all_docs


def retrieve_random_examples(n=3):
    docs = get_all_documents()
    return random.sample(docs, min(n, len(docs)))


def retrieve_similar(query, n=3):
    embedding = get_embedder().encode(query).tolist()
    results = get_collection().query(query_embeddings=[embedding], n_results=n)
    return results["documents"][0]
