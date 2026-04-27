from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def embed_documents(embedder, documents: List, batch_size=32):
    texts = [doc.page_content for doc in documents]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings.extend(embedder.embed_documents(batch))

    return embeddings


def embed_query(embedder, query: str):
    return embedder.embed_query(query)