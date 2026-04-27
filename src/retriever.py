from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def create_vectorstore(documents, embedder, persist_dir="vectorstore"):

    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedder
        )
        if vectorstore._collection.count() > 0:
            return vectorstore
    except:
        pass

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore


def get_retriever(vectorstore, k=5):

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10
        }
    )


def get_compression_retriever(llm, base_retriever):
    compressor = LLMChainExtractor.from_llm(llm)

    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )