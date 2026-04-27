import os
from langchain_community.llms import Ollama
from typing import Dict
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def get_llm():
    return Ollama(
        model="llama3",
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        temperature=0.2
    )

def create_rag_prompt():
    template = """You are an academic assistant.

Use ONLY the context below.

If answer not found, say:
"I could not find this in the provided documents."

Context:
{context}

Question: {question}

Answer (with sources):"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def create_qa_chain(llm, retriever, prompt=None):
    if prompt is None:
        prompt = create_rag_prompt()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # fixed
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt} # fixed
    )

def generate_response(qa_chain, query: str) -> Dict:
    result = qa_chain.invoke({"query": query})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content[:150]
        })

    return {
        "answer": result.get("result", ""),
        "sources": sources
    }