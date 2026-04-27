from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(data_dir: str = "data/") -> List:
    path = Path(data_dir)
    documents = []

    # TXT
    for file_path in path.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["type"] = "text"
        documents.extend(docs)

    # PDF
    for pdf_path in path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = pdf_path.name
            doc.metadata["page"] = i + 1
            doc.metadata["type"] = "pdf"
        documents.extend(docs)

    return documents


def chunk_documents(documents: List, chunk_size=600, chunk_overlap=100) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks