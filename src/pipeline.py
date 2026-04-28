import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import logging
from pathlib import Path

from src.loader import load_documents, chunk_documents
from src.embedder import get_embedder
from src.retriever import create_vectorstore, get_retriever
from src.generator import get_llm, create_rag_prompt, create_qa_chain, generate_response
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(self, data_dir="data/", persist_dir="vectorstore"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir

        self.embedder = get_embedder()
        self.llm = get_llm()

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_and_index(self):

        if Path(self.persist_dir).exists():
            logger.info("Loading existing vectorstore...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedder
            )
        else:
            docs = load_documents(self.data_dir)
            chunks = chunk_documents(docs)

            self.vectorstore = create_vectorstore(
                chunks,
                self.embedder,
                persist_dir=self.persist_dir
            )

        self.retriever = get_retriever(self.vectorstore)

        prompt = create_rag_prompt()
        self.qa_chain = create_qa_chain(self.llm, self.retriever, prompt)

    def query(self, question: str):
        return generate_response(self.qa_chain, question)