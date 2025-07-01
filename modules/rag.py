

import logging
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from modules.logger import logger
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))


logger = logging.getLogger("rag")

load_dotenv()


class RagPipeline:
    def __init__(self, doc_data, doc_name="input_pdf"):
        self.doc_data = doc_data
        self.doc_name = doc_name
        self.vectorstore = None

    # def _check_if_index_exists(self):
    #     if os.path.exists("faiss_index"):
    #         print("Index already exists. Loading from disk.")
    #         self.vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))
    #     else:
    #         print("Index does not exist. Building index.")
    #         self.build_index()

    def build_index(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        all_documents = []
        for k, v in self.doc_data.items():
            chunks = text_splitter.split_text(v)
            for chunk in chunks:
                all_documents.append(Document(page_content=chunk, metadata={
                                     "page": k, "source": self.doc_name}))

        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore = FAISS.from_documents(all_documents, embeddings)
        self.vectorstore.save_local(
            "vector_data/" + self.doc_name + "_faiss_index")


class QueryRAG:
    def __init__(self, doc_name="input_pdf"):
        self.doc_name = doc_name

        if not os.path.exists("vector_data/" + self.doc_name + "_faiss_index"):
            raise FileNotFoundError(
                f"Vector store for {self.doc_name} does not exist. Please build the index first.")

        logger.info(f"Loading vector store for {self.doc_name} from disk.")
        self.vectorstore = FAISS.load_local("vector_data/" + self.doc_name + "_faiss_index", OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")), allow_dangerous_deserialization=True)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1000,
            api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=60
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )

    def query(self, question):
        result = self.qa_chain.invoke(question)
        unique_documents = {
            doc.metadata["source"]: doc for doc in result["source_documents"]}.values()
        return result["result"], unique_documents
