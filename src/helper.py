from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import requests
import os
import logging

logger = logging.getLogger("medical-chatbot.helper")

class HuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {}
        # Support optional HF_TOKEN for higher rate limits in production
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"
        self._local_embeddings = None

    def _get_local_embeddings(self):
        if self._local_embeddings is None:
            logger.info("Initializing local HuggingFaceEmbeddings as fallback...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._local_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._local_embeddings

    def _query_api(self, inputs):
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": inputs, "options": {"wait_for_model": True}},
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Hugging Face Inference API returned status {response.status_code}: {response.text}")
        except Exception as e:
            logger.warning(f"Failed to query Hugging Face Inference API: {e}")
        return None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Attempt to use API
        res = self._query_api(texts)
        if res is not None and isinstance(res, list):
            return res
        # Fallback to local execution
        return self._get_local_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        # Attempt to use API
        res = self._query_api(text)
        if res is not None:
            if isinstance(res, list) and len(res) > 0:
                if isinstance(res[0], list):
                    return res[0]
                return res
        # Fallback to local execution
        return self._get_local_embeddings().embed_query(text)

#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

#Download the Embeddings 
def download_hugging_face_embeddings():
    logger.info("Setting up HuggingFaceInferenceEmbeddings wrapper...")
    return HuggingFaceInferenceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')