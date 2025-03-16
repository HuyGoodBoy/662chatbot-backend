GENERATE_MODEL_NAME="ft:gpt-4o-mini-2024-07-18:personal:dpl-mini:BAy9Of3X"
EMBEDDINGS_MODEL_NAME="vinai/phobert-base-v2"
QDRANT_URL = "https://78a9a0ef-6706-44f3-84d0-be177b999334.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_COLLECTION_NAME = "GGWP_LUAT_WITH_PAYLOAD"
#NGROK_STATIC_DOMAIN = "<NGROK_STATIC_DOMAIN>"
#NGROK_TOKEN=          "<NGROK_TOKEN>"
OPENAI_API_KEY =      ""
QDRANT_API_KEY =      ""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_rerank = AutoModelForSequenceClassification.from_pretrained('amberoad/bert-multilingual-passage-reranking-msmarco').to(device)
tokenizer_rerank = AutoTokenizer.from_pretrained('amberoad/bert-multilingual-passage-reranking-msmarco')

from langchain.schema.document import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from typing import List, Any
import torch
from pydantic import BaseModel, Field

class RerankRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(...)
    embedding_model: Any = Field(...)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            # Get documents from vectorstore using invoke instead of get_relevant_documents
            docs = self.vectorstore.invoke(query)
            if not docs:
                return []
                
            candidates = [doc.page_content for doc in docs]
            queries = [query]*len(candidates)
            
            # Add error handling for tokenization
            try:
                features = tokenizer_rerank(
                    queries, 
                    candidates, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,  # Add max length
                    return_tensors="pt"
                ).to(device)
            except Exception as e:
                print(f"Tokenization error: {str(e)}")
                return docs[:2]  # Return top 2 docs if reranking fails
                
            with torch.no_grad():
                scores = model_rerank(**features).logits
                values, indices = torch.sum(scores, dim=1).sort()
                
            return [docs[indices[0]], docs[indices[1]]]
        except Exception as e:
            print(f"Reranking error: {str(e)}")
            return docs[:2] if 'docs' in locals() else []  # Return top 2 docs if reranking fails

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

    def load_retriever(self, embeddings):
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY, 
            prefer_grpc=False
        )
        
        # Print collection info for debugging
        collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
        print(f"Collection info: {collection_info}")
        
        db = QdrantVectorStore(
            client=client,
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME
        )
        
        # Test the embeddings
        test_embedding = embeddings.embed_query("test")
        print(f"Embedding dimension: {len(test_embedding)}")
            
        base_retriever = db.as_retriever(
            search_kwargs={
                "k": 15,
                "score_threshold": 0.5  # Add minimum similarity threshold
            }
        )
        retriever = RerankRetriever(vectorstore=base_retriever, embedding_model=embeddings)
        return retriever

from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

class LLMServe:
    def __init__(self) -> None:
        self.embeddings = self.load_embeddings()
        self.retriever = self.load_retriever(embeddings=self.embeddings)
        self.llm = self.load_model()
        self.prompt = self.load_prompt_template()
        self.rag_pipeline = self.load_rag_pipeline(llm=self.llm,
                                            retriever=self.retriever,
                                            prompt=self.prompt)

    def load_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL_NAME,
            model_kwargs={'device': device}
        )
        return embeddings

    def load_retriever(self, embeddings):
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY, 
            prefer_grpc=False
        )
        
        # Print collection info for debugging
        collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
        print(f"Collection info: {collection_info}")
        
        db = QdrantVectorStore(
            client=client,
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME
        )
        
        # Test the embeddings
        test_embedding = embeddings.embed_query("test")
        print(f"Embedding dimension: {len(test_embedding)}")
            
        base_retriever = db.as_retriever(
            search_kwargs={
                "k": 15,
                "score_threshold": 0.5  # Add minimum similarity threshold
            }
        )
        retriever = RerankRetriever(vectorstore=base_retriever, embedding_model=embeddings)
        return retriever

    def load_model(self):
        llm = ChatOpenAI(
            model_name=GENERATE_MODEL_NAME,
            temperature=0.4,
            max_tokens=300,
            api_key=OPENAI_API_KEY
        )
        return llm

    def load_prompt_template(self):
        query_template = "Bạn là một chatbot thông minh chuyên cung cấp thông tin về luật bảo hiểm xã hội (context).\n\n### Context:{context} \n\n### Human: {question}\n\n### Assistant:"
        prompt = PromptTemplate(
            template=query_template,
            input_variables=["context","question"]
        )
        return prompt

    def load_rag_pipeline(self, llm, retriever, prompt):
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt
            },
            return_source_documents=True
        )
        return rag_pipeline

app = LLMServe()

from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
origins = ["*"]
app_api = FastAPI()
app_api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app_api.get("/")
def read_root():
    return "API RAG"

@app_api.get("/rag")
async def read_item(q: str | None = None):
    try:
        if q:
            data = app.rag_pipeline.invoke({"query": q})
            sources = []
            for docs in data["source_documents"]:
                sources.append(docs.to_json()["kwargs"])
            res = {
                "result": data["result"],
                "source_documents": sources
            }
            return JSONResponse(content=jsonable_encoder(res))
        return None
    except Exception as e:
        print(f"Error details: {str(e)}")  # Add detailed error logging
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({
                "error": str(e),
                "type": type(e).__name__,
                "details": "An error occurred while processing your request"
            })
        )

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app_api, host="0.0.0.0", port=8000)