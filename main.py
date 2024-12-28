from dotenv import load_dotenv
import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

DATA_PATH = "data/users"


PROJECT_ID = "test-aba81" 
LOCATION = "europe-west3" 

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)

embeddings = VertexAIEmbeddings(model="text-embedding-004")


qdrant_client = QdrantClient(
    url="https://d35a5d9c-123b-4936-962f-3df58c61680f.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_API_KEY"),
)

# qdrant_client.recreate_collection(
#     collection_name="my-collection",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="my-collection",
    embedding=embeddings,
)

def get_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md",show_progress=True)
    documents = loader.load()
    return documents

# docs = load_documents()
# chunks = get_chunks(docs)

# _ = vector_store.add_documents(documents=chunks)
