import os
from dotenv import load_dotenv
from pinecone.grpc import PiconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_csv_file, load_pdf_file, text_split, download_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get["PINECONE_API_KEY"]

path = "/Data"
extracted_data = load_pdf_file(path) + load_csv_file(path)
chunk_text = text_split(extracted_data)

embeddings = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

pc.create_index(
    index_name=index_name,
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1',
    )
)

doc_search = PineconeVectorStore.from_documents(
    documents=chunk_text,
    index_name=index_name,
    embedding=embeddings,
)
