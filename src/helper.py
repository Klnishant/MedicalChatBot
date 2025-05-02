import io
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from PIL import Image


def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)

    pdf = loader.load()

    return pdf


def load_csv_file(data):
    loader = DirectoryLoader(data,
                             glob="*.csv",
                             loader_cls=CSVLoader)

    csv = loader.load()

    return csv


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


def image_data(path):
    image = Image.open(path)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    return image_bytes.getvalue()
