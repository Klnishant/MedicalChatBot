import os
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from flask import Flask, request, Response, render_template, jsonify
from src.prompt import *
from src.helper import download_embeddings
from flask_cors import CORS
from src.diagnos import diagnose_disease
from src.helper import image_data
from werkzeug.utils import secure_filename
import mimetypes

load_dotenv()

app = Flask(__name__)

CORS(app)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()

index_name = "medicalbot"

docSearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriver = docSearch.as_retriever(
    search_type='similarity', search_kwargs={'k': 3})

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.4,
    max_tokens=500,
)

system_message_prompt = SystemMessagePromptTemplate.from_template(
    system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key='answer')

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriver,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
    return_source_documents=True,
)


def chat_with_bot(user_query):
    score = retriver.vectorstore.similarity_search_with_score(user_query, k=1)
    print(float(score[0][-1]) > 0.60)

    if score[0][-1] > 0.60:
        response = conversation_chain.invoke(
            {"question": user_query})['answer']
    else:
        response = "I don't know"
    return response


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=['GET', 'POST'])
def chat():
    data = request.get_json()
    msg = data['msg']
    user_query = msg
    return chat_with_bot(user_query)


@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    UPLOAD_FOLDER = "images"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    if "img" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    img = request.files["img"]

    if img.filename == "":
        return jsonify({"error": "No image file provided"}), 400
    filename = secure_filename(img.filename)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img.save(file_path)
    print(file_path)

    print(file_path)

    img_data = image_data(file_path)

    mimetype = mimetypes.guess_type(file_path)[0]

    image_parts = [
        {
            "mime_type": mimetype,
            "data": img_data
        }
    ]

    prompt_parts = [
        image_parts[0],
        system_prompts[0],
    ]
    result = diagnose_disease(prompt_parts)
    if (result):
        os.remove(file_path)
    return result


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
