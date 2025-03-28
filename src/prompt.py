from langchain.schema import SystemMessage

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

system_message = SystemMessage(content="""
You are a highly knowledgeable AI assistant. 
Use the retrieved documents as context but do not make up information. 
If you don’t know the answer, just say you don’t know.
""")