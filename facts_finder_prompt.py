from dotenv import load_dotenv
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from llms.gemini import GeminiLLm

llm = GeminiLLm().chatBot

load_dotenv()

embiddings = GooglePalmEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embiddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

result = chain.run("What is earth composed of?")

print(result)