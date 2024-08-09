from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Chroma
from llms.gemini import GeminiLLm
from dotenv import load_dotenv

load_dotenv()


llm = GeminiLLm().gemini_pro

embiddings = GooglePalmEmbeddings()

text_splitter = CharacterTextSplitter(chunk_size=200, separator="\n",
                                      chunk_overlap=0)
loader = TextLoader("facts.txt")
docs  = loader.load_and_split(
    text_splitter=text_splitter
)
db = Chroma.from_documents(docs, persist_directory='emb', embedding=embiddings)
# docs = text_splitter.split_documents(loader.load())

# '''Similarity search with score'''
# results = db.similarity_search_with_score("What is earth composed of?",
#                                           k=1)
#  sample result [0.56789, pageContent("matching text from the documet")]

# '''Similarity search without score where k is no of desired results'''

results = db.similarity_search("What is earth composed of?", k=1)
# results = ["list of texts"]
for result in results:
    print(result.page_content)