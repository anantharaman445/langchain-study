from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class GeminiLLm:
    @property
    def gemini_pro(self):
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        return llm


    @property
    def chatBot(self):
        chat = ChatGoogleGenerativeAI(model="gemini-pro")
        return chat