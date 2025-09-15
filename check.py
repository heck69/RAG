from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os   
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

chat = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
print(chat)