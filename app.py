import os
import sys
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Fix BEFORE importing gRPC / LangChain / Google libs
if sys.platform.startswith("win"):
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


# Load .env
load_dotenv()
# It's generally recommended to use GOOGLE_API_KEY as the env var name
# as LangChain/Google libraries often look for this by default.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Changed from GEMINI_API_KEY

if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not found. Please set it in your .env file.")


@st.cache_resource
def build_vector_store(topic: str):
    """Load data from Wikipedia and build FAISS vector store"""
    loader = WikipediaLoader(query=topic, load_max_docs=1)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    # Pass the API key explicitly here
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


@st.cache_resource
def create_qa_chain(_vectorstore):
    """Create Conversational RAG chain with memory"""
    # The ChatGoogleGenerativeAI model will typically pick up GOOGLE_API_KEY
    # from the environment automatically, but you can also pass it explicitly:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3, google_api_key=GOOGLE_API_KEY)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
    )
    return qa_chain


# ---- Streamlit App ----
st.title("Wikipedia RAG")

# Step 1: User enters topic
topic = st.text_input("Enter a Wikipedia topic:")

# Only build knowledge base if a topic is provided
if topic:
    vectorstore = build_vector_store(topic)
    qa_chain = create_qa_chain(vectorstore)

    st.success(f"Knowledge base for **{topic}** is ready!")

    # Step 2: User asks a specific question
    user_q = st.text_input(f"What do you want to know about **{topic}**?")

    if user_q:
        response = qa_chain({"question": user_q})
        st.markdown(f"**Answer:** {response['answer']}")

        with st.expander("Sources"):
            for doc in response["source_documents"]:
                st.write(doc.metadata.get("source", "Wikipedia"))
                st.write(doc.page_content[:200] + "...")