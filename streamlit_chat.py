import streamlit as st
if st.button("Exit"):
    st.session_state.clear()
    st.rerun()
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv 

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
 

# ----------------------
# Setup
# ----------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Conversational RAG Bot", layout="centered")

st.title("Wiki RAG")

# ----------------------
# Input: Topic
# ----------------------
if "db" not in st.session_state:
    topic = st.text_input("Enter a topic to build knowledge base:", key="topic_input")
    if topic:
        with st.spinner(f"Loading documents for '{topic}'..."):
            loader = WikipediaLoader(query=topic, load_max_docs=3)
            docs = loader.load()

            if not docs:
                st.error("No documents found.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = splitter.split_documents(docs)

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY
            )
            db = FAISS.from_documents(splits, embeddings)

            st.session_state.db = db
            st.session_state.retriever = db.as_retriever(search_kwargs={"k": 3})
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY
            )
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            st.session_state.prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use only the provided context. "
                           "If the context is insufficient, say 'I don't know'."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}"),
                ("system", "Context:\n{context}")
            ])
            st.session_state.parser = StrOutputParser()
            st.session_state.history = []  # for Streamlit UI

            st.success("Knowledge base created. Start chatting below.")

# ----------------------
# Chat function
# ----------------------
def conversational_chain(user_query):
    retriever = st.session_state.retriever
    llm = st.session_state.llm
    prompt = st.session_state.prompt
    memory = st.session_state.memory
    parser = st.session_state.parser

    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    inputs = {
        "history": memory.load_memory_variables({})["history"],
        "query": user_query,
        "context": context
    }

    response = (prompt | llm | parser).invoke(inputs)

    memory.save_context({"query": user_query}, {"response": response})
    return response

# ----------------------
# Chat UI
# ----------------------
if "db" in st.session_state:
    user_input = st.chat_input("Ask a question...")

    if user_input:
        st.session_state.history.append(("user", user_input))
        response = conversational_chain(user_input)
        st.session_state.history.append(("bot", response))

    # Render chat
    for role, text in st.session_state.history:
        if role == "user":
            with st.chat_message("user"):
                st.write(text)
        else:
            with st.chat_message("assistant"):
                st.write(text)
