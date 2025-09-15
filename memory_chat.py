from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv  

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Step 1: Build knowledge base
topic = input("Enter topic to build knowledge base: ")
loader = WikipediaLoader(query=topic, load_max_docs=3)
docs = loader.load()
if not docs:
    print("No documents found.")
    exit()

# Step 2: Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = splitter.split_documents(docs)

# Step 3: Vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
db = FAISS.from_documents(splits, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Step 4: LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# Step 5: Memory
memory = ConversationBufferMemory(return_messages=True)

# Step 6: Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use only the provided context to answer. "
               "If the context is insufficient, say 'I don't know'."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
    ("system", "Context:\n{context}")
])

# Step 7: Conversational chain
def conversational_chain(user_query):
    # retrieve docs
    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # build inputs
    inputs = {
        "history": memory.load_memory_variables({})["history"],
        "query": user_query,
        "context": context
    }

    # run LLM
    response = (prompt | llm | StrOutputParser()).invoke(inputs)

    # update memory
    memory.save_context({"query": user_query}, {"response": response})

    return response

# Step 8: Chat loop
while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    ans = conversational_chain(q)
    print("Bot:", ans)
