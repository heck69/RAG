from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv  

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


#QUERY FOR FINDING THE DOCUMENTS FOR THE TOPIC
query = input("Enter your topic: ")
loader = WikipediaLoader(query=query, load_max_docs=3)
docs = loader.load()

# SPLITTING THE DOCUMENTS INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
splitted = text_splitter.split_documents(docs)

# CREATING THE VECTOR STORE
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
db = FAISS.from_documents(splitted, embeddings)

# RETRIEVING RELEVANT DOCUMENTS
retriever = db.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(query)

#Augumentation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided document context.
      If the context is insufficient, just say you don't know.

      {context}
      query: {query}
    """,
    input_variables = ['context', 'query']
)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "query": query})



chain = (
    RunnableParallel(
        context=RunnableLambda(lambda x: context_text),
        query=RunnablePassthrough()
    )
    | prompt
    | llm
    
)
query = input("Enter your query: ")
response = chain.invoke(query)
print("Response:", response.content)









