import os
import streamlit as st
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

import asyncio
import sys

# Fix for Windows + Streamlit threads with gRPC AsyncIO
if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Use "gemini-1.5-pro" if supported in your environment
    temperature=0
)

st.title("News Research Tool 📰🔍")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...started...✅✅✅")
    data = loader.load()
    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    main_placeholder.text("Data Loading...started...✅✅✅")
    docs = text_splitter.split_documents(data)
    #create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore_gemini_ai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    #save the FAISS index
    vectorstore_gemini_ai.save_local("faiss_index")

query = main_placeholder.text_input("Question : ")
if query:
    if os.path.exists("faiss_index"):
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])

        #display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n") #split the sources by new link
            for source in sources_list:
                st.write(source)




