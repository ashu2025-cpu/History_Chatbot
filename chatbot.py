import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt_tab")
load_dotenv()
working_dir = os.getcwd()
docs_dir_path = working_dir+"/docs_dir"
vector_db_path = working_dir+"/vector_db"
collection_name  = "document_collection"
embedding = HuggingFaceEmbeddings()
loader = DirectoryLoader(
    path=docs_dir_path,
    glob="./*.pdf",
    loader_cls=UnstructuredFileLoader
)
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)
text_chunks = text_splitter.split_documents(documents)
vector_store = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory=vector_db_path,
    collection_name=collection_name
)

st.set_page_config(
    page_title="ChapterChaser – Helps chase down those long chapters.",
    page_icon="📜⌛🏛️🏺",
    layout="centered",
)
st.title("⏳ Time Machine GPT ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
)

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding,
    persist_directory=vector_db_path
)

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    response = qa_chain.invoke(
        input = [{"role": "system", "content": "You are a helpful assistant"}, *st.session_state.chat_history]
    )
    assistant_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
