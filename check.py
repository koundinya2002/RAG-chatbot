import streamlit as st
from secret_key import openapi_key
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain.text_splitter import LatexTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
import tempfile
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.docstore.document import Document

# Initialize the OpenAI API key
os.environ["OPENAI_API_KEY"] = openapi_key

# Set up session state if it doesn't exist
if "conversational_rag_chain" not in st.session_state:
    st.session_state.conversational_rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  # Ensure vectorstore is initialized globally

# Function to retrieve session history or create a new one if it doesn't exist
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

# User query function to handle the question-answering with context from previous chat history
def user_query(question):
    if question:
        llm = ChatOpenAI(model="gpt-4o-mini")
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Ensure retriever is created after documents are loaded
        retriever = st.session_state.vectorstore.as_retriever()  # Use the retriever from session_state.vectorstore
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        system_prompt = (
            "You are a helpful assistant. Use only the following pieces of context to answer the question at the end. "
            "Use only the content provided to answer and if the question is out of scope, just say I don't know and don't hallucinate the answers from external sources."
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Store the chain in session state
        if st.session_state.conversational_rag_chain is None:
            st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        
        # Call the chain
        result = st.session_state.conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "abc123"}}
        )["answer"]
        
        return result

# File and URL input
url_input = st.text_input(label="Enter your URL: (if multiple urls, use ',' to separate urls)")
file_input = st.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)



# Process URLs
if url_input:
    links = url_input.split(',')
    content = []
    for link in links:
        loader = WebBaseLoader(link)
        documents = loader.load()
        content.append(documents[0].page_content)
    
    # Split and embed the documents
    docs = [Document(page_content=text) for text in content]
    text_splitter = LatexTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.create_documents(content)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    st.session_state.vectorstore = vectorstore  # Store vectorstore in session state for reuse
    retriever = st.session_state.vectorstore.as_retriever()
    
    st.write("Documents processed and embedded.")
    # User question input
    user_question = st.text_input("Enter your question:")

    # If the user enters a question, process and respond
    if user_question:
        answer = user_query(user_question)
        st.write("Response:", answer)

# Process files
elif file_input:
    content = []
    for file in file_input:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getvalue())
            file_path = f.name
        
        # Handle different file types
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            content.append(documents[0].page_content)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents = loader.load()
            content.append(documents[0].page_content)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            content.append(documents[0].page_content)

    # Split and embed the documents
    docs = [Document(page_content=text) for text in content]
    st.write(docs)
    text_splitter = LatexTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.create_documents(content)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorstore = vectorstore  # Store vectorstore in session state for reuse
    retriever = st.session_state.vectorstore.as_retriever()
    st.write("Documents processed and embedded.")
    # User question input
    user_question = st.text_input("Enter your question:")

    # If the user enters a question, process and respond
    if user_question:
        answer = user_query(user_question)
        st.write("Response:", answer)
