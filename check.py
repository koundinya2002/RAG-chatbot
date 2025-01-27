import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain.text_splitter import LatexTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
import tempfile
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# initialization ig
if "messages" not in st.session_state:
    st.session_state.messages =[]

# chunking the text
def to_chunk(text):
    splitter = LatexTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    to_embed(documents)

# embedding the text
def to_embed(chunks):
    embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    user_query()
  
# main function
def main():
    st.title("chatBOT")
    url_input = st.text_input(label="Enter your URL: (if multiple urls, use ',' to seperate urls)")
    file_input = st.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    content= ""
    if file_input:
        for file in file_input:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getvalue())
                file_path = f.name
            
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                content += (loader.load())[0].page_content
                
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
                content += (loader.load())[0].page_content
                
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                content += (loader.load())[0].page_content
        to_chunk(content)
    elif url_input:
        links = url_input.split(',')
        for link in links:
            loader = WebBaseLoader(link)
            content += (loader.load())[0].page_content
        to_chunk(content)

# user questions
def user_query():
    user_question = st.chat_input("ask a question:")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        llm = ChatOpenAI(api_key=openapi_key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        new_vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        prompt_template = ChatPromptTemplate.from_template(""" 
                                                        Answer the following question based only on the provided context and donot mention anything about like 'based on the provided context':
                                                        <context>
                                                        {context}
                                                        </context>
                                                        Question: {input}
                                                        """)
        docs_chain = create_stuff_documents_chain(llm, prompt_template)
        retriver = new_vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriver, docs_chain)
        response = retrieval_chain.invoke({"input": user_question})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

if __name__ == "__main__":
    main()