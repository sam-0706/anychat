import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Load secrets from Streamlit's secrets manager
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PASSWORD = st.secrets["PASSWORD"]

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    user_password = st.text_input("Enter password to access:", type="password")
    if st.button("Login"):
        if user_password == PASSWORD:
            st.session_state.authenticated = True
            st.success("Access granted!")
            st.rerun()  # ✅ Fixed: Re-run the script to refresh UI after login
        else:
            st.error("Incorrect password!")
    st.stop()  # Stop execution if authentication fails

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to extract text from TXT files
def get_txt_text(txt_files):
    text = ""
    for txt in txt_files:
        text += txt.read().decode("utf-8") + "\n"
    return text

# Function to extract text from CSV or Excel
def get_csv_excel_text(files):
    text = ""
    for file in files:
        file_extension = file.name.split(".")[-1]
        if file_extension == "csv":
            df = pd.read_csv(file)
        else:  # xlsx
            df = pd.read_excel(file)
        text += df.to_string(index=False) + "\n"
    return text

# Function to process text files and convert them into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create vector embeddings
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )

# Function to handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main function
def main():
    st.set_page_config(page_title="AnyChat", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("AnyChat")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, CSV, or Excel files and click 'Process'",
            accept_multiple_files=True,
            type=["pdf", "txt", "csv", "xlsx"]
        )

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = ""

                # Separate files by type
                pdf_files = [file for file in uploaded_files if file.name.endswith(".pdf")]
                txt_files = [file for file in uploaded_files if file.name.endswith(".txt")]
                csv_excel_files = [file for file in uploaded_files if file.name.endswith((".csv", ".xlsx"))]

                # Extract text from different file types
                raw_text += get_pdf_text(pdf_files) if pdf_files else ""
                raw_text += get_txt_text(txt_files) if txt_files else ""
                raw_text += get_csv_excel_text(csv_excel_files) if csv_excel_files else ""

                if not raw_text.strip():
                    st.error("No readable text found in uploaded files.")
                    return

                # Process text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
