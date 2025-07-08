import os
import shutil
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from datetime import datetime
import time
import json

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv(r"C:\Users\aman\OneDrive\Desktop\hack\.env")

# Ensure OPENAI_API_KEY is available
openai_api_key = os.getenv('OPENAI_API_KEY')

CHROMA_PATH = "chroma"
PDF_PATHS = [
    "C:/Users/aman/OneDrive/Desktop/hack/AWS Validator setup procedure.pdf",
    "C:/Users/aman/OneDrive/Desktop/hack/DigitalOcean Validator setup procedure.pdf",
    "C:/Users/aman/OneDrive/Desktop/hack/GCP Validator setup procedure.pdf",
    "C:/Users/aman/OneDrive/Desktop/hack/Vultr Validator setup procedure.pdf"
]
URLS = [
    "https://www.alkimi.org/tokenomics?section=alkimi-exchange",
    "https://www.alkimi.org/how-it-works",
    "https://www.alkimi.org/tokenomics?section=validators"
]

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize session history list
session_history_file = "session_history.json"
session_history = []

# Load existing session history from file, if it exists
if os.path.exists(session_history_file):
    with open(session_history_file, 'r') as f:
        session_history = json.load(f)

def main():
    st.title("Alkimi Mate Bot")
    
    # Display session history
    display_session_history()

    # Sidebar for query text input
    query_text = st.sidebar.text_input("Enter your query:")

    # Button to trigger data generation and query
    if st.sidebar.button("Search"):
        st.info("Generating data and searching...")

        # Log query and timestamp to session history
        log_session(query_text)

        # Prepare the DB.
        generate_data_store()

        # Query the DB.
        query_db(query_text)

        # Update session history file
        update_session_history_file()

def generate_data_store():
    documents = []
    pdf_documents = load_pdfs(PDF_PATHS)
    url_documents = load_urls(URLS)
    documents.extend(pdf_documents)
    documents.extend(url_documents)
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_pdfs(pdf_paths):
    pdf_documents = []
    for pdf_path in pdf_paths:
        pdf_loader = PyPDFLoader(pdf_path)
        pdf_documents.extend(pdf_loader.load())
    return pdf_documents

def load_urls(urls):
    url_documents = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        url_documents.append(Document(page_content=text, metadata={"source": url}))
    return url_documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    st.write("Example Chunk Content:")
    st.write(document.page_content)
    st.write("Example Chunk Metadata:")
    st.write(document.metadata)

    return chunks

def save_to_chroma(chunks):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            st.write(f"Deleted existing {CHROMA_PATH}.")
        except PermissionError as e:
            st.warning(f"PermissionError: {e}. Retrying after delay.")
            time.sleep(1)  # Wait for 1 second
            try:
                shutil.rmtree(CHROMA_PATH)
                st.write(f"Deleted {CHROMA_PATH} after retry.")
            except Exception as e:
                st.error(f"Failed to delete {CHROMA_PATH}: {e}")
                return

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=openai_api_key), persist_directory=CHROMA_PATH
    )
    db.persist()
    st.write(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def query_db(query_text):
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        st.warning("Unable to find matching results.")
        st.info("Query does not match any data in the database.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(openai_api_key=openai_api_key)
    response = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"

    # Display results
    st.subheader("Query Results:")
    st.markdown(prompt)
    st.success(formatted_response)

def log_session(query_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_history.append({"query": query_text, "timestamp": timestamp})

def update_session_history_file():
    with open(session_history_file, 'w') as f:
        json.dump(session_history, f, indent=4)

def display_session_history():
    if session_history:
        st.sidebar.subheader("Session History")
        for idx, session in enumerate(session_history[::-1]):
            st.sidebar.write(f"{idx + 1}. Query: {session['query']} - Time: {session['timestamp']}")
    else:
        st.sidebar.info("Session history is empty.")

if __name__ == "__main__":
    main()
