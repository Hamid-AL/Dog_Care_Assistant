import streamlit as st
from dotenv import load_dotenv
#from PyPDF2 import PdfReader
#import pdfplumber
import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_cohere import CohereEmbeddings
import cohere


import os

def get_text_data(directory_path):
    """
    Reads text content from all .txt files in a given directory and combines them into a single string.

    Parameters:
    - directory_path: Path to the directory containing .txt files.

    Returns:
    - A single string containing the text from all the .txt files in the directory.
    """
    text = ""
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory_path):
        # Process only .txt files
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text += file.read() + "\n"
    return text



def get_text_chunks(text):
    SEPARATORS = ["\n", ".", ","]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
        separators=SEPARATORS,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks=None, index_file_path="./vectorstore.index"):
    # Check if the index file already exists
    if os.path.exists(index_file_path):
        print("Loading existing vector store...")
        # Load the existing FAISS index from file
        vectorstore = FAISS.load_local(index_file_path, CohereEmbeddings(model="embed-english-light-v3.0"), allow_dangerous_deserialization=True)
        print("Vector stroe loaded")
    elif text_chunks:
        print("Creating a new vector store...")
        # Create a new FAISS index from text chunks and save it
        embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local(index_file_path)  # Save the FAISS index to the specified file path
        print("vector store created")
    else:
        raise ValueError("No existing vector store found, and no text chunks provided to create one.")
    
    return vectorstore


def handle_userinput(user_question):
    if "vectorstore" in st.session_state and st.session_state.vectorstore:
        retrieved_docs = st.session_state.vectorstore.similarity_search(user_question)
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]

        context ="/n".join(
            [f" Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
        )
        
        cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        response = cohere_client.chat(
            chat_history=[
                {
                    "role": "USER",
                    "message": """You are a dog care assistant. If the user's input requires detailed information, use the context provided to answer their question. If the context is not relevant or the question does not require specific details (e.g., a simple greeting like 'hello', 'thank you'), respond based on your general knowledge. Always ensure your answer is concise, relevant, and directly addresses the user's input. If the question is not related to dog care, politely state that you are a dog care assistant and can only answer questions about dog care."""
                    },
                    {
                    "role": "CHATBOT",
                    "message": "Please provide the question and the context."
                    }

            ],
            message=f"Question:\n{user_question}\Context:{context}\nResponse:\n",
        )
        st.session_state.messages.append({"role": "assistant", "content": response.text})
    else:
        st.warning("Please process the PDF documents first.")


def main():
    load_dotenv()
    st.set_page_config(page_title='Dog Care Assistant', page_icon=':dog:')
    st.header('Dog Care Assistant :dog:')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    index_file_path = "./vectorstore.index"
    directory_path="./rag_data"

    # Check if the vector store is already loaded into the session state
    if "vectorstore" not in st.session_state:
        # If the vector store index exists on disk, load it directly.
        if os.path.exists(index_file_path):
            st.session_state.vectorstore = get_vectorstore(index_file_path=index_file_path)
        else:
            text=get_text_data(directory_path)
            text_chunks= get_text_chunks(text)
            st.session_state.vectorstore= get_vectorstore(text_chunks, index_file_path=index_file_path)

    user_question = st.chat_input("what do you want to know about dogs?")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        handle_userinput(user_question)


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


if __name__ == '__main__':
    main()