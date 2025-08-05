import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from pdf_loader import load_pdf
from embedding import get_embedding_function
from vectorstore import create_vectorstore, load_vectorstore
from retriever import get_retriever, retrieve_chunks
from chat_complettion import chat_with_model  # function for Q&A
from rag_pipeline import rag_chain
from response_processing import process_response
import hashlib
import os

def get_file_hash(file_path):
    """Generate a unique hash for the file to identify duplicates."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return hashlib.md5(file_bytes).hexdigest()  # Creates a unique identifier

def process_pdf(pdf_filename):
    """Check if the document already exists before reprocessing."""
    file_hash = get_file_hash(pdf_filename)
    vectorstore_path = f"app_vectorstore_{file_hash}"  # Use hash as unique ID
    
    if os.path.exists(vectorstore_path):  
        print("Vectorstore already exists! Loading existing embeddings.")
        vectorstore = load_vectorstore(vectorstore_path, get_embedding_function())
    else:
        print("New PDF detected! Processing and creating vectorstore.")
        pages = load_pdf(pdf_filename)
        embedding_function = get_embedding_function()
        vectorstore = create_vectorstore(pages, embedding_function, vectorstore_path)
    
    retriever = get_retriever(vectorstore)
    return retriever

# Streamlit UI
st.title("Chatbot: PDF Insights")

pdf_filename = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_filename:
    retriever = process_pdf(pdf_filename.name)

    st.subheader("Choose an action:")
    option = st.selectbox("Select an option:", ["Select an Action", "Ask a Question", "Get Summary"])


    if option == "Ask a Question":
        query = st.text_input("Enter your question about the document:")
        if st.button("Submit"):
            if query:
                retrieved_docs = retrieve_chunks(retriever, query)

                if not retrieved_docs:  # Check if retrieval returned anything
                    st.warning("No relevant information found in the uploaded PDF.")
                else:
                    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    prompt = f"Based on the retrieved text, answer concisely:\n\n{context}\n\n{query}"
                    response = chat_with_model(prompt)
                    st.write(response)

            else:
                st.warning("Please enter a question first.")

    elif option == "Get Summary":
        if st.button("Generate Summary"):
            response_text = rag_chain(
                "Give me the title, summary, publication date, authors of the research paper.", 
                retriever
            )
            structured_df = process_response(response_text)
            st.write(structured_df)
