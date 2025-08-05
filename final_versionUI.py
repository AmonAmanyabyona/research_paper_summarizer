import asyncio
import streamlit as st
import hashlib
import os
from pdf_loader import load_pdf
from embedding import get_embedding_function
from vectorstore import create_vectorstore, load_vectorstore
from retriever import get_retriever, retrieve_chunks
from chat_complettion import chat_with_model  # Function for Q&A
from rag_pipeline import rag_chain
from response_processing import process_response

# **Ensure proper event loop setup**
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# **Set page layout for wider view**
st.set_page_config(page_title="Chatbot", layout="wide")

# **Header Title**
st.markdown("<h1 style='text-align: center; color: black;'>SmartReader: Explore, Ask, Summarize 📘</h1>", unsafe_allow_html=True)
st.divider()  # Adds a visual separator

def get_file_hash(file_path):
    """Generate a unique hash for the file to identify duplicates."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return hashlib.md5(file_bytes).hexdigest()

def process_pdf(pdf_filepath):
    """Check if the document already exists before reprocessing."""
    file_hash = get_file_hash(pdf_filepath)
    vectorstore_path = f"app_vectorstore_{file_hash}"  # Use hash as unique ID
    
    if os.path.exists(vectorstore_path):  
        print("Vectorstore already exists! Loading existing embeddings.")
        vectorstore = load_vectorstore(vectorstore_path, get_embedding_function())
    else:
        print("🚀 New PDF detected! Processing and creating vectorstore.")
        pages = load_pdf(pdf_filepath)
        embedding_function = get_embedding_function()
        vectorstore = create_vectorstore(pages, embedding_function, vectorstore_path)
    
    retriever = get_retriever(vectorstore)
    return retriever

# **File Upload Section**
col1, col2, col3 = st.columns([1, 3, 1])  # Centers content nicely

with col2:
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    # Ensure a stable file path
    save_directory = "uploaded_pdfs"
    os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist
    save_path = os.path.join(save_directory, pdf_file.name)

    # Save the uploaded PDF
    with open(save_path, "wb") as f:
        f.write(pdf_file.getbuffer())  # Write the uploaded file to disk

    retriever = process_pdf(save_path)  # Pass full path instead of just the filename
    st.divider()  # Adds separation before the next section

    # **Action Selection**
    col1, col2 = st.columns([2, 2])

    with col1:
        option = st.selectbox("🔍 Select an action:", ["Select an Action", "Ask a Question", "Get Summary"])

    st.divider()

    if option == "Ask a Question":
        with col2:
            query = st.text_input("Enter your question about the document:")
        
        if st.button("Submit"):
            if query:
                retrieved_docs = retrieve_chunks(retriever, query)

                if not retrieved_docs:
                    st.warning("No relevant information found in the uploaded PDF.")
                else:
                    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    prompt = f"Based on the retrieved text, answer concisely:\n\n{context}\n\n{query}"
                    response = chat_with_model(prompt)
                    st.success("Response Generated:")
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
            st.success("Summary Generated:")
            st.write(structured_df)
