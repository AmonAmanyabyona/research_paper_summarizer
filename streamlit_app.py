import streamlit as st
from second_app import load_pdf, get_embedding_function, create_vectorstore, load_vectorstore, get_retriever, rag_chain, process_response

# Step 1: Title and UI Setup
st.title("Research Paper Information Extractor")
st.write("Upload a PDF and extract key details like title, summary, publication year, and authors.")

# Step 2: File Upload Widget
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Step 3: Load PDF and Extract Information
    pages = load_pdf(uploaded_file.name)
    
    # Step 4: Generate Embeddings
    embedding_function = get_embedding_function()

    # Step 5: Create and Load Vectorstore
    vectorstore_path = f"vectorstore_{uploaded_file.name}"
    vectorstore = create_vectorstore(pages, embedding_function, vectorstore_path)
    vectorstore = load_vectorstore(vectorstore_path, embedding_function)

    # Step 6: Initialize Retriever
    retriever = get_retriever(vectorstore)

    # Step 7: Query and Get Processed Response
    response_text = rag_chain("Give me the title, summary, publication date, authors of the research paper.", retriever)
    structured_df = process_response(response_text)

    # Step 8: Display Extracted Information
    st.write("### Extracted Information")
    st.dataframe(structured_df)  # Interactive table view

else:
    st.warning("Please upload a PDF file to proceed.")


