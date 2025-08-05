from pdf_loader import load_pdf
from embedding import get_embedding_function
from vectorstore import create_vectorstore, load_vectorstore
from retriever import get_retriever, retrieve_chunks
from rag_pipeline import rag_chain
from response_processing import process_response
import pandas as pd

# Step 1: Load PDF
pages = load_pdf("Wav2Vec.pdf")

# Step 2: Generate Embeddings
embedding_function = get_embedding_function()

# Step 3: Create Vectorstore
pdf_filename = "Wav2Vec.pdf"  # Update dynamically if needed
vectorstore_path = f"second_app_vectorstore_{pdf_filename}"
vectorstore = create_vectorstore(pages, embedding_function, vectorstore_path)
vectorstore = load_vectorstore(vectorstore_path, embedding_function)

# Step 4: Initialize Retriever
retriever = get_retriever(vectorstore)

# Step 5: Query and Process Response
response_text = rag_chain("Give me the title, summary, publication date, authors of the research paper.", retriever)
structured_df = process_response(response_text)


print(structured_df.to_string())  # Shows table neatly formatted in console
