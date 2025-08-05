from pdf_loader import load_pdf
from embedding import get_embedding_function
from vectorstore import create_vectorstore, load_vectorstore
from retriever import get_retriever, retrieve_chunks
from rag_pipeline import rag_chain
from response_processing import process_response
import pandas as pd

# Wrap execution in a function
def process_pdf(pdf_filename):
    pages = load_pdf(pdf_filename)
    embedding_function = get_embedding_function()

    # Create unique vectorstore per file
    vectorstore_path = f"second_app_vectorstore_{pdf_filename}"
    vectorstore = create_vectorstore(pages, embedding_function, vectorstore_path)
    vectorstore = load_vectorstore(vectorstore_path, embedding_function)

    retriever = get_retriever(vectorstore)
    return retriever

# Ensure script runs only when executed directly
if __name__ == "__main__":
    pdf_filename = "Wav2Vec.pdf"  # Static file for standalone execution
    retriever = process_pdf(pdf_filename)

    # Query and process response
    response_text = rag_chain("Give me the title, summary, publication date, authors of the research paper.", retriever)
    structured_df = process_response(response_text)

    print(structured_df.to_string())  # Neatly formatted table output
