from pdf_loader import load_pdf
from embedding import get_embedding_function
from vectorstore import create_vectorstore, load_vectorstore
from retriever import get_retriever, retrieve_chunks
from chat_complettion import chat_with_model

# Wrap execution in a function
def process_pdf(pdf_filename):
    pages = load_pdf(pdf_filename)  # Load PDF dynamically
    embedding_function = get_embedding_function()

    # Create unique vectorstore path per file
    vectorstore_path = f"app_vectorstore_{pdf_filename}"
    vectorstore = create_vectorstore(pages, embedding_function, vectorstore_path)
    vectorstore = load_vectorstore(vectorstore_path, embedding_function)

    retriever = get_retriever(vectorstore)
    return retriever

# Ensure script runs only when executed directly
if __name__ == "__main__":
    pdf_filename = "Wav2Vec.pdf"  # Static file for standalone execution
    retriever = process_pdf(pdf_filename)

    # Sample query
    query = "What is Machine Translation?"
    retrieved_docs = retrieve_chunks(retriever, query)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"Based on the following retrieved text, answer concisely:\n\n{context}\n\nWhat is Machine Translation?"
    response = chat_with_model(prompt)

    print("\n\n")
    print(response)
