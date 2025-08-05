def format_pdf_docs(docs):
    """Formats retrieved PDF documents into plain text."""
    return "\n\n".join(page.page_content for page in docs)
