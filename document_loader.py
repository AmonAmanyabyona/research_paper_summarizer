from pdf_loader import load_pdf
from csv_loader import load_csv

def load_document(file_path):
    """Determines file type and loads appropriate document processor."""
    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".csv"):
        return load_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or CSV.")
