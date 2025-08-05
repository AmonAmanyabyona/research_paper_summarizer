from retriever import get_retriever, retrieve_chunks
from formatter import format_docs
from prompt_template import prompt_template
from chat_complettion import chat_with_model
from config import GITHUB_TOKEN, AZURE_ENDPOINT, AZURE_MODEL
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Initialize Azure Client
client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)
def rag_chain(question, retriever):
    """Runs retrieval-augmented generation (RAG) pipeline."""
    
    # Step 1: Retrieve relevant chunks
    retrieved_docs = retrieve_chunks(retriever, question)
    formatted_context = format_docs(retrieved_docs)

    # Step 2: Define prompt with explicit JSON format instruction
    prompt = f"""
    You are an AI system that extracts structured information from text.
    Given the context below, respond **only in valid JSON format** using the structure provided.

    Context:
    {formatted_context}

    Question:
    {question}

    Respond strictly in JSON format without any introductory text or explanations. The output must match this JSON structure:

    {{
        "paper_title": {{
            "answer": "...",
            "sources": "...",
            "reasoning": "..."
        }},
        "paper_summary": {{
            "answer": "...",
            "sources": "...",
            "reasoning": "..."
        }},
        "publication_year": {{
            "answer": "...",
            "sources": "...",
            "reasoning": "..."
        }},
        "paper_authors": {{
            "answer": "...",
            "sources": "...",
            "reasoning": "..."
        }}
    }}
    """

    # Step 3: Send query to GPT model
    response = client.complete(
        messages=[SystemMessage("You are a helpful assistant."), UserMessage(prompt)],
        temperature=1,
        top_p=1,
        model=AZURE_MODEL
    )

    # Step 4: Debug Response
    response_text = response.choices[0].message.content
    print("Raw Response from GPT:", response_text)  # 🚀 Debugging step

    if not response_text.strip():  # Check if response_text is empty
        raise ValueError("Error: GPT response is empty!")

    return response_text
