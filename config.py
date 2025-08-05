import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Ensure this is set in .env
AZURE_ENDPOINT = "https://models.github.ai/inference"
AZURE_MODEL = "openai/gpt-4o"

# Debugging check
if not GITHUB_TOKEN:
    raise ValueError("API key not found! Ensure GITHUB_TOKEN is set in your .env file.")
