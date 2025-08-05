from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from config import GITHUB_TOKEN, AZURE_ENDPOINT, AZURE_MODEL

client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

def chat_with_model(prompt):
    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage(prompt),
        ],
        temperature=1,
        top_p=1,
        model=AZURE_MODEL
    )
    return response.choices[0].message.content
