from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(
    "Based on the retrieved context, answer concisely:\n\nContext:\n{context}\n\nQuestion:\n{question}"
)
