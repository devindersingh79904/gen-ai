# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

message = [
    SystemMessage(content="You are a helpful Math tutute who provide answer bit of sarcasm"),
    HumanMessage(content="what is sequre of 25")
]


response = llm.invoke(message)
print(response.content)