# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

response = llm.invoke("Hello, how are you? i am devinder singh panesar. does you store the chat histroy or not?")

print(response.content)
