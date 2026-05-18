
# pyrefly: ignore [missing-import]
from langgraph.graph import MessagesState,StateGraph,START,END
from typing import Literal
# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage
# pyrefly: ignore [missing-import]
from langgraph.prebuilt import ToolNode,tools_condition



class MyMessageState(MessagesState):
    pass

llm = ChatOpenAI(
    model="gpt-4o"
)


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

llm_with_tool = llm.bind_tools([multiply])


def tool_calling_llm(state:MyMessageState):
    print(state)
    response = llm_with_tool.invoke(state["messages"])
    print("---------------------")
    print(response)
    print("---------------------")
    return {"messages": [response]}


builder = StateGraph(MyMessageState)

builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode([multiply]))


builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm",tools_condition)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

messageState = {
    "messages" : [HumanMessage(content="hello")]

}


print("------------------------------")
print("Executing tool calling LLM")

messages = graph.invoke(messageState)

print("------------------------------")
print("Result of tool calling LLM")

for m in messages['messages']:
    print(m.content)


print("------------------------------")
print("Executing tool calling LLM")


messageState = {
    "messages": [
        HumanMessage(content="Multiply 2 and 3")
    ]
}
messages = graph.invoke(messageState)

print("------------------------------")
print("Result of tool calling LLM")

for m in messages['messages']:
  print(m.content)