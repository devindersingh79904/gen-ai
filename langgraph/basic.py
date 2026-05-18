from typing_extensions import TypedDict
import random
from typing import Literal
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph,START,END



class State(TypedDict):
    graph_state : str


def node_1(state):
    print("-----------Node 1---------")
    return {"graph_state" : state["graph_state"]  + "AGI."}


def node_2(state):
    print("-----------Node 2---------")
    return {"graph_state" : state["graph_state"]  + " Acheived!"}

def node_3(state):
    print("-----------Node 3---------")
    return {"graph_state" : state["graph_state"]  + " Not Acheived :("}


def decide_mode(state) -> Literal["node_2", "node_3"]:

    user_input = state["graph_state"]
    print("------------decide_mode----------")
    print(user_input)
    print("----------------------------------")

    if random.random() < 0.5:
        return "node_2"
    return "node_3"



builder = StateGraph(State)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1",decide_mode)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()

print(graph.invoke({"graph_state": "Has AGI Been achevied?"}))
print(graph.get_graph().draw_ascii())