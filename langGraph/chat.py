from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    llm_call_count: int

def initial_state(state:State):
    response = llm.invoke((state.get("messages")))
    return {"messages": [response], "llm_call_count": 1}

def second_state(state:State):
    response = llm.invoke((state.get("messages")))
    return {"messages": [response], "llm_call_count": 2}
    

graph_builder = StateGraph(State)

graph_builder.add_node("initial_state",initial_state)
graph_builder.add_node("second_state", second_state)

graph_builder.add_edge(START, "initial_state")
graph_builder.add_edge("initial_state", "second_state")
graph_builder.add_edge("second_state", END)

graph = graph_builder.compile()

result = graph.invoke(State(messages=["Hi! I am Rajib Gupta"], llm_call_count=0))
print("Final Result:", result)
