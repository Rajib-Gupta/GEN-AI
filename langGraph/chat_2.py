from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from typing import Optional, Literal
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
load_dotenv()

client = OpenAI()

class State(TypedDict):
    user_query:str
    llm_output: Optional[str]
    is_good: Optional[bool]


def chatBot (state:State):
    print("User Query:", state["user_query"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"user","content": state["user_query"]}
        ]
    )
    state["llm_output"] = response.choices[0].message.content
    return state

def evaluate_response(state:State)->Literal["second_chat_bot","end_node"]:
     print("Evaluating Response...")
     is_response_good = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are an evaluator. Assess if the following response is good and relevant to the user's query. Respond with true if it is good, otherwise false."},
            {"role":"user","content": f"User Query: {state['user_query']}\nResponse: {state['llm_output']}"}
        ]
    )
     response_text = is_response_good.choices[0].message.content.strip().lower()
     state["is_good"] = response_text == "true"
     if response_text == "true":
         return "end_node"
     return "second_chat_bot"


def end_node(state:State):
    print("Ending the chat process.")
    return state


def second_chat_bot(state:State):
    print("Generating Second Response...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"user","content": state["user_query"]}
        ]
    )
    state["llm_output"] = response.choices[0].message.content
    return state

grap_builder = StateGraph(State)

grap_builder.add_node("chat_bot", chatBot)
grap_builder.add_node("evaluate_response", evaluate_response)
grap_builder.add_node("second_chat_bot", second_chat_bot)
grap_builder.add_node("end_node", end_node)


grap_builder.add_edge(START, "chat_bot")
# grap_builder.add_edge("chat_bot", "evaluate_response")
grap_builder.add_conditional_edges("chat_bot", evaluate_response)
grap_builder.add_edge("second_chat_bot", END)
grap_builder.add_edge("end_node", END)

graph = grap_builder.compile()

updated_chat = graph.invoke({"user_query":"what is the value of 12+12", "llm_output":None, "is_good":None})
print("Final Chat Output:", updated_chat["llm_output"])