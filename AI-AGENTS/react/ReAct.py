import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Annotated - Provide additional context without affecting the type itself
# ex: email = Annotated[str, "This is a valid email format!"]
# print email.__metadata__

# Sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history

# add_messages is a Reducer Function
# Rule that controls how updates from nodes are combined with the existing state 
# Tells us how to merge new data into current state 
# Without a reducer, updates would have replaced the existing value entirly


load_dotenv()


@tool
def addition(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a+b

@tool
def multiply(a: int, b: int):
    """This is a multiplication function that multiplies 2 numbers together"""
    return a*b

@tool
def susbstract(a: int, b: int):
    """This is a Substraction function that substract 2 numbers together"""


tools = [addition, multiply, susbstract]


api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    api_key= api_key,
    model = "openai/gpt-oss-120b",
    temperature=0
).bind_tools(tools)


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


def call_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query using the tools."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools= tools)

def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]

    if not getattr(last_message, "tool_calls", None):
        return "END"
    else:
        return "continue" 

graph = StateGraph(AgentState)
graph.add_node("Agent", call_agent)
graph.add_node("tool" ,tool_node)
graph.set_entry_point("Agent")
graph.add_conditional_edges(
    "Agent",
    should_continue,
    {"END": END,
    "continue": "tool"}
)
graph.add_edge("tool", "Agent")
agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages" : [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(agent.stream(inputs, stream_mode="values"))