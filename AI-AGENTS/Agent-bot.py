import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0,
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
bot = graph.compile()

user_input = input("Enter : ")
while user_input != "exit": 
    bot.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter : ")