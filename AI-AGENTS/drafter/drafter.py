import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# 1. ENVIRONMENT

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")


# 2. DOCUMENT STATE
document_content = ""

# 3. LANGGRAPH STATE

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 
# 4. TOOLS

@tool
def update(content: str) -> str:
    """
    Updates the document with the provided complete content.

    The content must be the COMPLETE document after applying
    the requested changes.
    """

    global document_content

    document_content = content

    return (
        "Document has been updated successfully!\n\n"
        "Current document:\n"
        f"{document_content}"
    )


@tool
def save(filename: str) -> str:
    """
    Saves the current document to a text file and finishes
    the drafting process.

    Args:
        filename: Name of the file to save.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:

        with open(filename, "w", encoding="utf-8") as file:
            file.write(document_content)

        print(f"\n💾 Document saved to: {filename}")

        return (
            f"Document has been saved successfully to "
            f"'{filename}'."
        )

    except Exception as e:

        return f"Error saving document: {str(e)}"


# 5. TOOLS LIST

tools = [update, save]

# 6. LLM

model = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0,
).bind_tools(tools)

# 7. AGENT NODE

def agent(state: AgentState) -> AgentState:

    system_prompt = SystemMessage(
        content=f"""
You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
"""
    )
    if not state["messages"]:

        user_input = input(
            "I'm ready to help you create a document. "
            "What would you like to create?"
        )

    else:

        user_input = input(
            "\n👤 What would you like to do with the document? "
        )

        print(f"\n👤 USER: {user_input}")

    user_message = HumanMessage(
        content=user_input
    )
    all_messages = (
        [system_prompt]
        + list(state["messages"])
        + [user_message]
    )
    response = model.invoke(all_messages)
    print(f"\n🤖 AI: {response.content}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages": [user_message,response]}

tool_node = ToolNode(tools=tools)


# 9. ROUTING FUNCTION

def should_continue(state: AgentState) -> str:
    """
    Decide what happens after the tools node.

    update → go back to Agent
    save   → END
    """

    messages = state["messages"]

    # Search for the latest AI message containing tool calls
    for message in reversed(messages):

        if isinstance(message, AIMessage) and message.tool_calls:

            tool_name = message.tool_calls[0]["name"]

            print(f"\n ROUTER: {tool_name}")

            if tool_name == "update":
                return "continue"

            if tool_name == "save":
                return "end"
    return "end"


# 10. PRINT TOOL MESSAGES

def print_messages(messages):
    """
    Print recent tool results in a readable format.
    """

    if not messages:
        return

    for message in messages[-3:]:

        if isinstance(message, ToolMessage):

            print(
                f"\n🛠️ TOOL RESULT:\n"
                f"{message.content}"
            )


# 11. BUILD GRAPH

graph = StateGraph(AgentState)

graph.add_node(
    "agent",
    agent
)

graph.add_node(
    "tools",
    tool_node
)
graph.set_entry_point("agent")

graph.add_edge(
    "agent",
    "tools"
)

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)
app = graph.compile()


# 13. RUN DRAFTER

def run_document_agent():

    print("\n================================")
    print("           DRAFTER")
    print("================================")

    state = {
        "messages": []
    }

    try:

        for step in app.stream(
            state,
            stream_mode="values"
        ):

            if "messages" in step:

                print_messages(
                    step["messages"]
                )

    except KeyboardInterrupt:

        print("\n\n Drafter stopped by user.")

    except Exception as e:

        print(
            f"\n\nERROR:\n{type(e).__name__}: {e}"
        )

    print("\n================================")
    print("       DRAFTER FINISHED")
    print("================================")


# 14. MAIN
if __name__ == "__main__":
    run_document_agent()