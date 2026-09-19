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
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()

# ========== llm configuration ==========

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("api_key is not set !")

llm = ChatGroq(api_key=api_key, model="openai/gpt-oss-120b",temperature=0) #Temperature=0 to avoid the llm Hallucination


# ========== RAG configuration ==========

# Our embedding model(to convert text into embeddings) - It has to also be compatible with the LLM model
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)


# Loading the pdf document
# The document path
pdf_path = "/home/ayoub/Desktop/LangGraph-practice/AI-AGENTS/Stock_Market_Performance_2024.pdf"
# A safety measure
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"File not found :{pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This load the pdf
# Check if the pdf is there
try:
    pages = pdf_loader.load()
    print(f"pdf has been loaded and has {len(pages)} pages ")
except Exception as e :
    print("File not found !")
    raise


# Chunking process 
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=200
)
pages_split = text_spliter.split_documents(pages) # Applying the splitter on our pages


# The chroma vector database

persist_directory = r"/home/ayoub/Desktop/LangGraph-practice/AI-AGENTS"
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command 
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Here we actually create the chroma database using our embeddings model
try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    print("created ChromaDB vector store")
except Exception as e:
    print(f"Error setting up chromaDB :{str(e)}")
    raise


# Create the retriever
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 4 } # k is the amount of chunks to return
)


# ========== The tools and Noods  ==========

# The tools (The retriever tool)

@tool
def retriever_tool(query: str) -> str:
    """"This tool search and return the informations from the stock_market_Performance_2024"""

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the stock_market_Performance_2024"

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1} :\n{doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools)


# The state class
class StateAgent(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


# The noods

# The should_continue nood
def should_continue(state: StateAgent) -> StateAgent:
    """Check if the last message contains tool calls"""
    last_message = state["messages"][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

# The llm nood 
def call_llm(state: StateAgent) -> StateAgent:
    """"Function to call the LLM with the current state"""
    system_prompt = SystemMessage(content = """You are an AI assistant specialized in Stock Market Performance in 2024.

You have access to a retriever tool containing information from a PDF.

Rules:
- Use the retriever when you need information from the PDF.
- Call the retriever at most TWO times per user question.
- After receiving the retrieved documents, answer directly.
- Do not call the retriever again.
- If the information is not present in the retrieved documents, say so.
- Never invent information.
- Mention the source page when possible.""")

    messages = [system_prompt] + list(state["messages"])
    result = llm.invoke(messages)
    return {"messages": [result]}


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools
# Retriever Agent
def take_action(state: StateAgent) -> StateAgent:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


# ========== The graph  ==========

graph = StateGraph(StateAgent)
graph.add_node("LLM", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "LLM",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

graph.add_edge("retriever_agent", "LLM")
graph.set_entry_point("LLM")

rag_agent = graph.compile()


# ========== Running the graph  ==========

def running_agent():
    print("\n======RAG Agent======")

    while True:
        user_input = input("What's your question ?")
        if user_input.lower() in ['quit', 'exit']:
            break

        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()