# LangGraph Practice

A hands-on repository for learning and experimenting with **LangGraph**, **LangChain**, LLMs, tool calling, and AI agent architectures.

This repository contains progressively built exercises focused on understanding how modern AI agents work internally rather than only using high-level abstractions.

## 🎯 Objectives

The main goal of this repository is to understand and implement:

* LangGraph state management
* Graph-based agent workflows
* LLM and tool interaction
* Tool calling
* Message-based state
* `ToolMessage` and `tool_call_id`
* `Command` and state updates
* `ToolRuntime`
* Agent and tool architecture
* Human-in-the-loop workflows

## 🛠️ Technologies

* Python
* LangGraph
* LangChain
* Groq
* LLMs
* Pydantic
* Python-dotenv

## 📂 Project Structure

```text
LangGraph-practice/
│
├── AI-AGENTS/
│   └── ...
│
├── ...
│
├── .env.example
├── requirements.txt
└── README.md
```

## 🤖 AI Agent Exercises

### Drafter Agent

A document creation agent built with LangGraph.

The agent can:

* Generate document content
* Update an existing document
* Save the document
* Use tools through LLM tool calling
* Maintain document state across interactions

The project helped me understand how `ToolRuntime`, `Command`, `ToolMessage`, and `tool_call_id` work together when a tool needs to modify or communicate with graph state.

## 🧠 Concepts Practiced

### StateGraph

Used to define the state and workflow of an AI agent.

### Tools

Implemented tools that can be called by the LLM depending on the user's request.

### Tool Calling

Explored the interaction between:

```text
User
 ↓
LLM
 ↓
Tool Call
 ↓
Tool Execution
 ↓
ToolMessage
 ↓
LLM
```

### Command

Used `Command` when a tool needs to update the graph state in addition to returning information to the model.

### ToolRuntime

Used `ToolRuntime` to access runtime information such as the current state and tool call context.

### ToolMessage

Used `ToolMessage` to send the result of a tool execution back to the LLM using the corresponding `tool_call_id`.

## 🚀 Setup

Clone the repository:

```bash
git clone https://github.com/AyoubSs1/LangGraph-practice.git
cd LangGraph-practice
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

## ▶️ Running the Examples

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Then run an example:

```bash
python AI-AGENTS/drafter2.py
```

## 📚 Learning Approach

This repository is part of my practical learning journey in AI agent development.

Rather than relying only on tutorials, I implement the architecture myself and investigate the behavior of each component through experimentation, debugging, and iterative improvements.

## 🔮 Future Work

Planned experiments include:

* Multi-agent systems
* Human-in-the-loop workflows
* MCP integration
* FastAPI integration
* Persistent agent state
* More advanced LangGraph architectures

---

**Author:** Ayoub Soussi
**GitHub:** https://github.com/AyoubSs1
