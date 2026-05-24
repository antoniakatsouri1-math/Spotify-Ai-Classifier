import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are the Spotify Music Intelligence Agent.
You can answer questions about Spotify audio features, AI-generated music detection, and classify tracks.
You have access to the following tools:
- retrieval_tool: Use this to search the music knowledge base for theoretical questions.
- prediction_tool: Use this to predict if a song is Human-made or AI-generated using features.
- dataset_stats_tool: Use this to get statistical metrics for the dataset.

When the user provides audio features and asks for a prediction, you MUST call the prediction_tool.
Format the input as a JSON string like this example:
{"acousticness": 0.05, "danceability": 0.85, "duration_ms": 210000, "energy": 0.91, "instrumentalness": 0.82, "key": 5, "liveness": 0.08, "loudness": -4.5, "mode": 1, "speechiness": 0.03, "tempo": 128.0, "time_signature": 4, "valence": 0.6, "popularity": 40, "short_form": 0}

Always maintain a professional and helpful tone."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


_session_histories: dict[str, list[BaseMessage]] = {}


def get_session_history(session_id: str) -> list[BaseMessage]:
    if session_id not in _session_histories:
        _session_histories[session_id] = []
    return _session_histories[session_id]


def clear_session(session_id: str) -> None:
    _session_histories.pop(session_id, None)


def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key=api_key, temperature=0)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set.")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=api_key, temperature=0)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

    else:  # groq (default)
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set.")
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)

    return llm.bind_tools(ALL_TOOLS)


def build_graph():
    llm_with_tools = _build_llm()

    def llm_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return END

    tool_node = ToolNode(ALL_TOOLS)
    graph = StateGraph(AgentState)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges("llm_node", should_continue)
    graph.add_edge("tool_node", "llm_node")
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def chat(message: str, session_id: str) -> str:
    graph = get_graph()
    history = get_session_history(session_id)
    all_messages = history + [HumanMessage(content=message)]
    result = graph.invoke({"messages": all_messages})
    response_messages = result["messages"]
    ai_response = ""
    for msg in reversed(response_messages):
        if isinstance(msg, AIMessage) and msg.content:
            ai_response = msg.content
            break
    _session_histories[session_id] = [SystemMessage(content=SYSTEM_PROMPT)] + response_messages
    return ai_response
