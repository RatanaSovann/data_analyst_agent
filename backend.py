import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any
from workflow import AgentState
from langgraph.graph import StateGraph
from workflow import InputData, call_model, call_tools, route_to_tools
from typing import List
from langchain_core.runnables import RunnableLambda



class PythonChatbot:
    def __init__(self):
        super().__init__()
        self.reset_chat()
        self.graph = self.create_graph()

    # ------------------------------------------------
    # 🧩 Helper: lightweight safe dict
    # ------------------------------------------------
    @staticmethod
    def get_safe_dict(obj):
        return obj if isinstance(obj, dict) else {}

    # ------------------------------------------------
    # ⚙️ Graph creation (agent ↔ tools cycle)
    # ------------------------------------------------
    def create_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)

        workflow.add_conditional_edges("agent", route_to_tools)
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")

        graph = workflow.compile()
        print("✅ Graph compiled: agent ↔ tools cycle")
        return graph

    # ------------------------------------------------
    # 🧠 MAIN EXECUTION
    # ------------------------------------------------
    def user_sent_message(self, user_query, input_data: list):
        """Efficiently run LangGraph pipeline for a new user query."""

        # ⚙️ Lightweight references
        file_path = getattr(self, "file_path", "")
        context_added = getattr(self, "context_added", False)
        current_vars = self.get_safe_dict(getattr(self, "current_variables", {}))

        # 🧹 Keep concise conversation memory
        short_history = self.chat_history[-4:] if len(self.chat_history) > 4 else self.chat_history

        # ⚙️ Prepare state for the graph — only primitives
        input_state = {
            "messages": short_history + [HumanMessage(content=user_query)],
            "input_data": input_data[:1] if input_data else [],
            "file_path": file_path,
            "current_variables": current_vars,
            "context_added": context_added,
            "intermediate_outputs": [],
            "output_image_paths": [],
        }

        print("\n🟩 [DEBUG] Input state (compact):")
        for k, v in input_state.items():
            if isinstance(v, list):
                print(f"  - {k}: list[{len(v)}]")
            elif isinstance(v, dict):
                print(f"  - {k}: dict[{len(v)}]")
            else:
                print(f"  - {k}: {v}")

        # 🚀 Run the LangGraph cycle
        result = self.graph.invoke(input_state, {"recursion_limit": 25})

        # 🧾 Store back results
        self.file_path = result.get("file_path", file_path)
        self.context_added = result.get("context_added", True)
        self.current_variables = self.get_safe_dict(result.get("current_variables", current_vars))

        # 🧩 Append messages
        new_msgs = result.get("messages", [])
        if new_msgs:
            self.chat_history.extend(new_msgs)
            MAX_HISTORY = 8
            if len(self.chat_history) > MAX_HISTORY:
                print(f"✂️ Trimming chat history {len(self.chat_history)} → {MAX_HISTORY}")
                self.chat_history = self.chat_history[-MAX_HISTORY:]

        # ✅ Sync outputs for UI tabs
        self.intermediate_outputs = result.get("intermediate_outputs", [])
        self.output_image_paths = result.get("output_image_paths", {})  # now guaranteed visible

        print("🧩 [DEBUG] intermediate_outputs:", len(self.intermediate_outputs))
        print("🧩 [DEBUG] output_image_paths:", self.output_image_paths)

    # ------------------------------------------------
    # 🧹 RESET FUNCTION
    # ------------------------------------------------
    def reset_chat(self):
        self.chat_history = []
        self.intermediate_outputs = []
        self.output_image_paths = {}
        self.file_path = ""
        self.context_added = False
        self.current_variables = {}
        print("🧹 Chatbot state fully reset.")

