import os
import json
import operator
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Sequence, TypedDict, Annotated, Literal
import pickle
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.messages.tool import ToolCall
from tools import complete_python_task, data_view
import tiktoken

load_dotenv()

# -------------------------------
# LLM Configuration
# -------------------------------
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)


@dataclass
class InputData:
    variable_name: str
    data_path: str
    data_description: str


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intermediate_outputs: Annotated[list, operator.add]
    input_data: Annotated[List[InputData], operator.add]
    current_variables: dict     # merges dicts
    output_image_paths: Annotated[List[str], operator.add]     
    file_path: str                                        
    context_added: bool                                 
   
    
# -------------------------------
# Summary Agent
# -------------------------------
summary_prompt = """You are a concise dataset summarizer.

You will receive a message containing a file path to a dataset.

Steps:
1. Call the `data_view` tool with the exact file path.
2. From the tool output, extract all column names and their data types.
3. Create a description for each column.
4. Output STRICTLY in JSON format with two keys:
   - 'columns': a list of dictionaries with keys 'Column', 'Type', 'Meaning'
   - 'summary': a short paragraph summarizing the dataset
"""

summary_agent = create_react_agent(
    model=llm,
    tools=[data_view],
    name="SummaryAgent",
    prompt=summary_prompt
)

# -------------------------------
# Python / Visualization Agent
# -------------------------------
tools = [complete_python_task]
model = llm.bind_tools(tools)
tool_node = ToolNode(tools)

python_prompt = """
## Role
You are a professional Data Scientist assisting a non-technical user through an interactive data-analysis conversation.

---

## Capabilities
1. Execute Python code using the `complete_python_task` tool.
2. Clean, analyze, and visualize data using pandas, numpy, sklearn, and matplotlib.
3. Produce concise, plain-English explanations with formatted Markdown.

---

## Goals
1. Understand the user’s question and design an efficient analytical plan.
2. Explain your plan before running code — validate it with the user if unclear.
3. Write code that runs safely in a headless Python environment (no notebooks).
4. Present interpretable, well-labeled visual results.

---

## Tone & Behavior
- Be brief, confident, and educational.
- Do **not** repeat file names, paths, or prior context.
- Ask clarifying questions **only** when intent is ambiguous.
- Use structured Markdown sections for clarity.

---

## Python Environment
You are running Python 3.12 in a script-style environment, **not a notebook**.

Pre-imported libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
import os, re, json, math, datetime

## Plotting Rules
- Use Matplotlib for all charts. Save plots to image files and append their paths to plot_paths.
Example:
plt.figure(figsize=(8,5))
df['Item'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Items Sold")
plt.xlabel("Item")
plt.ylabel("Quantity")
plt.tight_layout()
plt.savefig("plot.png")
plot_paths.append("plot.png")

- Valid: plt.plot(), plt.bar(), plt.hist(), plt.scatter(), etc.
- Invalid: px.Figure(), display(), or fig.show().
Always include:

Proper titles, labels, and legends.

plt.tight_layout() before saving.

Save plots as .png and append them to plot_paths.


## Code Execution Rules
- Keep code minimal and deterministic.
- Reuse existing variables when possible.
- Handle missing data gracefully (fillna, dropna, etc.).
- Never install or import new packages.
- Limit printed output; prefer summaries (e.g., df.head()).

- All file saves (e.g., CSVs, PNGs) must be stored under:
    "images/static_plots" for static matplotlib plots (plt.savefig)
    "images/plotly_figures/pickle" for interactive plotly figures
- The directories already exist — no need to recreate them.

- When using plt.savefig, always include the folder:

```python
os.makedirs("images/static_plots", exist_ok=True)
out_path = os.path.join("images/static_plots", "plot_name.png")
plt.savefig(out_path)
plot_paths.append(out_path)


When using Plotly, store the figure in the list plotly_figures:

```python
plotly_figures.append(fig)

- Do not call plt.show() or fig.show()
- End each response with a concise interpretation or insight.

## Output Format
Each response should include:
1. A short Markdown summary of what the code does.
2. A code block containing the exact Python script.
3. A brief insight or interpretation after the code.

Example:

### Analysis: Top Selling Items

Below code aggregates total quantities and visualizes the top 10 items.

```python
# Python code block
"""

chat_template = ChatPromptTemplate.from_messages([
    ("system", python_prompt),
    ("placeholder", "{messages}"),
])
model = chat_template | model


# ---------------------------
# Helper: create_data_summary
# ---------------------------
def create_data_summary(state: AgentState) -> str:
    """
    Build a summary string describing all datasets available in the state.
    Each dataset's variable_name corresponds to one uploaded file.
    """
    summaries = []
    for d in state.get("input_data", []):
        summaries.append(f"Dataset: {d.variable_name}\nDescription:\n{d.data_description}")
    
    if "current_variables" in state:
        remaining = [v for v in state["current_variables"] if v not in [d.variable_name for d in state.get("input_data", [])]]
        if remaining:
            summaries.append("\nUnprocessed variables:\n" + "\n".join(remaining))
    
    return "\n\n".join(summaries) if summaries else "No dataset descriptions available."


# ---------------------------
# Routing logic
# ---------------------------
def route_to_tools(state: AgentState) -> Literal["tools", "__end__"]:
    if not state.get("messages"):
        raise ValueError("No messages found in state for routing")

    print("\n🔹 [DEBUG] route_to_tools called")
    print("Current state keys:", list(state.keys()))
    last_msg = state["messages"][-1]
    print("Last message type:", type(last_msg).__name__)
    if hasattr(last_msg, "content"):
        print("Last message content:", last_msg.content[:300])
    print("---------------------------------------------")

    ai_message = state["messages"][-1]
    if isinstance(ai_message, AIMessage) and getattr(ai_message, "tool_calls", None):
        print("🧭 Route decision: → tools\n")
        return "tools"
    print("🧭 Route decision: → __end__\n")
    return "__end__"


# Debugging: Token Usage


def debug_message_summary(messages, model_obj=None):
    """
    Compact summary of messages being sent to the model.
    Prints token count per message and total estimated tokens.
    """
    # 🧩 Try to infer tokenizer from actual model
    model_name = getattr(model_obj, "model_name", None) or getattr(model_obj, "_model", None) or "gpt-4o"
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    print("\n🔍 [DEBUG] Messages about to be sent to model")
    print("─────────────────────────────────────────────")
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", getattr(msg, "role", "unknown"))
        content = getattr(msg, "content", "")
        snippet = (content[:120] + "…") if len(content) > 120 else content
        tokens = len(enc.encode(content or ""))
        total_tokens += tokens
        print(f"{i+1:>2}. {role:<10} | {tokens:>6} tokens | {snippet}")
    print("─────────────────────────────────────────────")
    print(f"Total estimated tokens → {total_tokens:,}\n")
    return total_tokens


def dump_full_message_json(messages, path="debug_messages.json"):
    """
    Optional: Save the message contents to a file for deep inspection.
    """
    data = []
    for m in messages:
        data.append({
            "role": getattr(m, "type", getattr(m, "role", "unknown")),
            "has_tool_calls": bool(getattr(m, "tool_calls", None)),
            "content_preview": (m.content[:1000] + "…") if len(m.content) > 1000 else m.content
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Full message dump written to {path}")

# ---------------------------------------------------------------------
# CALL MODEL NODE 
# ---------------------------------------------------------------------
def call_model(state: AgentState):
    print("\n🟦 [DEBUG] call_model START")

    all_messages = state.get("messages", [])

    # ✅ Keep only relevant message types
    messages = [m for m in all_messages if isinstance(m, (HumanMessage, AIMessage, ToolMessage))]

    # ✅ Ensure valid ordering: remove ToolMessages without a preceding AIMessage with tool_calls
    cleaned_messages = []
    last_ai_had_tool_calls = False
    for m in messages:
        if isinstance(m, AIMessage):
            cleaned_messages.append(m)
            last_ai_had_tool_calls = bool(getattr(m, "tool_calls", None))
        elif isinstance(m, ToolMessage):
            if last_ai_had_tool_calls:
                cleaned_messages.append(m)
            else:
                print("⚠️ Skipping stray ToolMessage (no preceding tool_calls)")
        else:
            cleaned_messages.append(m)

    # ✅ Limit history size to reduce token load
    messages = cleaned_messages[-6:]

    file_path = state.get("file_path", "")
    input_data = state.get("input_data", [])

    # ---------------------------------------------------------------
    # 🧹 Clean up any massive or malformed ToolMessage content
    # ---------------------------------------------------------------
    for m in messages:
        if isinstance(m, ToolMessage):
            if isinstance(m.content, tuple):
                print("⚠️ Detected tuple ToolMessage → keeping only first element (short output)")
                m.content = str(m.content[0])
            elif isinstance(m.content, dict):
                print("⚠️ Detected dict ToolMessage → replacing with summary marker")
                m.content = "[tool state omitted]"
            # Truncate any long string content
            if isinstance(m.content, str) and len(m.content) > 1200:
                print(f"⚠️ Truncating long ToolMessage content ({len(m.content)} chars)")
                m.content = m.content[:800] + "... [truncated for model]"

    # ---------------------------------------------------------------
    # 🩺 Fix invalid ordering for OpenAI chat schema
    # ---------------------------------------------------------------
    # Drop any leading ToolMessage without an AIMessage before it
    while messages and isinstance(messages[0], ToolMessage):
        print("⚠️ Dropping leading ToolMessage (no preceding tool_calls)")
        messages.pop(0)

    # Recheck sequence validity — only allow ToolMessages that follow AIMessage w/ tool_calls
    valid_sequence = []
    expecting_tool_response = False
    for m in messages:
        if isinstance(m, AIMessage):
            valid_sequence.append(m)
            expecting_tool_response = bool(getattr(m, "tool_calls", None))
        elif isinstance(m, ToolMessage):
            if expecting_tool_response:
                valid_sequence.append(m)
                expecting_tool_response = False
            else:
                print("⚠️ Skipping unpaired ToolMessage inside history")
        else:
            valid_sequence.append(m)
            expecting_tool_response = False
    messages = valid_sequence

    # ---------------------------------------------------------------
    # Add dataset context once
    # ---------------------------------------------------------------
    if not state.get("context_added", False) and input_data:
        dataset = input_data[0]
        context_summary = (
            f"You have access to dataset `{os.path.basename(file_path)}`.\n"
            f"Description: {dataset.data_description[:500]}"
        )
        messages.insert(0, HumanMessage(content=context_summary))
        state["context_added"] = True

    print(f"📦 Valid messages passed to model: {len(messages)}")

    # ---------------------------------------------------------------
    # 🧠 DEBUG: token usage summary BEFORE calling model
    # ---------------------------------------------------------------
    total_tokens = debug_message_summary(messages, model_obj=model)
    if total_tokens > 200000:
        print(f"⚠️ WARNING: Very large payload ({total_tokens:,} tokens). May exceed model limit!")
        dump_full_message_json(messages)

    # ---------------------------------------------------------------
    # 🧩 Actual model call
    # ---------------------------------------------------------------
    try:
        ai_message = model.invoke({"messages": messages})
        print("✅ Model invocation complete.")
    except Exception as e:
        ai_message = AIMessage(content=f"⚠️ Model call failed: {e}")
        print(f"❌ Model invocation failed: {e}")

    return {
        **state,
        "messages": [ai_message],
        "context_added": True,
        "output_image_paths": state.get("output_image_paths", []),
        "intermediate_outputs": state.get("intermediate_outputs", []),
    }

# ---------------------------------------------------------------------
# 🛠️ TOOL NODE HANDLER (
# ---------------------------------------------------------------------
def call_tools(state: AgentState):
    print("\n🟩 [DEBUG] call_tools START")

    last_ai_message = state["messages"][-1]
    if not isinstance(last_ai_message, AIMessage) or not getattr(last_ai_message, "tool_calls", None):
        print("⚠️ No tool calls found — skipping ToolNode.")
        return state

    tool_messages = []
    merged_updates = {}

    for tc in last_ai_message.tool_calls:
        tool_name = tc["name"]
        tool_id = tc["id"]
        print(f"🔧 Executing tool: {tool_name}")

        try:
            if tool_name == "complete_python_task":
                thought = tc.get("args", {}).get("thought", "")
                python_code = tc.get("args", {}).get("python_code", "")

                # ✅ Call your tool directly and unpack the tuple
                result_text, updated_state = complete_python_task(
                    graph_state=state,
                    thought=thought,
                    python_code=python_code,
                )

                # ✅ Ensure updated_state is always a dict
                if not isinstance(updated_state, dict):
                    updated_state = {}

                # ✅ Merge state updates (UI data, images, reasoning)
                for k, v in updated_state.items():
                    if k in merged_updates and isinstance(v, list):
                        merged_updates[k].extend(v)
                    else:
                        merged_updates[k] = v

                # ✅ Add success ToolMessage
                tool_messages.append(
                    ToolMessage(
                        content=result_text,
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )

            else:
                print(f"⚠️ Unknown tool: {tool_name}")
                tool_messages.append(
                    ToolMessage(
                        content=f"⚠️ Unknown tool: {tool_name}",
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )

        except Exception as e:
            print(f"❌ Tool execution failed: {e}")
            # Always build a fallback ToolMessage
            tool_messages.append(
                ToolMessage(
                    content=f"⚠️ Tool execution failed: {e}",
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            )

    # ✅ Merge all updates from the tool back into state
    state["messages"].extend(tool_messages)
    for k, v in merged_updates.items():
        if k not in ["messages", "input_data"]:
            state[k] = v

    print("🧩 [DEBUG] merged_updates keys:", list(merged_updates.keys()))
    print("🧩 [DEBUG] output_image_paths after merge:", state.get("output_image_paths", []))
    print("🧩 [DEBUG] intermediate_outputs after merge:", len(state.get("intermediate_outputs", [])))

    return state
