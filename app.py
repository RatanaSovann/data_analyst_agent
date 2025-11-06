import pickle
import streamlit as st
import os
import pandas as pd
import json
from langchain_core.messages import HumanMessage, AIMessage
from backend import PythonChatbot
from workflow import summary_agent, InputData

# -----------------------------------------------------------------
# 🧩 PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("🤖 AI Data Analysis Agent")

os.makedirs("uploads", exist_ok=True)
os.makedirs("images/plotly_figures/pickle", exist_ok=True)

# -----------------------------------------------------------------
# 🧠 INITIALIZE SESSION STATE
# -----------------------------------------------------------------
if "chatbot" not in st.session_state:
    st.session_state.chatbot = PythonChatbot()

if "schema_summary" not in st.session_state:
    st.session_state.schema_summary = {}

chatbot = st.session_state.chatbot

# -----------------------------------------------------------------
# 🧭 LAYOUT
# -----------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📂 Data Management", "💬 Chat Interface", "🧰 Debug Information"])

# ===============================================================
# TAB 1: 📂 Upload & Summarization
# ===============================================================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a dataset (CSV, XLS, XLSX)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        file_path = os.path.abspath(os.path.join("uploads", uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        chatbot.file_path = file_path
        st.session_state.chatbot = chatbot
        st.subheader(f"📊 Preview: {uploaded_file.name}")

        # Try previewing
        try:
            df = pd.read_csv(file_path) if uploaded_file.name.endswith(".csv") else pd.read_excel(file_path)
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

        # Summarize dataset via summary agent
        with st.spinner("🔍 Generating dataset summary..."):
            try:
                result = summary_agent.invoke({
                    "file_path": file_path,
                    "messages": [HumanMessage(content=f"Summarize dataset in Streamlit JSON format: {file_path}")]
                })
                output_text = result["messages"][-1].content
                output_json = json.loads(output_text)
                st.session_state.schema_summary = output_json

                cols_df = pd.DataFrame(output_json.get("columns", []))
                st.markdown("### 🧾 Column Summary")
                st.table(cols_df)

                st.markdown("### 📌 Dataset Overview")
                st.markdown(output_json.get("summary", "No summary provided."))
                st.success("✅ Schema summary stored for agent use.")

            except Exception as e:
                st.error(f"❌ Error generating summary: {e}")

# ===============================================================
# TAB 2: 💬 Chat Interface
# ===============================================================
with tab2:
    st.subheader("Chat with the Analysis Agent")

    # Show which file is active
    if chatbot.file_path and os.path.exists(chatbot.file_path):
        st.caption(f"**Active file:** `{os.path.basename(chatbot.file_path)}`")
        st.success("✅ File found")
    else:
        st.warning("⚠️ No valid file loaded yet. Please upload a dataset first.")

    # --- Compact dataset description for token control
    schema_summary = st.session_state.get("schema_summary", {})
    dataset_description = ""
    if schema_summary:
        cols = schema_summary.get("columns", [])[:10]
        summary_text = schema_summary.get("summary", "No summary available.")
        col_lines = "\n".join(
            f"- {c.get('Column', 'Unknown')} ({c.get('Type', 'Unknown')})"
            for c in cols if isinstance(c, dict)
        )
        dataset_description = f"{summary_text}\n\n### Columns (sample):\n{col_lines}"

    # --- Display chat messages
    chat_container = st.container(height=550)
    with chat_container:
        for i, msg in enumerate(chatbot.chat_history):
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
                    # Inline visualizations (Plotly or static)
                    if i in chatbot.output_image_paths:
                       for path in chatbot.output_image_paths:
                           # Normalize path
                           ext = os.path.splitext(path)[1].lower()
                           base_dir = ""
                    
                           if ext == ".pickle":
                               base_dir = "images/plotly_figures/pickle"
                           elif ext in [".png", ".jpg", ".jpeg"]:
                               base_dir = "images/static_plots"
                    
                           # Build full path
                           full_path = (
                               os.path.join(base_dir, os.path.basename(path))
                               if not os.path.isabs(path)
                               else path
                           )
                    
                           # ✅ Display Plotly or PNG/JPG output
                           if os.path.exists(full_path):
                               if ext == ".pickle":
                                   try:
                                       with open(full_path, "rb") as f:
                                           fig = pickle.load(f)
                                       st.plotly_chart(fig, use_container_width=True)
                                   except Exception as e:
                                       st.warning(f"⚠️ Could not render Plotly figure: {e}")
                    
                               elif ext in [".png", ".jpg", ".jpeg"]:
                                   st.image(full_path, use_container_width=True)

    # --- Input box for chat
    def on_submit():
        user_query = st.session_state["user_query"].strip()
        if not user_query:
            st.warning("Please type a message.")
            return

        # Prepare compact dataset context
        input_data_list = []
        if chatbot.file_path and dataset_description:
            input_data_list = [
                InputData(
                    variable_name=os.path.splitext(os.path.basename(chatbot.file_path))[0],
                    data_path=chatbot.file_path,
                    data_description=dataset_description[:1200],
                )
            ]

        chatbot.user_sent_message(user_query, input_data=input_data_list)
        st.session_state.chatbot = chatbot
        st.session_state["user_query"] = ""  # clear input box

    st.chat_input(
        placeholder="Ask a question about your dataset...",
        key="user_query",
        on_submit=on_submit,
    )


# ===============================================================
# TAB 3: 🧰 Debug / Reasoning Trace
# ===============================================================
with tab3:
    st.subheader("🧠 Agent Reasoning Trace")

    outputs = chatbot.intermediate_outputs
    st.write(outputs)
    if not outputs:
        st.info("No debug steps yet — start chatting to see reasoning trace.")
    else:
        for idx, step in enumerate(outputs):
            with st.expander(f"Step {idx+1}"):
                if isinstance(step, dict):
                    # 🧩 Thought process
                    if step.get("thought"):
                        st.markdown("### 💭 Thought Process")
                        st.markdown(step["thought"])

                    # 🧮 Code execution
                    if step.get("tool") == "complete_python_task":
                        if step.get("code"):
                            st.markdown("### 🧩 Code Executed")
                            st.code(step["code"], language="python")
                        if step.get("output"):
                            st.markdown("### 🖥️ Output")
                            st.text(step["output"])
                    else:
                        # Other tools
                        if step.get("code"):
                            st.markdown(f"**Tool Used:** `{step['code']}`")
                        if step.get("output"):
                            st.text(step["output"])
                else:
                    st.text(str(step))