# 🧠 Data Analyst AI Agent  
**Powered by LangGraph, Streamlit, and Agentic AI Workflows**

---

## 📖 Overview
An intelligent **AI Data Analyst** that performs data analysis tasks automatically — summarizing uploaded datasets, generating executable Python code, and producing interactive charts.  
It uses **LangGraph** for reasoning, **Streamlit** for the chat interface, and a tool-based architecture for executing data tasks safely.

---

## 🚀 Features
- Automatic dataset summarization  
- Natural language → Python code generation  
- Safe in-memory execution (pandas, numpy, matplotlib)  
- Dynamic visualization rendering  
- Persistent context between chat turns  
- Built-in debugging and observability tools  

---

## 🧩 Architecture
```
User Input (HumanMessage)
      ↓
Reasoning Node (call_model)
      ↓
Tool Execution Node (call_tools)
      ↓
Python Tool (complete_python_task)
      ↓
Streamlit UI (Text + Charts)
```

| Component | Description |
|------------|--------------|
| **LangGraph** | Handles reasoning and tool routing |
| **Streamlit** | Provides interactive front-end |
| **Tools** | `data_view` (dataset parser) and `complete_python_task` (Python executor) |
| **PythonChatbot** | Controls the reasoning–execution loop and manages session state |

---

## 🗂️ Repository Structure
```
.
├── app.py                     # Streamlit entry point
├── backend.py                 # Defines PythonChatbot (graph orchestration)
├── workflow.py                # LangGraph nodes and routing (call_model, call_tools)
├── tools.py                   # Tool functions (data_view, complete_python_task)
├── helper_function.py         # Debug utilities (pretty_print)
├── test.py                    # Testing the agent reasoning flow
├── cafe.xlsx                  # Sample dataset
├── hourly_sales_by_hour.html  # Example visualization output
├── debug_messages.json        # Message trace logs
├── requirements.txt           # Dependencies
├── uploads/                   # Uploaded datasets
├── images/
│   ├── static_plots/          # Saved PNG charts
│   └── plotly_figures/pickle/ # Saved Plotly figures
└── README.md
```

---

## ⚙️ Installation
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/data-analyst-agent.git
cd data-analyst-agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Application
```bash
streamlit run app.py
```

Then open the displayed local URL (e.g. `http://localhost:8501`) in your browser.

---

## 💬 How It Works
1. **Upload your dataset** on **Tab 1**.  
   The agent summarizes its structure and key statistics.  
2. **Chat with the AI Analyst** on **Tab 2**.  
   Ask analytical questions in plain English (e.g. “Show me top 10 items by total sales”).  
3. The model:  
   - Generates a Python code plan  
   - Executes it using the `complete_python_task` tool  
   - Returns results + visualizations directly in chat  

---

## 🧠 Example Interaction

**User:**  
> “Summarize the dataset and show average quantity by item.”

**AI Analyst:**  
> “The dataset contains 63 874 rows across 102 unique products.  
> Here’s the average quantity sold per item.”  
> *(Bar chart rendered inline)*  

**User:**  
> “Now plot monthly sales trend for 2024.”

**AI Analyst:**  
> “Below is the monthly trend of total sales volume.”  
> *(Line chart displayed)*

---

## 🧩 Debugging & Observability
The project includes built-in debugging utilities:

- **`debug_message_summary()`** – token count per message + total usage  
- **`dump_full_message_json()`** – message log saved as JSON  
- **Console logs in `user_sent_message()`** – summarize input state & returned results  

These make it easy to catch malformed messages, token bloat, or missing chart outputs.

---

## 🏁 Results
- Fully functional end-to-end **AI data analysis pipeline**.  
- Interactive chat workflow for dataset exploration.  
- Real-time chart generation and inline rendering in Streamlit.  
- Safe execution environment and clear state observability.  

---

## 🔮 Future Extensions
- Add SQL-query generation agent  
- Integrate AutoViz or Seaborn for richer visuals  
- Extend toolset for multi-file or time-series analysis  
- Deploy via FastAPI backend for multi-user sessions  

---

**Author:** Ratana Sovann  
**Project:** Data Analyst AI Agent using LangGraph and Agentic AI Workflows  
**License:** MIT (or specify your preferred license)
