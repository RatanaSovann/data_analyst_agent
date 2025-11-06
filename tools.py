from io import StringIO
import sys, os, uuid
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Annotated, Tuple, List, Dict
from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_experimental.utilities import PythonREPL
import pickle
from datetime import datetime
from matplotlib import font_manager, rcParams


# Persistent runtime vars across calls
persistent_vars = {}

# Utility function for saving Plotly figures
def save_plotly_figures(figures, output_dir="images/plotly_figures/pickle"):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for fig in figures:
        if not isinstance(fig, go.Figure):
            continue
        unique_id = str(uuid.uuid4())
        filename = f"{datetime.now():%Y%m%d_%H%M%S}_{unique_id}.pickle"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as f:
            pickle.dump(fig, f)
        saved_paths.append(filename)   # 👈 basename only
    return saved_paths


@tool(parse_docstring=True)
def complete_python_task(
    graph_state: Annotated[dict, InjectedState],
    thought: str,
    python_code: str,
) -> Tuple[str, dict]:
    """
    Executes Python analysis or visualization code in a controlled environment.

    Args:
        graph_state (dict): The current agent state, including file paths and persisted variables.
        thought (str): Model’s reasoning about what it’s about to do.
        python_code (str): The Python code string to execute.

    Returns:
        Tuple[str, dict]: A short string summary for the model, and an updated state dictionary for the UI.
    """

    # ---------------------------------------------------
    # 1️⃣ Prepare execution context
    # ---------------------------------------------------
    current_variables = graph_state.get("current_variables", {}).copy()
    file_path = graph_state.get("file_path")

    if file_path and os.path.exists(file_path):
        file_name = os.path.basename(file_path)
        if file_name not in current_variables:
            try:
                if file_path.endswith(("xlsx", "xls")):
                    current_variables[file_name] = pd.read_excel(file_path)
                elif file_path.endswith("csv"):
                    current_variables[file_name] = pd.read_csv(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
            except Exception as e:
                print(f"⚠️ Error loading dataset from {file_path}: {e}")
    else:
        print("⚠️ No valid file_path provided in graph_state.")

       # ---------------------------------------------------
    # 2️⃣ Execute code safely
    # ---------------------------------------------------
    os.makedirs("images/plotly_figures/pickle", exist_ok=True)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    updated_state = {}   # ✅ ensure it always exists

    try:
        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "plotly_figures": [],
            **persistent_vars,
            **current_variables,
        }

        exec(python_code, exec_globals)
        output = sys.stdout.getvalue().strip()
    finally:
        sys.stdout = old_stdout

    # ---------------------------------------------------
    # 3️⃣ Save new figures (Plotly + Matplotlib PNGs)
    # ---------------------------------------------------
    new_files = []
    if exec_globals.get("plotly_figures"):
        new_files = save_plotly_figures(exec_globals["plotly_figures"])
        persistent_vars["plotly_figures"] = []

    # ✅ Move PNGs to dedicated folder
    png_dir = "images/static_plots"
    os.makedirs(png_dir, exist_ok=True)

    png_files = []
    for f in os.listdir():
        if f.lower().endswith(".png"):
            new_path = os.path.join(png_dir, f)
            try:
                shutil.move(f, new_path)
                # store only relative path (cleaner for frontend)
                png_files.append(os.path.relpath(new_path, start=os.getcwd()))
            except Exception as e:
                print(f"⚠️ Could not move {f}: {e}")

    # ---------------------------------------------------
    # 4️⃣ Update persistent vars
    # ---------------------------------------------------
    persistent_vars.update({
        k: v for k, v in exec_globals.items()
        if not k.startswith("__") and k not in ["pd", "plotly_figures"]
    })

    # ---------------------------------------------------
    # 5️⃣ Dual output: full for UI, short for model
    # ---------------------------------------------------
    full_output = output or "(No printed output)"
    if len(full_output) > 2500:
        full_output = full_output[:2500] + "\n...[truncated in tool]..."

    safe_output = (
        full_output[:800]
        + ("... [truncated for model]" if len(full_output) > 800 else "")
    )

    # ✅ Unified state object
    updated_state = {
        "intermediate_outputs": [
            {
                "tool": "complete_python_task",
                "thought": thought[:600] + ("..." if len(thought) > 600 else ""),
                "code": python_code[:1500] + ("..." if len(python_code) > 1500 else ""),
                "output": full_output,   # full for UI
            }
        ],
        "current_variables": persistent_vars,
    }

    # ✅ Combine all new image files (plotly + static)
    all_images = new_files + png_files
    if all_images:
        updated_state["output_image_paths"] = all_images

    return safe_output, updated_state





@tool("data_view")

def data_view(file_path: str) -> str:
    """
    Load a single CSV or Excel file from the given file path and return a preview.
    """

    try:
        # Step 2: Load dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return f"❌ Unsupported file type: {file_path}"
    
        # Step 3: Create a simple preview
        preview = df.head(5).to_markdown(index=False)
        info = f"✅ Loaded `{os.path.basename(file_path)}` successfully.\n\n**Preview:**\n{preview}"
        return info

    except Exception as e:
        return f"❌ Error while viewing dataset: {e}"



