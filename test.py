from backend import PythonChatbot, InputData
from langchain_core.messages import HumanMessage

# 1️⃣ Create chatbot and graph
chatbot = PythonChatbot()
graph = chatbot.graph  # access the compiled LangGraph

# 2️⃣ Prepare the dataset info (same as before)
file_path = "C:/Users/sovan/Desktop/data_analyst_agent - Copy/uploads/cafe.xlsx"

input_data_list = [
    InputData(
        variable_name="cafe",
        data_path=file_path,
        data_description=(
            "This dataset contains point-of-sale transaction records for a café, including timestamped receipts, "
            "item-level descriptions, and monetary fields (gross sales, discounts, and net sales). It is suitable "
            "for analyzing sales volumes, revenue over time, discount impacts, and item purchase patterns.\n\n"
            "### Columns:\n"
            "- Date (datetime): Timestamp of the transaction (date and time when the sale occurred).\n"
            "- Receipt number (string): Unique identifier for each transaction/receipt.\n"
            "- Gross sales (numeric (integer/currency)): Total amount charged before any discounts for the transaction.\n"
            "- Discounts (numeric (integer/currency)): Total discount amount applied to the transaction.\n"
            "- Net sales (numeric (integer/currency)): Amount received after discounts (Gross sales minus Discounts).\n"
            "- Description (string): Text description of items purchased in the transaction (quantities and product names)."
        ),
    )
]

# 3️⃣ Manually recreate what user_sent_message() would normally build
state = {
    "messages": [HumanMessage(content="Plot the top 10 most popular item by quantity sold in 2024")],
    "input_data": input_data_list,
    "file_path": file_path,
    "current_variables": {},
    "intermediate_outputs": [],
    "output_image_paths": [],
}

# 4️⃣ DEBUG: print the initial state
print("\n🟩 [TEST] Initial state before invoking graph:")
for k, v in state.items():
    if isinstance(v, list):
        print(f"  - {k}: list[{len(v)}]")
    else:
        print(f"  - {k}: {v}")

# 5️⃣ Directly call the LangGraph (bypassing the wrapper)
result = graph.invoke(state, {"recursion_limit": 25})

# 6️⃣ Inspect the returned results
print("\n🟦 [RESULT] Keys in returned state:", list(result.keys()))
print("Message count:", len(result.get("messages", [])))

print("\n---- Conversation ----")
for msg in result["messages"]:
    print(f"{msg.type.upper()}: {msg.content}")

# 7️⃣ If you want to test node-by-node, you can also invoke the model directly:
from backend import call_model
model_result = call_model(state)
print(model_result["messages"][0].content)
