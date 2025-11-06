import json
from langchain.schema import HumanMessage, AIMessage

def pretty_print(messages):
    """
    Pretty-print LangGraph messages (HumanMessage, AIMessage).
    """
    if not messages:
        print("⚠️ No messages to display")
        return

    for msg in messages:
        # Human message
        if isinstance(msg, HumanMessage):
            print("="*15 + " Human Message " + "="*15)
            print(msg.content, "\n")

        # AI message
        elif isinstance(msg, AIMessage):
            print("="*15 + " AI Message " + "="*15)
            if getattr(msg, "name", None):
                print(f"Agent Name: {msg.name}\n")

            tool_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls", [])
            if tool_calls:
                print("🔧 Tool Calls:")
                for tool in tool_calls:
                    tool_name = tool.get("function", {}).get("name", "unknown_tool")
                    tool_id = tool.get("id", "no_id")
                    args = tool.get("function", {}).get("arguments", {})

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {"raw_args": args}

                    print(f"  - Tool: {tool_name}")
                    print(f"    Call ID: {tool_id}")
                    if args:
                        print("    Args:")
                        for k, v in args.items():
                            print(f"      {k}: {v}")
                print()

            if msg.content:
                print("Response:\n")
                print(msg.content, "\n")

        else:
            print(f"⚠️ Unknown message type: {type(msg)}")
            print(msg)
