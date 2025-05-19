from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage


@tool
def get_job_title(user_name: str) -> str:
    """Get the job title for a given user.

    Args:
        user_name (str): The name of the user.

    Returns:
        str: The job title.
    """
    if user_name.lower() == "bob":
        return "Astronaut"
    else:
        return "Unknown"


@tool
def get_user_name() -> str:
    """Get the user's name.

    Returns:
        str: The user's name.
    """
    return "Bob"


tools = [get_job_title, get_user_name]
tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        try:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                )
            )
        except Exception as e:
            result.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                )
            )
    return result


state: dict = {"messages": []}


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a highly capable AI assistant with access to external tools. "
        "Before answering, carefully analyze the user's request and determine if you need more information. "
        "If you lack required information, call the appropriate tool to obtain it. "
        "Call only one tool at a time unless multiple are absolutely necessary. "
        "Never make up information or assume unknown facts. "
        "Wait for tool results before proceeding to the next step or providing a final answer. "
        "Chain tool calls logically if the answer requires multiple steps. "
        "Your final answers must be concise, accurate, and directly address the user's question. "
        "Emulate the robust, step-by-step reasoning and tool usage of advanced AI systems."
    )
)

model_names = [
    "llama3.3:70b",
    "llama3-groq-tool-use:8b",
    "llama3.1:8b",
    "llama3.2:latest",
]

model_name = model_names[0]

model = ChatOllama(model=model_name)

model_with_tool = model.bind_tools(
    tools=[get_job_title, get_user_name],
)


def invoke(state: dict, max_iterations: int = 10):
    for _ in range(max_iterations):
        response = model_with_tool.invoke(state["messages"])

        print("LLM Response:", repr(response))

        state["messages"].append(response)

        # if response.tool_calls and len(response.tool_calls) > 1:
        #     response.tool_calls = [response.tool_calls[0]]

        if not response.tool_calls:
            return state

        tool_results = tool_node(state)

        print("Tool Response:", repr(tool_results))

        state["messages"].extend(tool_results)

    raise RuntimeError("Max iterations reached without a final answer.")


HumanMessage(content="What is my name?")
state["messages"].extend(
    [
        SYSTEM_PROMPT,
        HumanMessage(content="What is my name, and then tell me my job title?"),
    ]
)

state = invoke(state)

print("====================")
for msg in state["messages"]:
    print(repr(msg), "\n")
