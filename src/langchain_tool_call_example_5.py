from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama


@tool
def get_job_title(user_name: str) -> str:
    """Get the job title for a given person.

    Args:
        user_name (str): The name of the person.

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


system_prompt = """You are a helpful ai assistant.
You are equipped with various tools to assist users in obtaining information.
Some tools are dependent on the output of other tools.
Focus on calling a single tool at a time.
If a tool is called, wait for the result before calling another tool.
Don't call a tool again if you already have the information.
The final result should be a concise answer to the user's question.
"""

models = [
    "llama3.3:70b",
    "llama3-groq-tool-use:8b",
    "llama3.1:8b",
    "llama3.2:latest",
]

model = models[2]

llm = ChatOllama(model=model, temperature=0.0)

print("Using model:", model)

tools = [get_user_name, get_job_title]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is my name and job title?"}]},
)

for msg in result["messages"]:
    print("--------------------")
    print(repr(msg), "\n")

print("====================")
print(result["messages"][-1].content)
