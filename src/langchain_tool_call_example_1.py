from langchain_ollama.llms import OllamaLLM
from langchain.agents import initialize_agent, Tool
from langchain_core.prompts import ChatPromptTemplate


def get_current_weather(city: str) -> str:
    # Dummy implementation
    return f"The weather in {city} is sunny."


template = """Question: {question}

Answer: Let's think step by step.
"""

prompt = ChatPromptTemplate.from_template(template)


llm = OllamaLLM(model="llama3.2")

tools = [
    Tool(
        name="get_current_weather",
        func=get_current_weather,
        description="Get the current weather for a city",
    )
]


agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
)


chain = prompt | agent

response = chain.invoke({"question": "What is the weather in Toronto?"})

# response = agent.run("What is the weather in Toronto?")
print(response)
