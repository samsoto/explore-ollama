from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3.2")


model_with_tool = model.bind_tools(
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
    ]
)

chain = prompt | model_with_tool

response = chain.invoke({"question": "What is the weather in Toronto?"})

print(response)
