import requests
import json

# Base URL for Ollama
BASE_URL = "http://localhost:11434"


def plain_completion(model: str, prompt: str):
    """
    /api/generate: This endpoint is for plain text completions.
    You provide a single prompt, and the model generates a single response.
    It does not maintain any conversational context between calls.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(f"{BASE_URL}/api/generate", json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("completion") or data


def chat_completion(model: str, messages: list[dict]):
    """
    /api/chat: This endpoint is for chat-based interactions.
    You provide a list of messages (with roles like "system", "user", "assistant"),
    and the model generates a response considering the entire conversation history.
    This allows for context-aware, multi-turn conversations.
    """
    payload = {"model": model, "messages": messages}
    response = requests.post(f"{BASE_URL}/api/chat", json=payload)
    response.raise_for_status()
    # data = response.json()
    return response.content.decode("utf-8")


if __name__ == "__main__":
    # Example 1: Plain completion
    answer = plain_completion("llama3.2", "Explain the Doppler effect in one tweet")
    print("Plain completion:", answer)

    # Example 2: Chat completion
    chat_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "List 3 facts about Jupiter"},
    ]
    chat_answer = chat_completion("llama3.2", chat_msgs)
    print("Chat completion:", chat_answer)
