https://ollama.com/library/gemma3
https://ollama.com/blog/tool-support
https://python.langchain.com/docs/integrations/llms/ollama/


# Minimum-to-Working Chat with **Ollama**

Keep this playbook open in a terminal tab while you experiment—or jump straight to the **⌨️ Quickstart**.

---

## 1 · Install Ollama

| OS | One-liner | Notes |
| --- | --- | --- |
| **macOS ≥ 11** | `brew install ollama`  (or `brew install --cask ollama` for the GUI bundle) | Adds the `ollama` CLI to `/usr/local/bin` and a small helper app. |
| **Linux (x86-64 / ARM64)** | `curl -fsSL https://ollama.com/install.sh \| sh` | Script drops a static binary in `/usr/local/bin` and sets up a systemd service. |
| **Windows 10 + (Preview)** | 1. Download **OllamaSetup.exe** →<br>2. double-click →<br>3. `ollama run llama2` to verify | GPU drivers only—no CUDA toolkit required; NVIDIA acceleration is enabled automatically. |
| **Docker (any OS)** | `docker run -d --name ollama -p 11434:11434 ollama/ollama` | Ideal for CI or WSL2. |

<details>
<summary>Where models live</summary>

Models are stored in `~/.ollama` (or `%USERPROFILE%\.ollama` on Windows), so you can wipe and reinstall Ollama without losing downloads.
</details>

---

## 2 · ⌨️ Quickstart (works on every platform)

```bash
# ➊ Launch an interactive chat with Llama 3.2
ollama run llama3.2

# ➋ Ask a one-off question from any shell script
ollama run llama3.2 "Explain the Doppler effect in one tweet"

# ➌ List and manage models
ollama list          # what’s installed
ollama pull phi4     # download a different model
ollama rm phi4       # delete it
```
---

## 3 · System requirements & performance tips

- RAM / VRAM rule-of-thumb – 7 B → 8 GB RAM, 13 B → 16 GB, 33 B → 32 GB.
Models load partly into GPU VRAM if available, otherwise into system memory.

- On Windows, Ollama auto-detects NVIDIA GPUs and AVX/AVX2 CPUs—no extra setup.

- To run Ollama as a background service without the desktop app, execute `ollama serve` (or `brew services start ollama` on macOS).

---

## 4 · Using the local REST API

The installer starts a service on http://localhost:11434 that is OpenAI-compatible.

```bash
# Generate a plain completion
curl http://localhost:11434/api/generate -d '{
  "model":"llama3.2",
  "prompt":"Why is the sky blue?",
  "stream":false
}'

# Structured chat
curl http://localhost:11434/api/chat -d '{
  "model":"llama3.2",
  "messages":[{"role":"user","content":"List 3 facts about Jupiter"}]
}'
```

Drop-in libraries exist for Python (`pip install ollama`), JavaScript, Go, and more—the API surface mirrors OpenAI’s `/v1/chat/completions`, so most tooling (LangChain, LlamaIndex, etc.) works by simply swapping the base-URL.

---

## 5 · Customising & adding models

1. Import GGUF / Safetensors
```bash
echo 'FROM ./vicuna-33b.Q4_0.gguf' > Modelfile
ollama create vicuna-local -f Modelfile
```

2. Import GGUF / Safetensors
```bash
FROM llama3.2
PARAMETER temperature 0.7
SYSTEM """You are a pirate ship’s parrot. Reply in squawks."""
```

3. Import GGUF / Safetensors
```bash
ollama create pirate -f Modelfile
ollama run pirate
```

## Common gotchas

1. “Not enough memory to allocate …”
Use a smaller parameter count (e.g. `:1b` variant) or pass `--numa` on Linux to pin to one NUMA node.

2. CPU-only run is slow	
Ensure GPU driver is recent. (Windows preview bundles CUDA runtime; Linux needs a matching NVIDIA driver—no toolkit.)

3. Port 11434 already in use	
Stop the desktop app with `ollama quit`, or change the port: `OLLAMA_HOST=0.0.0.0:11500 ollama serve`.

## Final

Open two terminals—one with `ollama run llama3.2`, the other hitting the REST endpoint—and you’ve got a fully offline LLM playground.