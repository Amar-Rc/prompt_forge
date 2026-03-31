# prompt-forge

Transform raw, sloppy prompts into production-quality prompts using any LLM. Rewrites your prompt with a system role, structured output format, XML delimiters, and few-shot example placeholders — instantly.

## Features

- Rewrites prompts using prompt engineering best practices
- Supports **Anthropic**, **OpenAI**, **Gemini**, **Groq**, **Ollama**, **MLX**, and any OpenAI-compatible endpoint
- Optional domain-aware rewrites (e.g. "data engineering", "support triage")
- `--compare` flag to run original vs. rewritten prompt side by side
- Prompt versioning — save and review past rewrites

## Installation

```bash
git clone https://github.com/Amar-Rc/prompt_forge.git
cd prompt_forge
pip install -r requirements.txt
```

Copy the environment file and add your API keys:

```bash
cp .env.example .env
# edit .env and fill in the keys for the providers you use
```

## Usage

```bash
python prompt_forge.py "your raw prompt here"
```

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--provider NAME` | `-p` | LLM provider (default: `anthropic`) |
| `--model MODEL` | `-m` | Model override (defaults to provider's default) |
| `--domain DOMAIN` | `-d` | Domain context for the rewrite |
| `--compare` | `-c` | Run original and rewritten prompt side by side |
| `--sample-input TEXT` | `-i` | Sample input to use with `--compare` |
| `--save NAME` | `-s` | Save the rewritten prompt to `prompt_versions.json` |
| `--list-versions` | `-l` | List all saved prompt versions |
| `--show-version ID` | | Show a saved version by ID |
| `--list-providers` | | List providers and API key status |
| `--base-url URL` | | Base URL for custom OpenAI-compatible endpoints |
| `--api-key KEY` | | API key override (default: read from environment) |

### Examples

```bash
# Default — rewrites using Anthropic Claude
python prompt_forge.py "look at this data and find problems"

# Use OpenAI GPT-4o
python prompt_forge.py --provider openai "find bugs in this code"

# Use Groq for fast inference
python prompt_forge.py --provider groq --model llama-3.3-70b-versatile "classify this ticket"

# Domain-aware rewrite
python prompt_forge.py --domain "data engineering" "find issues in the pipeline"

# Compare original vs rewritten on a sample input
python prompt_forge.py --compare --sample-input "col1,col2\n1,null" "find data problems"

# Save a named version
python prompt_forge.py --save "data-check-v1" "look at this data"

# Use a local Ollama model
python prompt_forge.py --provider ollama --model llama3.2 "analyze these logs"

# Use a local MLX model on Mac
python prompt_forge.py --provider mlx --model mlx-community/Llama-3.2-3B-Instruct-4bit "explain this"

# Any OpenAI-compatible endpoint (LM Studio, vLLM, llamafile, etc.)
python prompt_forge.py --provider custom --base-url http://localhost:1234/v1 --model phi-4 "help"

# Version management
python prompt_forge.py --list-versions
python prompt_forge.py --show-version 3

# Check provider and API key status
python prompt_forge.py --list-providers
```

## Providers

| Provider | Default Model | API Key Env Var |
|----------|--------------|-----------------|
| `anthropic` | `claude-opus-4-6` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `gemini` | `gemini-3-flash-preview` | `GEMINI_API_KEY` |
| `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `ollama` | `llama3.2` | none (local) |
| `mlx` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | none (local) |
| `custom` | *(specify with `--model`)* | none (use `--api-key` if needed) |

## Prompt Versioning

Saved versions are stored in `prompt_versions.json` in the project directory.

```bash
# Save a version
python prompt_forge.py --save "my-prompt-v1" "summarize support tickets"

# List saved versions
python prompt_forge.py --list-versions

# Inspect a version
python prompt_forge.py --show-version 1
```
