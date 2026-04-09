# Open Deep Research

Autonomous multi-step web research agent implementing core [Gemini Deep Research](https://blog.google/products/gemini/google-gemini-deep-research/) algorithms. Plans sub-questions, searches the web, reads and summarizes sources, iterates until convergence, and produces cited markdown reports.

## Quickstart

```bash
# Clone and install
git clone https://github.com/Harduex/open-deep-research.git
cd open-deep-research
uv sync          # or: pip install -e .

# Run with Ollama (default)
odr research "What are the implications of quantum computing on cryptography?"

# Run with a different model
odr research "topic" --model openai/gpt-4o --max-sources 20 --budget 100000
```

## Requirements

- Python 3.10+
- An LLM provider (Ollama local, OpenAI, Anthropic, etc. via [LiteLLM](https://github.com/BerriAI/litellm))

### Optional

```bash
pip install -e ".[quality]"   # Embedding-based dedup (sentence-transformers)
```

## Configuration

Create a `config.yaml` in the project root (or pass `--config path/to/config.yaml`):

```yaml
llm:
  model: "ollama/llama3.1"
  api_base: "http://localhost:11434"  # for Ollama
  temperature: 0.3

search:
  provider: "duckduckgo"  # or "searxng"

research:
  max_iterations: 10
  max_sources: 30
  budget_tokens: 500000
```

Environment variables with `ODR_` prefix also work (e.g. `ODR_LLM__MODEL=openai/gpt-4o`).

## CLI

```bash
odr research "query"                            # Run research
odr sessions                                    # List saved sessions
odr resume <session-id>                         # Resume interrupted session
odr follow-up <session-id> "expand on X"        # Follow-up on completed research
odr export <session-id>                         # Export report as markdown
odr export <session-id> -o out.md               # Export to file
```

## How it works

1. **Plan** - LLM decomposes the query into sub-questions
2. **Search** - Generates search queries, executes via DuckDuckGo or SearXNG
3. **Read** - Fetches pages, extracts text with trafilatura, summarizes long content
4. **Iterate** - Evaluates coverage/saturation, updates plan, loops until converged or ceiling hit
5. **Synthesize** - Generates report sections with inline citations, detects contradictions

State is checkpointed to disk after each iteration for crash recovery.

## SearXNG (optional)

For privacy-first search, run a local SearXNG instance:

```bash
docker compose up -d
```

Then set `search.provider: "searxng"` in your config.

## License

Apache 2.0
