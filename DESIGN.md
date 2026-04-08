# open-deep-research — Design Document

> Open-source implementation of Gemini Deep Research algorithms: autonomous, multi-step web research with iterative planning, process-supervised verification, and exhaustive source synthesis.

## 1. Vision

A compact, minimalist, well-scaffolded Python research agent that implements the core algorithmic concepts from Gemini Deep Research — pluggable with any LLM via Ollama or commercial APIs. It autonomously plans, searches the web, reads sources, verifies findings, and produces comprehensive cited reports.

## 2. Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Domain | Web research agent | Matches Gemini Deep Research's primary use case |
| Output | Structured markdown reports with numbered inline citations | Standard, portable, human-readable |
| Interfaces | Python library + CLI | Library for embedding, CLI for direct use |
| Execution | Async single-process (asyncio + aiohttp) | Simple, sufficient for I/O-bound workloads |
| Loop termination | LLM-driven stopping with hard ceiling | Autonomous but bounded |
| State | Checkpoint to disk (JSON) | Enables resume without database dependency |
| Search | Plugin interface + SearXNG default | Flexible, privacy-first default |
| Document reading | HTTP + HTML extraction (trafilatura) | Handles 90%+ of web pages, lightweight |
| Source limit | 20-30 default, configurable | Balances thoroughness with cost |
| LLM interface | LiteLLM | 100+ providers via unified API |
| Multi-model | Single model (v1) | Simple; multi-model is v2+ |
| Structured output | Pydantic models | Type-safe, validated, reliable |
| Verification | Self-verification pass | Generator-Verifier-Reviser lite |
| Contradictions | Prompt-based detection | Lightweight, effective |
| Citations | Numbered inline [1], [2] | Standard academic style |
| Context management | Summarize-then-store | Fits more sources within context limits |
| Deduplication | Embedding-based clustering | Semantic matching via sentence-transformers |
| Cost control | Token budget with ceiling | Prevents runaway costs |
| Streaming | Real-time phase streaming | UX transparency, like Gemini's thinking_summaries |
| Resumability | Session ID + resume command | Crash recovery for long sessions |
| Follow-ups | Supported | Conversational refinement of reports |
| Testing | pytest + recorded fixtures | Deterministic CI, no API costs |
| License | Apache 2.0 | Permissive, matches ecosystem |

## 3. Architecture

### 3.1 The Research Loop

```
User Query
    │
    ▼
┌─────────┐     ┌──────────┐     ┌─────────┐
│  PLAN   │────▶│  SEARCH  │────▶│  READ   │
│         │     │          │     │         │
│ Decompose     │ Execute   │     │ Fetch & │
│ into sub-│     │ queries   │     │ extract │
│ questions│     │ via Search│     │ content │
└────▲────┘     │ Provider  │     └────┬────┘
     │          └──────────┘          │
     │                                 ▼
     │          ┌──────────┐     ┌─────────┐
     │          │ STOPPING │◀────│ ITERATE │
     │          │ CHECK    │     │         │
     │          │          │     │ Evaluate│
     │          │ LLM eval │     │ progress│
     │          │ + hard   │     │ update  │
     │          │ ceiling  │     │ plan    │
     │          └────┬─────┘     └─────────┘
     │               │
     │    ┌──────────┴──────────┐
     │    │ Not converged       │ Converged
     │    ▼                     ▼
     └────┘               ┌─────────┐
                          │ VERIFY  │
                          │ & SYNTH │
                          │         │
                          │ Generate│
                          │ Verify  │
                          │ Revise  │
                          └────┬────┘
                               │
                               ▼
                          ┌─────────┐
                          │ OUTPUT  │
                          │         │
                          │ Format  │
                          │ Cite    │
                          │ Report  │
                          └─────────┘
```

### 3.2 Module Architecture

```
open_deep_research/
├── __init__.py              # Public API: research(), ResearchSession
├── __main__.py              # CLI entry point
├── config.py                # Pydantic Settings + YAML loader
├── models.py                # Pydantic models (Plan, Finding, Source, Report, etc.)
├── core/
│   ├── __init__.py
│   ├── planner.py           # Decomposes query into sub-questions
│   ├── searcher.py          # Executes search queries via provider
│   ├── reader.py            # Fetches + extracts web page content
│   ├── evaluator.py         # Step-level evaluation (process supervision)
│   ├── synthesizer.py       # Generates report sections from findings
│   └── verifier.py          # Self-verification pass (Generator-Verifier-Reviser)
├── providers/
│   ├── __init__.py
│   ├── base.py              # SearchProvider abstract base class
│   ├── searxng.py           # SearXNG implementation (default)
│   └── tavily.py            # Tavily implementation (example plugin)
├── state/
│   ├── __init__.py
│   ├── session.py           # Session management (create, resume, follow-up)
│   ├── checkpoint.py        # JSON checkpoint save/load
│   └── budget.py            # Token budget tracking
├── embeddings/
│   ├── __init__.py
│   └── dedup.py             # Embedding-based entity resolution / dedup
├── llm/
│   ├── __init__.py
│   └── client.py            # LiteLLM wrapper with structured output parsing
└── cli/
    ├── __init__.py
    ├── app.py               # Typer app definition
    ├── commands.py           # research, resume, follow-up commands
    └── display.py            # Rich streaming display
```

### 3.3 Key Abstractions

```python
# --- models.py ---

class SubQuestion(BaseModel):
    id: str
    question: str
    status: Literal["pending", "investigating", "answered", "unanswerable"]
    findings: list[Finding] = []

class Finding(BaseModel):
    content: str                    # Summarized finding
    source_ids: list[int]           # References to Source entries
    confidence: Literal["high", "medium", "low"]
    embedding: list[float] | None = None  # For dedup clustering

class Source(BaseModel):
    id: int
    url: str
    title: str
    snippet: str                    # Extracted summary
    retrieved_at: datetime

class ResearchPlan(BaseModel):
    query: str                      # Original user query
    sub_questions: list[SubQuestion]
    iteration: int = 0
    max_iterations: int = 10        # Hard ceiling

class Report(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection]
    contradictions: list[str]
    sources: list[Source]
    metadata: ReportMetadata

class SessionState(BaseModel):
    session_id: str
    plan: ResearchPlan
    sources: list[Source]
    findings: list[Finding]
    budget: TokenBudget
    status: Literal["planning", "investigating", "synthesizing", "complete", "failed"]


# --- providers/base.py ---

class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        ...

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
```

### 3.4 The LLM Interaction Layer

All LLM interactions go through `llm/client.py`, which wraps LiteLLM:

```python
class LLMClient:
    async def complete(self, prompt: str, response_model: type[T]) -> T:
        """Call LLM and parse response into Pydantic model with retry."""
        ...

    async def complete_text(self, prompt: str) -> str:
        """Call LLM and return raw text (for report generation)."""
        ...

    def track_usage(self, response) -> None:
        """Update token budget tracker."""
        ...
```

Structured output flow:
1. Construct prompt with JSON schema instruction
2. Call LiteLLM `acompletion()`
3. Parse response into Pydantic model
4. If validation fails, retry with error feedback (max 2 retries)
5. Track token usage against budget

### 3.5 Context Management: Summarize-Then-Store

When the reader fetches a web page:
1. Extract text via trafilatura
2. If text > 1000 tokens, call LLM to summarize to ~500 tokens
3. Store summary + metadata as a `Source` entry
4. Full text is discarded after summarization (not stored)

This ensures the context window holds 20-30 source summaries comfortably within even small model windows (8k-32k tokens).

### 3.6 Embedding-Based Dedup

After each search-read cycle:
1. Compute embeddings for all new findings (sentence-transformers, `all-MiniLM-L6-v2`)
2. Compare against existing findings using cosine similarity
3. If similarity > 0.85 threshold, merge findings (keep more detailed version, combine source refs)
4. Flag near-misses (0.7-0.85) for LLM-based manual resolution

### 3.7 Stopping Criteria

The LLM evaluates stopping after each iteration using a structured prompt:

```python
class StoppingEvaluation(BaseModel):
    should_stop: bool
    reasoning: str
    coverage_score: float          # 0-1: how well sub-questions are covered
    saturation_detected: bool      # Are new searches returning duplicates?
    unanswered_questions: list[str] # Sub-questions still open
```

Stop if: `should_stop == True` OR `iteration >= max_iterations`

### 3.8 Self-Verification (Generator-Verifier-Reviser)

During synthesis of each report section:

```
1. GENERATE: LLM drafts the section from findings + sources
2. VERIFY:   LLM re-reads the draft against source summaries
             → Returns: correct | minor_fixes | fundamentally_flawed
3. REVISE:   If minor_fixes → LLM patches specific issues
             If fundamentally_flawed → restart from GENERATE
             Max 2 revision cycles per section
```

### 3.9 Token Budget

```python
class TokenBudget(BaseModel):
    max_tokens: int = 500_000       # Default ceiling
    used_tokens: int = 0
    warn_threshold: float = 0.8     # Warn at 80%

    @property
    def remaining(self) -> int: ...
    @property
    def is_exceeded(self) -> bool: ...
    @property
    def should_warn(self) -> bool: ...
```

Budget is checked before every LLM call. If exceeded, the agent enters synthesis mode immediately with whatever findings are available.

## 4. CLI Interface

```bash
# Basic research
odr research "What are the implications of quantum computing on cryptography?"

# With options
odr research "topic" --model ollama/llama3 --max-sources 50 --max-iterations 15 --budget 100000

# Resume interrupted session
odr resume abc123

# Follow-up on completed research
odr follow-up abc123 "Expand on the section about post-quantum algorithms"

# List sessions
odr sessions

# Export report
odr export abc123 --format markdown --output report.md
```

## 5. Configuration

```yaml
# config.yaml
llm:
  model: "ollama/llama3.1"           # LiteLLM model string
  api_base: "http://localhost:11434"  # For Ollama
  temperature: 0.3
  # api_key: set via LITELLM_API_KEY env var

search:
  provider: "searxng"
  searxng:
    base_url: "http://localhost:8888"
  # Alternative:
  # provider: "tavily"
  # tavily:
  #   api_key: set via TAVILY_API_KEY env var

research:
  max_iterations: 10
  max_sources: 30
  budget_tokens: 500000
  source_summary_tokens: 500

embedding:
  model: "all-MiniLM-L6-v2"
  similarity_threshold: 0.85

output:
  format: "markdown"
  include_confidence: true
  include_contradictions: true

sessions:
  storage_dir: "~/.open-deep-research/sessions"
```

## 6. Dependencies

### Core (v0.1)
```
litellm           # LLM provider abstraction
aiohttp           # Async HTTP client
trafilatura       # HTML content extraction
pydantic>=2.0     # Data models + settings
typer             # CLI framework
rich              # Terminal UI + streaming
pyyaml            # Config file parsing
```

### Quality layer (v0.2)
```
sentence-transformers  # Local embeddings for dedup
numpy                  # Embedding math (cosine similarity)
```

### DevOps (v0.3)
```
pytest            # Testing
pytest-asyncio    # Async test support
respx             # HTTP mocking
pytest-recording  # VCR-style recording (or vcrpy)
```

## 7. Release Plan

### v0.1 — Core Pipeline
- Research loop (plan → search → read → iterate → output)
- SearXNG search provider + plugin interface
- LiteLLM integration
- Pydantic structured outputs
- Typer CLI with Rich streaming
- Summarize-then-store context management
- LLM-driven stopping with hard ceiling
- JSON checkpointing
- Token budget tracking
- Numbered inline citations
- Basic session management (create, list)
- docker-compose.yml for SearXNG

### v0.2 — Quality Layer
- Self-verification pass (Generator-Verifier-Reviser)
- Prompt-based contradiction detection
- Embedding-based entity resolution / dedup
- Follow-up query support
- Session resume command
- Configurable report templates

### v0.3 — Infrastructure
- Comprehensive test suite (pytest + recorded fixtures)
- CI/CD (GitHub Actions)
- PyPI packaging
- Documentation site
- Example configs for popular models (GPT-4, Claude, Llama 3, Mistral, Gemma)
- Docker image for the full agent (not just SearXNG)

## 8. Algorithmic Mapping

How this project maps to the Gemini Deep Research paper's concepts:

| Gemini Concept | open-deep-research Implementation |
|---|---|
| Plan-Search-Read-Iterate-Output loop | `core/planner.py`, `core/searcher.py`, `core/reader.py`, `core/evaluator.py`, `core/synthesizer.py` |
| Monte Carlo Tree Search (MCTS) | Simplified to iterative sub-question exploration with LLM-guided branching (no explicit tree data structure in v1) |
| Process-Supervised Reward Models (PRMs) | `core/evaluator.py` — LLM-based step evaluation after each search-read cycle |
| Systematic Collation | Graph-based exploration via follow-up queries extracted from source content |
| Entity Resolution / Dedup | `embeddings/dedup.py` — embedding clustering with cosine similarity |
| Epistemic Stopping Criteria | `StoppingEvaluation` model — LLM evaluates saturation, coverage, convergence |
| Generator-Verifier-Reviser (Aletheia) | `core/verifier.py` — self-verification pass during synthesis |
| Multi-LLM Adaptive Branching (AB-MCTS) | Deferred to v2 (single model in v1) |
| Inference-Time Scaling | Inherent — more iterations = more compute = better results |
| thinking_summaries | `cli/display.py` — Rich streaming of intermediate reasoning |
| Interactions API (async execution) | `state/session.py` + `state/checkpoint.py` — session management with resume |
| Context Management (900k tokens) | Summarize-then-store — fits findings within any model's window |
| Dynamic query generation | `core/planner.py` — LLM generates new sub-questions based on findings |

## 9. Out of Scope (for now)

- Multi-model routing (AB-MCTS) — v2+
- PDF/CSV/image ingestion — v2+
- Formal MCTS tree data structure — may not be needed for practical use
- Web UI — v2+
- REST API server — v2+
- Multi-user / concurrent sessions — v2+
- MCP tool integration — v2+
