# Technical Context: AgentLab

## Technologies Used

### Core Frameworks
| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | 1.3.1 | LLM application framework, LCEL chains |
| `langgraph` | 1.2.0 | Stateful agent orchestration, StateGraph |
| `langchain-litellm` | 0.6.6 | LiteLLM integration for LangChain |
| `langchain-community` | 0.4.1 | Community integrations |
| `langchain-postgres` | 0.0.17 | PostgreSQL checkpoints (LangGraph) |
| `langchain-tavily` | 0.2.18 | Tavily search integration |
| `langsmith` | 0.8.5 | Tracing and observability |

### Infrastructure
| Package | Version | Purpose |
|---------|---------|---------|
| `python-dotenv` | 1.2.1 | Environment variable management (.env files) |
| `asyncpg` | 0.31.0 | Async PostgreSQL driver |
| `psycopg-binary` | 3.3.2 | PostgreSQL adapter |

### Runtime
- **Python**: 3.12 (via conda environment `learn`)
- **Package manager**: pip (within conda env)
- **Environment**: conda (miniconda)

## Development Setup

1. Install miniconda
2. Create env: `conda create -n learn python=3.12`
3. Activate: `conda activate learn`
4. Install ipykernel: `conda install -c anaconda ipykernel && python -m ipykernel install --user --name learn`
5. Configure `.env` from `.example.env` template
6. Install deps: `pip install -r requirements.txt`

## Technical Constraints

### LLM Provider Configuration
The project uses **LiteLLM** as a unified gateway. Configuration is via `.env`:
- `LLM_MODEL` — Model ID (e.g., `dashscope/qwen3.6-27b`)
- `LLM_API_KEY` — API key for the provider
- `LLM_BASE_URL` — Optional: set for local OpenAI-compatible deployments
- `LITELLM_LOCAL_MODEL_COST_MAP` — Set to `True` to suppress remote cost-map fetches

### Text Splitting
- **RecursiveCharacterTextSplitter** with Chinese-aware separators: `["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]`
- Default chunk sizes: 3000 tokens (01), 10000 chars (03)
- Overlap: 100 characters

### Thresholds
- Default context window threshold: **10,000 characters** (Chinese text ≈ 1 char per token)
- Default summary max length: **300 characters**
- Batch concurrency: **5** parallel LLM calls

### Supported LLM Providers (via LiteLLM)
- DashScope (Alibaba Cloud, e.g., qwen models)
- Anthropic (Claude models)
- OpenAI (GPT models)
- Any OpenAI-compatible local deployment (via `LLM_BASE_URL`)

## Dependencies Graph
```
requirements.txt
├── langchain (core)
│   ├── langchain-community
│   ├── langchain-litellm (LLM gateway)
│   ├── langchain-postgres (checkpoints)
│   └── langchain-tavily (search)
├── langgraph (agents)
│   └── langgraph-checkpoint-postgres
├── langsmith (tracing)
├── python-dotenv (config)
├── asyncpg (async DB)
└── psycopg-binary (DB driver)
```

## Tool Usage Patterns

### IDE
- **VS Code** (recommended in README)

### Version Control
- **Git** with remote at `https://github.com/realyinchen/AgentLab.git`

### LLM Invocation Methods (via `get_llm()`)
1. `llm.invoke(messages)` — synchronous
2. `llm.stream(messages)` — synchronous streaming
3. `await llm.ainvoke(messages)` — async
4. `async for chunk in llm.astream(messages)` — async streaming

Currently, all summarization scripts use `invoke()` and `batch()` (sync). Streaming is supported by the utility but not yet demonstrated in examples.