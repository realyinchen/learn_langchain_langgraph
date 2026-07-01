# Progress: AgentLab

## What Works

### Basics Module — `01_create_agent` ✅
| File | Description | Status |
|------|-------------|--------|
| `01_chatbot.py` | Web-search chatbot: Agent = Model + Harness | ✅ Complete |

Features demonstrated:
- **Model**: `get_llm(temperature=0)` via LiteLLM
- **Harness**: `create_agent()` with tools, system prompt, checkpointer
- **Tools**: `TavilySearch` for web search capability
- **System Prompt**: Static time context + identity + tool usage rules (fully Chinese-localized for target audience)
- **Memory**: `MemorySaver` checkpoint for short-term (same-thread) memory
- **Test A**: Static question → no tool call
- **Test B**: Real-time question → auto web_search
- **Test C-1**: Multi-turn within same thread_id → remembers prior turns
- **Test C-2**: Different thread_id → no knowledge of prior session (记忆隔离)
- Ruff import order compliance (all imports before filterwarnings)
- All LLM-facing prompts (TavilySearch description, system context, identity, tool usage, response style) translated to Chinese

### Summarization Module ✅
All three summarization strategies and the intelligent routing agent are implemented and functional:

| File | Strategy | Status |
|------|----------|--------|
| `01_single_large_docuement.py` | MapReduce (split → parallel map → reduce) | ✅ Complete |
| `02_multi_documents.py` | Refine (chapter parse → initial → iterative refine) | ✅ Complete |
| `03_summarize_agent.py` | LangGraph agent (auto-routing: Stuff/MapReduce/Refine) | ✅ Complete |

### LLM Utility Layer ✅
- `utils/llm_uitls.py`: LiteLLM gateway with multi-provider support
- Environment-based configuration via `.env`
- Support for `enable_thinking` mode and local deployments
- All four invocation methods supported (invoke, stream, ainvoke, astream)

### Test Data ✅
- Chinese classical novels in `example_data/`: 三国演义, 水浒传, 红楼梦, 西游记 (full + excerpts)

### Documentation ✅
- `README.md` with setup instructions
- Chinese-language code comments and docstrings throughout
- Tutorial/report markdown files for the summarization agent

### Memory Bank ✅
- All 6 core files created and initialized

## What's Left to Build

### Near-Term
- [ ] Continue `basics/` module: tool call observation, streaming, multi-agent patterns
- [ ] Streaming demonstration examples (stream/astream usage)
- [ ] Evaluation framework for comparing summarization quality across strategies
- [ ] LangGraph checkpoint/persistence examples (PostgreSQL-based)

### Medium-Term
- [ ] RAG (Retrieval-Augmented Generation) module
- [ ] More advanced tool-calling agent examples
- [ ] LangSmith tracing integration examples

### Long-Term
- [ ] Production deployment patterns
- [ ] Cost tracking and optimization examples
- [ ] Multi-agent collaboration patterns

## Current Status
**Phase**: Basics module in progress — `01_chatbot.py` complete.  
**Branch**: `main` (commit `cb0ae65`)  
**Last Updated**: 2026-06-30

## Known Issues
1. LiteLLM cost-map fetch warnings are suppressed but the underlying issue (network access to GitHub for cost data) may affect users behind firewalls
2. `psycopg-binary` is listed as a dependency but PostgreSQL checkpoints are not yet demonstrated in any example
3. `langchain-postgres` dependency is installed but unused in current code (now used by `01_chatbot.py` for `langchain-tavily`)
4. The `03_summarize_agent.py` duplicates chapter parsing logic from `02_multi_documents.py` rather than sharing via import

## Evolution of Project Decisions

### Why LiteLLM over direct provider SDKs?
- **Decision**: Use LiteLLM as unified gateway
- **Rationale**: Single `get_llm()` interface works across DashScope, Anthropic, OpenAI, and local deployments. Readers only need to change `.env` to switch providers.
- **Trade-off**: Adds an abstraction layer and dependency

### Why rule-based routing over LLM routing?
- **Decision**: Pure Python `select_strategy()` function instead of asking an LLM to choose
- **Rationale**: Zero latency, zero cost, 100% deterministic. The routing logic is simple enough (4 rules based on count + length) that an LLM adds no value.
- **Trade-off**: Rules must be manually maintained if new strategies are added

### Why temperature=0 for the agent?
- **Decision**: Use `get_llm(temperature=0)` in `03_summarize_agent.py`
- **Rationale**: Summarization quality comparison requires deterministic output. Higher temperatures would add noise to cross-strategy comparisons.
- **Trade-off**: Less creative/varied summaries

### Why Chinese novels as test data?
- **Decision**: Use 四大名著 (Four Great Classical Novels)
- **Rationale**: Long-form, chapter-structured Chinese text provides realistic test cases. Culturally relevant for the target audience (Chinese-speaking developers following the WeChat account).
- **Trade-off**: Non-Chinese-speaking users cannot easily evaluate summary quality