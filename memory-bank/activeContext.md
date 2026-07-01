# Active Context: AgentLab

## Current Work Focus
The project is currently working on the **Basics** module (`basics/`), starting with `01_create_agent/01_chatbot.py` — a tutorial that teaches the Agent = Model + Harness formula by building a web-search-capable chatbot with LangChain's `create_agent`.

## Recent Changes
- **`basics/01_create_agent/01_chatbot.py`** — Complete chatbot tutorial with:
  - Model (get_llm) + Harness (create_agent) + Tools (TavilySearch) + System Prompt + Memory (MemorySaver)
  - Three test suites: A) static Q&A (no tool), B) real-time search (web_search), C) multi-turn memory (same session + cross-session isolation)
  - Test C-1: same thread_id remembers prior turns (短期记忆)
  - Test C-2: different thread_id = no knowledge of prior session (记忆隔离)
  - Ruff import order fixed (all imports before filterwarnings)
  - **Chinese localization**: TavilySearch description, system context note, and system prompt (identity/tool usage/response style) all translated to Chinese for the Chinese-speaking target audience
- Previous: Summarization module complete (01/02/03 all functional)
- `utils/llm_uitls.py` — LiteLLM-based LLM abstraction layer with `.env` configuration
- Example data: Chinese classical novels (三国演义, 水浒传, 红楼梦, 西游记, 西游记1-7)
- Git initialized, remote set to `https://github.com/realyinchen/AgentLab.git`

## Next Steps
- Continue `basics/` module tutorials (observe tool calls, streaming, etc.)
- Add more LangChain/LangGraph patterns (RAG, tool-calling agents, chatbots)
- Add evaluation/comparison framework for summarization quality
- Add streaming support demonstrations
- Add checkpoint/persistence patterns with LangGraph

## Active Decisions & Considerations
- **LiteLLM as LLM gateway**: Chosen for provider flexibility — supports DashScope, Anthropic, OpenAI, and local deployments through a single interface
- **Chinese test data**: Using 四大名著 provides realistic, long-form text for testing summarization
- **Rule-based routing over LLM routing**: The `analyze_document` node uses pure Python rules (zero latency, zero cost) instead of asking an LLM to choose a strategy
- **Temperature=0 for agent**: Ensures deterministic, reproducible summaries
- **max_concurrency=5** for batch operations: Balances parallelism with API rate limits

## Important Patterns & Preferences
- All Python files are self-contained with `if __name__ == "__main__"` test blocks
- Extensive Chinese-language comments and docstrings
- `sys.path.insert(0, ...)` pattern for sibling imports from `utils/`
- Suppressed LiteLLM cost-map warnings as they are non-critical
- TypedDict for LangGraph state schema (`SummarizeAgentState`)

## Learnings & Project Insights
- MapReduce with `batch()` provides good parallelism for independent chunk processing
- Refine strategy produces higher-quality summaries at the cost of sequential LLM calls (higher latency)
- The 10,000-character threshold is a reasonable heuristic for distinguishing "small" vs "large" documents in Chinese text
- LangGraph's `add_conditional_edges` enables clean strategy routing without complex if/else chains