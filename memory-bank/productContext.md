# Product Context: AgentLab

## Why This Project Exists
LangChain and LangGraph are rapidly evolving frameworks with steep learning curves. Official documentation is extensive but often lacks end-to-end, runnable examples that demonstrate practical patterns. AgentLab fills this gap by providing:

- **Executable reference code** that readers can clone and run immediately
- **Progressive complexity** — each example builds on concepts from previous ones
- **Chinese-language context** — using Chinese classical novels (四大名著) as test data, making examples relatable for Chinese-speaking developers

## Problems It Solves
1. **Framework fragmentation**: Abstracts away the complexity of LangChain/LangGraph API surfaces behind clean, documented functions
2. **LLM provider lock-in**: Uses LiteLLM as a unified gateway, supporting multiple LLM providers (DashScope, Anthropic, OpenAI, local deployments) through a single `get_llm()` interface
3. **Strategy selection paralysis**: The Summarize Agent demonstrates automatic strategy routing, removing the burden of manual strategy choice
4. **Reproducibility**: Conda environment + requirements.txt + .env pattern ensures consistent setup

## How It Should Work
1. Clone repo → set up conda env → configure `.env` with API keys → run any `.py` file
2. Each `.py` file is self-contained and runnable (`if __name__ == "__main__"`)
3. Examples use real data (Chinese novels in `example_data/`) for authentic results
4. Code is heavily commented in Chinese, explaining each architectural decision

## User Experience Goals
- **Zero-config startup**: Only `.env` setup required, everything else is automated
- **Immediate feedback**: Each script prints progress, chunk counts, strategy choices, and timing
- **Learn by reading code**: Comments and docstrings serve as inline tutorials
- **Copy-paste friendly**: Utility functions are designed to be extracted and reused in readers' own projects