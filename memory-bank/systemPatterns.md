# System Patterns: AgentLab

## System Architecture

```
AgentLab/
├── utils/
│   └── llm_uitls.py          # LLM abstraction layer (LiteLLM gateway)
├── Summarization/
│   ├── 01_single_large_docuement.py   # MapReduce chain
│   ├── 02_multi_documents.py          # Refine chain
│   └── 03_summarize_agent.py          # LangGraph agent (router)
├── example_data/              # Chinese novel test data
└── .env                       # API keys & model config
```

## Key Design Patterns

### 1. Chain Composition (LangChain LCEL)
All summarization logic is built using LangChain Expression Language (LCEL) with the `|` pipe operator:
```
prompt | llm | output_parser
```
This creates composable, declarative chains that are easy to read and modify.

**Used in**: All three summarization scripts for map, reduce, initial, and refine chains.

### 2. RunnableLambda for Custom Logic
Python functions are wrapped in `RunnableLambda` to integrate custom logic into LCEL chains:
```python
text_chunks_chain = RunnableLambda(text_splitter.split_text)
summarize_map_chain = RunnableLambda(parallel_summarize)
```
This allows mixing pure Python logic (text splitting, list joining, parallel batch calls) with LLM chains.

**Used in**: `01_single_large_docuement.py`, `02_multi_documents.py`

### 3. StateGraph + Conditional Edges (LangGraph)
The Summarize Agent uses LangGraph's `StateGraph` with conditional routing:
```
START → analyze_document → [route by strategy] → stuff_summarize → END
                                                  → mapreduce_summarize → END
                                                  → refine_summarize → END
```
The `TypedDict` state schema (`SummarizeAgentState`) carries data between nodes.

**Used in**: `03_summarize_agent.py`

### 4. Rule-Based Strategy Routing
Instead of using an LLM to choose a summarization strategy (which costs tokens and adds latency), `select_strategy()` uses pure Python rules:
- Single doc < 10k chars → Stuff
- Single doc ≥ 10k chars → Refine
- Multiple docs, all < 10k → MapReduce
- Multiple docs, any ≥ 10k → Refine

**Used in**: `03_summarize_agent.py` (`analyze_document` node)

### 5. Batch Parallelism
LangChain's `.batch()` method with `max_concurrency` enables parallel LLM calls:
```python
summarize_single_chain.batch(
    [{"text": chunk} for chunk in chunks],
    config={"max_concurrency": 5},
)
```
This is the core of the MapReduce "Map" phase.

**Used in**: `01_single_large_docuement.py`, `03_summarize_agent.py`

### 6. LLM Gateway Abstraction
`get_llm()` in `utils/llm_uitls.py` provides a single entry point for LLM configuration:
- Reads model/provider from environment variables
- Supports both cloud providers (DashScope, Anthropic, OpenAI) and local OpenAI-compatible deployments
- Handles `enable_thinking` mode for compatible models
- Automatically sets `LITELLM_LOCAL_MODEL_COST_MAP=True` to suppress cost-map fetches

**Used in**: All summarization scripts via `from utils.llm_uitls import get_llm`

## Component Relationships

```
utils/llm_uitls.py  ←── 01_single_large_docuement.py
                    ←── 02_multi_documents.py
                    ←── 03_summarize_agent.py (also imports 02's parse_chapters logic)
```

The `03_summarize_agent.py` script reimplements (rather than imports) the chapter parsing logic from `02_multi_documents.py`, making it fully self-contained. The three summarization scripts are independent of each other; they only share the `utils/llm_uitls.py` dependency.

## Critical Implementation Paths

### MapReduce Flow (01)
```
Input text → RecursiveCharacterTextSplitter → List[chunks]
  → batch(summarize_single_chain, max_concurrency=5) → List[partial_summaries]
  → join with separators → reduce_chain → final_summary
```

### Refine Flow (02)
```
Input text → CHAPTER_PATTERN regex → List[Document] (by chapter)
  → initial_summary_chain(doc[0]) → current_summary
  → for doc in docs[1:]: refine_chain(existing=current, new=doc) → current_summary
  → final_summary
```

### Agent Flow (03)
```
Input docs → analyze_document (rule-based routing)
  → stuff: merge all → single LLM call
  → mapreduce: merge all → text_splitter → batch map → reduce
  → refine: split large doc or treat each as Document → iterative refine
  → final_summary + llm_call_count