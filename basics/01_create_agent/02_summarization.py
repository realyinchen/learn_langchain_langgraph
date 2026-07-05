"""
用 LangChain 实现 MapReduce & Refine 两种策略的文档总结 Pipeline
"""

import logging
import os
import warnings
from typing import List, Dict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from langchain_litellm import ChatLiteLLM  # noqa: E402

# 屏蔽 LiteLLM 非关键警告
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

# =============================================================================
# 0. 初始化 LLM
# =============================================================================

kwargs = {
    "model": os.getenv("LLM_MODEL"),
    "api_key": os.getenv("LLM_API_KEY"),
    "api_base": os.getenv("LLM_BASE_URL"),
    "streaming": True,
    "model_kwargs": {"extra_body": {"enable_thinking": False}},
}
llm = ChatLiteLLM(**kwargs)

print(f"[Model] 已初始化: {llm.model}")

# =============================================================================
# 1. 文档拆分：两阶段策略
#    Phase 1 — MarkdownHeaderTextSplitter：按标题层级（#/##/###）拆分
#    Phase 2 — RecursiveCharacterTextSplitter：对超大块按长度二次拆分
# =============================================================================

# ---------------------------------------------------------------------------
# Phase 1: Markdown 标题结构拆分
# ---------------------------------------------------------------------------

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),  # 一级标题：如 "# Background"
    ("##", "h2"),  # 二级标题：如 "## Retriever"
    ("###", "h3"),  # 三级标题：如 "### Chunking"
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,  # 保留标题文本在 chunk 中，帮助 LLM 理解语义上下文
)

# ---------------------------------------------------------------------------
# Phase 2: 递归字符长度拆分（仅对超大块生效）
# ---------------------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
)


def split_document(text: str) -> List[Dict[str, str]]:
    """
    两阶段文档拆分: Markdown 标题结构 + 递归字符长度。
    """

    # 阶段1：按 Markdown 标题层级拆分为语义块
    md_docs = header_splitter.split_text(text)

    # 阶段2：对每个语义块按长度二次拆分（小块保持原样，超大块才进一步切分）
    splits = text_splitter.split_documents(md_docs)

    sections: List[Dict[str, str]] = []
    for doc in splits:
        headers = [doc.metadata.get(k, "") for k in ["h1", "h2", "h3"]]
        title = " > ".join(h for h in headers if h) or "导语"
        sections.append({"title": title, "content": doc.page_content})

    return sections


# =============================================================================
# 2. MapReduce 总结策略
#    Split → Map (batch 并行) → Reduce
# =============================================================================

map_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的文本摘要助手。请用简洁精炼的语言对以下文本片段进行摘要，"
            "提取出最核心的信息和关键要点。\n"
            "要求：\n"
            "1. 用中文输出\n"
            "2. 保留关键人物、事件、数据、术语等重要信息\n"
            "3. 摘要长度不超过 500 个汉字\n"
            "4. 只输出摘要内容，不要添加任何额外说明",
        ),
        (
            "human",
            "以下片段来自「{title}」：\n\n{content}",
        ),
    ]
)

summarize_single_chain = map_prompt | llm | StrOutputParser()


def parallel_summarize(sections: List[Dict[str, str]]) -> List[str]:
    """使用 batch() 并行对所有 section 进行摘要"""
    if not sections:
        return []
    inputs = [{"title": sec["title"], "content": sec["content"]} for sec in sections]
    return summarize_single_chain.batch(inputs, config={"max_concurrency": 5})


reduce_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的文本摘要助手。现在有一组分段摘要，每段摘要对应原文的一个部分。"
            "请将这些分段摘要整合成一篇完整、连贯的总体摘要。\n"
            "要求：\n"
            "1. 用中文输出\n"
            "2. 保持逻辑连贯，去除重复信息\n"
            "3. 按原文的逻辑顺序组织内容\n"
            "4. 最终摘要不超过 500 个汉字\n"
            "5. 只输出最终摘要，不要添加任何额外说明",
        ),
        (
            "human",
            "以下是从原文各部分提取的分段摘要，请整合为一份完整的总体摘要：\n\n{summaries}",
        ),
    ]
)

reduce_chain = reduce_prompt | llm | StrOutputParser()


def join_summaries(summaries: List[str]) -> dict:
    """将所有 section 摘要拼接为带标记的文本"""
    combined = "\n\n---\n\n".join(
        f"【第{i + 1}部分摘要】\n{s}" for i, s in enumerate(summaries)
    )
    return {"summaries": combined}


def mapreduce_summarize(sections: List[Dict[str, str]]) -> str:
    """MapReduce 完整流程"""
    print(f"\n  [MapReduce] Map 阶段：并行处理 {len(sections)} 个分段...")
    summaries = parallel_summarize(sections)
    for i, s in enumerate(summaries):
        print(f"    分段 {i + 1} 摘要: {len(s)} 字符")

    print(f"  [MapReduce] Reduce 阶段：整合 {len(summaries)} 份摘要...")
    combined = join_summaries(summaries)
    final = reduce_chain.invoke(combined)
    return final


# =============================================================================
# 3. Refine 总结策略
#    Initial Summary → 逐段迭代精炼
# =============================================================================

initial_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的文本摘要助手。请用简洁精炼的语言对以下文档进行摘要，"
            "提取出最核心的信息和关键要点。\n"
            "要求：\n"
            "1. 用中文输出\n"
            "2. 保留关键人物、事件、数据、术语等重要信息\n"
            "3. 摘要长度不超过 500 个汉字\n"
            "4. 只输出摘要内容，不要添加任何额外说明",
        ),
        (
            "human",
            "以下内容来自「{title}」：\n\n{content}",
        ),
    ]
)

initial_summary_chain = initial_prompt | llm | StrOutputParser()

refine_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的文本摘要助手。现在你已经有了一份对前面文档的摘要草案，"
            "请基于下面提供的新文档内容，更新并完善这份摘要。\n\n"
            "具体要求：\n"
            "1. 用中文输出\n"
            "2. 将新文档中的关键信息融入现有摘要，保持逻辑连贯\n"
            "3. 如果新内容与旧摘要存在矛盾，以新内容为准进行修正\n"
            "4. 去除重复信息，保持摘要精炼\n"
            "5. 按文档的逻辑顺序组织内容\n"
            "6. 更新后的摘要不超过 500 个汉字\n"
            "7. 只输出更新后的完整摘要，不要添加任何额外说明",
        ),
        (
            "human",
            "现有摘要：\n{existing_summary}\n\n"
            "新文档内容（来自「{title}」）：\n{content}\n\n"
            "请基于新内容更新摘要。",
        ),
    ]
)

refine_chain = refine_prompt | llm | StrOutputParser()


def refine_summarize(sections: List[Dict[str, str]]) -> str:
    """Refine 完整流程：初始摘要 → 逐段迭代精炼"""
    if not sections:
        return ""

    # 第一步：对第一个 section 生成初始摘要
    first = sections[0]
    print(
        f"\n  [Refine] 1/{len(sections)} 生成初始摘要 "
        f"（{first['title']}，{len(first['content'])} 字符）..."
    )
    current_summary = initial_summary_chain.invoke(
        {
            "title": first["title"],
            "content": first["content"],
        }
    )
    print(f"    初始摘要: {len(current_summary)} 字符")

    # 第二步：逐段精炼
    for i in range(1, len(sections)):
        sec = sections[i]
        print(
            f"  [Refine] {i + 1}/{len(sections)} 精炼摘要 "
            f"（{sec['title']}，{len(sec['content'])} 字符）..."
        )
        current_summary = refine_chain.invoke(
            {
                "existing_summary": current_summary,
                "title": sec["title"],
                "content": sec["content"],
            }
        )
        print(f"    更新后摘要: {len(current_summary)} 字符")

    return current_summary


# =============================================================================
# 4. 测试入口
# =============================================================================

if __name__ == "__main__":
    # 加载示例文档
    doc_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "example_data", "rag_wiki.md"
    )
    with open(doc_path, "r", encoding="utf-8") as f:
        document = f.read()

    # -------------------------------------------------------------------
    # 步骤 1：文档拆分
    # -------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("【步骤 1】文档拆分")
    print("─" * 70)

    sections = split_document(document)

    # -------------------------------------------------------------------
    # 步骤 2：MapReduce 总结
    # -------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("【步骤 2】MapReduce 总结策略")
    print("─" * 70)

    mapreduce_result = mapreduce_summarize(sections)

    print("\n" + "=" * 70)
    print("【MapReduce 最终摘要】")
    print("=" * 70)
    print(mapreduce_result)

    # -------------------------------------------------------------------
    # 步骤 3：Refine 总结
    # -------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("【步骤 3】Refine 总结策略")
    print("─" * 70)

    refine_result = refine_summarize(sections)

    print("\n" + "=" * 70)
    print("【Refine 最终摘要】")
    print("=" * 70)
    print(refine_result)
