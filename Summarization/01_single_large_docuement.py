"""
MapReduce 长文档摘要 Chain

使用 MapReduce 策略对超出 LLM 上下文窗口的长文档进行摘要：
1. Split: 将长文档按 token 长度拆分为多个 chunk
2. Map: 并行对每个 chunk 生成局部摘要
3. Reduce: 将所有局部摘要汇总为最终摘要
"""

import sys
import os
import warnings
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# 必须在导入项目内部模块之前添加父目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_uitls import get_llm

# 屏蔽 LiteLLM 网络请求失败的警告
warnings.filterwarnings("ignore", message=".*Failed to fetch remote model cost map.*")
# 屏蔽 Pydantic 序列化类型不匹配的警告（LangChain 上游已知问题）
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

# ============================================================
# 1. Split: text_chunks_chain (RunnableLambda)
#    将输入文本按 chunk_size=3000, chunk_overlap=100 拆分
# ============================================================

# 初始化文本分割器（基于 token 估算：中文约 1 字符 ≈ 1 token）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)


text_chunks_chain = RunnableLambda(text_splitter.split_text)

# ============================================================
# 2. Map: summarize_map_chain (batch 并行)
#    使用 batch() 并行对每个 chunk 进行局部摘要
#    输入: List[str] (chunks) → 输出: List[str] (summaries)
# ============================================================

map_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的文本摘要助手。请用简洁精炼的语言对以下文本片段进行摘要，"
            "提取出最核心的信息和关键要点。\n"
            "要求：\n"
            "1. 用中文输出\n"
            "2. 保留关键人物、事件、数据等重要信息\n"
            "3. 摘要长度控制在原文的 20%-30% 以内\n"
            "4. 只输出摘要内容，不要添加任何额外说明",
        ),
        ("human", "请对以下文本进行摘要：\n\n{text}"),
    ]
)

llm = get_llm()

# 单个 chunk 的摘要链
summarize_single_chain = map_prompt | llm | StrOutputParser()


def parallel_summarize(chunks: List[str]) -> List[str]:
    """使用 batch() 并行对所有 chunk 进行摘要"""
    if not chunks:
        return []
    return summarize_single_chain.batch(
        [{"text": chunk} for chunk in chunks],
        config={"max_concurrency": 5},
    )


summarize_map_chain = RunnableLambda(parallel_summarize)

# ============================================================
# 3. Reduce: summarize_reduce_chain (RunnableLambda)
#    将所有 chunk 摘要汇总为最终摘要
# ============================================================

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
            "4. 只输出最终摘要，不要添加任何额外说明",
        ),
        (
            "human",
            "以下是从原文各部分提取的分段摘要，请整合为一份完整的总体摘要：\n\n{summaries}",
        ),
    ]
)

reduce_chain = reduce_prompt | llm | StrOutputParser()


def join_summaries(summaries: List[str]) -> dict:
    """将所有 chunk 摘要拼接为带标记的文本"""
    combined = "\n\n---\n\n".join(
        f"【第{i + 1}部分摘要】\n{s}" for i, s in enumerate(summaries)
    )
    return {"summaries": combined}


summarize_reduce_chain = RunnableLambda(join_summaries) | reduce_chain

# ============================================================
# 4. 最终 Chain: map_reduce_chain
#    text_chunks_chain → summarize_map_chain (batch并行) → summarize_reduce_chain
# ============================================================

map_reduce_chain = text_chunks_chain | summarize_map_chain | summarize_reduce_chain

# ============================================================
# 5. 测试入口
# ============================================================

if __name__ == "__main__":
    # 读取测试文件
    test_file = os.path.join(
        os.path.dirname(__file__), "..", "example_data", "西游记1-7.txt"
    )
    with open(test_file, "r", encoding="utf-8") as f:
        document = f.read()

    print(f"原文长度: {len(document)} 字符")
    print("=" * 60)

    # 1. 测试分块
    chunks = text_chunks_chain.invoke(document)
    print(f"拆分为 {len(chunks)} 个 chunk")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}: {len(chunk)} 字符")

    print("=" * 60)

    # 2. 运行完整 MapReduce 摘要
    print("正在生成摘要，请稍候...")
    final_summary = map_reduce_chain.invoke(document)

    print("=" * 60)
    print("最终摘要:")
    print(final_summary)
