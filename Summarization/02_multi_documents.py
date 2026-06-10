"""
Refine 多文档摘要 Chain

使用 Refine（精炼/迭代优化）策略对多篇文档进行摘要：
1. Parse: 将文本按章节拆分为多个 Document 对象
2. Initial: 对第一篇文档生成初始摘要草案
3. Refine: 迭代地将"当前摘要 + 新文档"输入 LLM，逐步完善摘要
4. 循环直至所有文档处理完毕，输出最终摘要
"""

import sys
import os
import re
import warnings
from typing import List

from langchain_core.documents import Document
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
# 1. Parse: parse_chapters_chain (RunnableLambda)
#    将输入文本按"第X回"分割为多个 Document 对象
#    每个 Document 包含一回的完整内容及元数据
# ============================================================

# 匹配回目标题的正则：第X回 标题
CHAPTER_PATTERN = re.compile(r"(第[一二三四五六七八九十百千]+回\s+[^\n]+)")


def parse_chapters(text: str) -> List[Document]:
    """按回目分割文本，返回 Document 列表"""
    # 找到所有回目标题的匹配位置
    matches = list(CHAPTER_PATTERN.finditer(text))

    if not matches:
        # 如果没有匹配到回目，将整个文本作为一个 Document
        return [Document(page_content=text, metadata={"chapter": 1})]

    documents = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()  # 正文从标题之后开始

        # 正文结束位置：下一个回目标题的开始，或文本末尾
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()
        chapter_num = i + 1

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "chapter": chapter_num,
                    "title": title,
                },
            )
        )

    return documents


parse_chapters_chain = RunnableLambda(parse_chapters)

# ============================================================
# 2. Initial: initial_summary_chain
#    对第一篇文档生成初始摘要草案
#    输入: str (文档内容) → 输出: str (初始摘要)
# ============================================================

initial_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的文本摘要助手。请用简洁精炼的语言对以下文档进行摘要，"
            "提取出最核心的信息和关键要点。\n"
            "要求：\n"
            "1. 用中文输出\n"
            "2. 保留关键人物、事件、地点等重要信息\n"
            "3. 摘要长度控制在原文的 20%-30% 以内\n"
            "4. 只输出摘要内容，不要添加任何额外说明",
        ),
        ("human", "请对以下文档进行摘要：\n\n{document_content}"),
    ]
)

llm = get_llm()

initial_summary_chain = initial_prompt | llm | StrOutputParser()

# ============================================================
# 3. Refine: refine_chain
#    将"现有摘要草案"与"新文档内容"一起发送给 LLM，
#    要求 LLM 基于新内容更新并完善摘要
#    输入: dict {existing_summary, document_content} → 输出: str (更新后摘要)
# ============================================================

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
            "6. 只输出更新后的完整摘要，不要添加任何额外说明",
        ),
        (
            "human",
            "现有摘要：\n{existing_summary}\n\n"
            "新文档内容：\n{document_content}\n\n"
            "请基于新内容更新摘要。",
        ),
    ]
)

refine_chain = refine_prompt | llm | StrOutputParser()

# ============================================================
# 4. Refine Summary: refine_summary_chain (RunnableLambda)
#    迭代精炼流程：
#    初始摘要(doc[0]) → refine(+doc[1]) → refine(+doc[2]) → ... → 最终摘要
#    输入: List[Document] → 输出: str (最终摘要)
# ============================================================


def refine_summarize(documents: List[Document]) -> str:
    """迭代精炼多文档摘要"""
    if not documents:
        return ""

    if len(documents) == 1:
        # 只有一篇文档，直接返回初始摘要
        return initial_summary_chain.invoke(
            {"document_content": documents[0].page_content}
        )

    # 第一步：对第一篇文档生成初始摘要
    print(
        f"  [1/{len(documents)}] 正在生成初始摘要 "
        f"(第{documents[0].metadata['chapter']}回: {documents[0].metadata['title']})..."
    )
    current_summary = initial_summary_chain.invoke(
        {"document_content": documents[0].page_content}
    )
    print(f"       初始摘要长度: {len(current_summary)} 字符")

    # 第二步：逐篇精炼
    for i in range(1, len(documents)):
        doc = documents[i]
        chapter = doc.metadata["chapter"]
        title = doc.metadata["title"]
        print(f"  [{i + 1}/{len(documents)}] 正在精炼摘要 (第{chapter}回: {title})...")

        current_summary = refine_chain.invoke(
            {
                "existing_summary": current_summary,
                "document_content": doc.page_content,
            }
        )
        print(f"       更新后摘要长度: {len(current_summary)} 字符")

    return current_summary


refine_summary_chain = RunnableLambda(refine_summarize)

# ============================================================
# 5. 最终 Chain: refine_multi_doc_chain
#    parse_chapters_chain → refine_summary_chain
# ============================================================

refine_multi_doc_chain = parse_chapters_chain | refine_summary_chain

# ============================================================
# 6. 测试入口
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

    # 1. 测试解析
    docs = parse_chapters_chain.invoke(document)
    print(f"解析为 {len(docs)} 篇文档:")
    for doc in docs:
        ch = doc.metadata["chapter"]
        title = doc.metadata["title"]
        content_len = len(doc.page_content)
        print(f"  第{ch}回: {title} ({content_len} 字符)")

    print("=" * 60)

    # 2. 运行完整 Refine 摘要
    print("正在使用 Refine 策略生成摘要，请稍候...")
    print("-" * 40)
    final_summary = refine_multi_doc_chain.invoke(document)

    print("-" * 40)
    print("=" * 60)

    print("最终摘要:")
    print(final_summary)
