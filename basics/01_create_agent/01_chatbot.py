"""
用 LangChain 的 `create_agent` 构建一个能联网搜索的 chatbot
"""

import logging
import os
import warnings
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


load_dotenv()
# 禁止 LiteLLM 从 GitHub 拉取
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from langchain_litellm import ChatLiteLLM  # noqa: E402


# 屏蔽 LiteLLM 非关键警告
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

# =============================================================================
# 1. 初始化 Model
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

# =========================================================================
# 2. TOOLS —— 外部能力接口
# =========================================================================

web_search = TavilySearch(
    max_results=3,
    description=(
        "搜索网络获取实时信息。"
        "当你需要查找当前新闻、天气、股票价格、近期事件，"
        "或任何可能随时间变化、超出你训练数据截止日期之外的信息时，使用此工具。"
    ),
)

tools = [web_search]
print(f"[Tools] 已注册 {len(tools)} 个工具: {[t.name for t in tools]}")

# =========================================================================
# 3. SYSTEM PROMPT —— 行为规则说明书
# =========================================================================

tz_cn = timezone(timedelta(hours=8))
now = datetime.now(tz_cn)

system_prompt = f"""

身份定位
--------
你是 AgentHub，一个乐于助人的 AI 助手。

当前时间为 {now.strftime("%Y-%m-%d %H:%M:%S")}，时区是 Asia/Shanghai (UTC+8)。

工具使用规则
----------
可用工具：`web_search`

1. **对时效性信息使用 `web_search`**——天气、新闻、股票价格、当前事件、近期数据，或任何可能随时间变化的内容。

2. **对静态问题直接回答**——问候语、笑话、数学、常识、历史、编程概念等——无需调用工具。

回复风格
--------
1. 简洁且有帮助。
2. 自然地呈现信息——不要说"根据搜索结果"之类的话。
3. 系统上下文中的时间是准确的，可以作为参考。
"""

print("[System Prompt] 已配置")

# =========================================================================
# 4. Memory
# =========================================================================

# 创建短期记忆存储器
memory = InMemorySaver()

# =========================================================================
# 5. Agent
# =========================================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory,
)
print("[Harness] Agent 已创建（含 MemorySaver 短期记忆）")


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # 测试 A：静态问题 — 不触发工具
    # -------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 测试 A：静态问题（不触发 web_search）")
    print("#" * 60)
    thread_id = "tutorial-session-1"
    config = {"configurable": {"thread_id": thread_id}}
    content = "你好！请用一句话介绍 LangChain 是什么。"
    result = agent.invoke(
        {"messages": [HumanMessage(content=content)]},
        config=config,  # type: ignore
    )
    print(f"[用户]：{content}")
    print(f"[Agent]: {result['messages'][-1].content}")

    # -------------------------------------------------------------------
    # 测试 B：实时问题 — 触发 web_search
    # -------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 测试 B：实时问题（自动调用 web_search）")
    print("#" * 60)
    thread_id = "tutorial-session-2"
    config = {"configurable": {"thread_id": thread_id}}
    content = "今天有什么重要的科技新闻？"
    result = agent.invoke(
        {"messages": [HumanMessage(content=content)]},
        config=config,  # type: ignore
    )
    print(f"[用户]：{content}")
    print(f"[Agent]: {result['messages'][-1].content}")

    # -------------------------------------------------------------------
    # 测试 C-1：多轮对话 — 短期记忆（同一会话内）
    # -------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 测试 C-1：短期记忆（同一 thread_id 内记住前文）")
    print("#" * 60)

    thread_id = "tutorial-session-3"
    config = {"configurable": {"thread_id": thread_id}}
    content = "我叫小明，是一名 Python 开发者。"
    result = agent.invoke(
        {"messages": [HumanMessage(content=content)]},
        config=config,  # type: ignore
    )
    print(f"[用户]：{content}")
    print(f"[Agent]: {result['messages'][-1].content}")

    content = "我叫什么名字？我是做什么的？"
    result = agent.invoke(
        {"messages": [HumanMessage(content="")]},
        config=config,  # type: ignore
    )
    print(f"[用户]：{content}")
    print(f"[Agent]: {result['messages'][-1].content}")

    # -------------------------------------------------------------------
    # 测试 C-2：跨会话记忆隔离
    # -------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 测试 C-2：记忆隔离（不同 thread_id 之间状态不共享）")
    print("# 换一个全新的 thread_id，chatbot 不应该知道上一个会话的信息")
    print("#" * 60)

    thread_id = "tutorial-session-4"
    config = {"configurable": {"thread_id": thread_id}}
    content = "我叫什么名字？我是做什么的？"
    result = agent.invoke(
        {"messages": [HumanMessage(content="")]},
        config=config,  # type: ignore
    )
    print(f"[用户]：{content}")
    print(f"[Agent]: {result['messages'][-1].content}")
