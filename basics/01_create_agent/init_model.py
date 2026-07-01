"""
ChatLiteLLM 的 invoke、stream、batch 三种调用方式
"""

import logging
import os
import warnings
from dotenv import load_dotenv

load_dotenv()

# 禁止 LiteLLM 从 GitHub 拉取
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from langchain_litellm import ChatLiteLLM  # noqa: E402


# 屏蔽 LiteLLM 非关键警告
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

# =============================================================================
# 初始化 Model
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
# 1. invoke —— 同步调用，返回完整结果
# =============================================================================
print("\n" + "=" * 60)
print("测试 1: invoke")
print("=" * 60)
response = llm.invoke("用一句话介绍什么是 LangChain。")
print(response.content)

# =============================================================================
# 2. stream —— 流式输出，逐 token 返回
# =============================================================================
print("\n" + "=" * 60)
print("测试 2: stream")
print("=" * 60)
for chunk in llm.stream("用三句话介绍什么是 Python。"):
    print(chunk.text, end="", flush=True)
print()
