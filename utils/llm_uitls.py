import os
from typing import Optional
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM

# Load environment variables
load_dotenv()


def get_llm(
    model_id: Optional[str] = None,
    enable_thinking: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    streaming: bool = False,
) -> ChatLiteLLM:
    """
    Get LLM instance with configurable parameters.

    Args:
        model_id: Model identifier, e.g., "dashscope/qwen3.6-27b".
                  Defaults to LLM_MODEL from env.
        enable_thinking: Whether to enable thinking mode, defaults to False.
                         Uses extra_body={"enable_thinking": True/False}
        api_key: API key, defaults to LLM_API_KEY from env
        base_url: Base URL for the API, defaults to LLM_BASE_URL from env
        temperature: Temperature for generation, defaults to 0.7
        streaming: Whether to enable streaming mode, defaults to False

    Returns:
        ChatLiteLLM: Configured LLM instance that supports invoke, stream,
                     ainvoke, astream methods

    Usage:
        # Get LLM instance
        llm = get_llm(model_id="dashscope/qwen3.6-27b", enable_thinking=True)

        # Four calling methods:
        # 1. invoke - synchronous call
        result = llm.invoke(messages)

        # 2. stream - synchronous streaming
        for chunk in llm.stream(messages):
            print(chunk.content, end="")

        # 3. ainvoke - async call
        result = await llm.ainvoke(messages)

        # 4. astream - async streaming
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")
    """
    # Get model configuration
    model_name = model_id or os.getenv("LLM_MODEL")
    api_key_value = api_key or os.getenv("LLM_API_KEY")
    base_url_value = base_url or os.getenv("LLM_BASE_URL")

    if not model_name:
        raise ValueError("LLM_MODEL not found in environment variables or parameters")
    if not api_key_value:
        raise ValueError("LLM_API_KEY not found in environment variables or parameters")

    # Build model string with openai/ prefix for LiteLLM compatibility
    if not model_name.startswith("openai/") and base_url_value:
        model_name = f"openai/{model_name}"

    # Build extra_body for thinking mode
    extra_body = {"enable_thinking": enable_thinking}

    # Build kwargs for ChatLiteLLM
    kwargs = {
        "model": model_name,
        "api_key": api_key_value,
        "temperature": temperature,
        "streaming": streaming,
        "model_kwargs": {"extra_body": extra_body},
    }

    # Set api_base for custom endpoint (LiteLLM uses api_base, not base_url)
    if base_url_value:
        kwargs["api_base"] = base_url_value

    return ChatLiteLLM(**kwargs)


if __name__ == "__main__":
    # Test the function
    llm = get_llm()
    print("Successfully created LLM instance (thinking disabled)")

    llm_thinking = get_llm(enable_thinking=True)
    print("Successfully created LLM instance (thinking enabled)")

    llm_custom_model = get_llm(model_id="dashscope/qwen3.6-27b")
    print("Successfully created LLM instance with custom model_id")
