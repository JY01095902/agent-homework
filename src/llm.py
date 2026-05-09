"""OpenAI Chat Completions 封装，用于多轮对话。"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI


def get_openai_client(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """构建客户端；未传 api_key 时使用环境变量 OPENAI_API_KEY。"""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("缺少 API Key：请传入 api_key 或设置环境变量 OPENAI_API_KEY")

    kwargs: Dict[str, Any] = {"api_key": key}
    url = base_url or os.environ.get("OPENAI_BASE_URL")
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def chat(
    messages: Sequence[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    client: Optional[OpenAI] = None,
    **kwargs: Any,
) -> str:
    """
    调用 OpenAI Chat Completions，返回助手回复文本。

    messages 示例::
        [
            {"role": "system", "content": "你是 helpful 助手。"},
            {"role": "user", "content": "你好"},
        ]

    model 默认读取环境变量 OPENAI_CHAT_MODEL，未设置则为 gpt-4o-mini。
    其余关键字参数透传给 ``client.chat.completions.create``（如 max_tokens）。
    """
    c = client or get_openai_client()
    m = model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    response = c.chat.completions.create(
        model=m,
        messages=list(messages),
        temperature=temperature,
        **kwargs,
    )

    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        raise RuntimeError("模型返回空内容（content 为 None）")
    return content

def chat_simple(
    user_message: str,
    *,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """单轮用户消息 → 助手回复；可选 system 提示。"""
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_message})
    return chat(msgs, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="试用 OpenAI Chat Completions")
    parser.add_argument(
        "question",
        nargs="?",
        default="用一句话用中文介绍你自己。",
        help="要问模型的问题（可省略，使用默认）",
    )
    args = parser.parse_args()
    try:
        reply = chat([{"role": "user", "content": args.question}])
    except ValueError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e
    print(reply)


if __name__ == "__main__":
    main()

