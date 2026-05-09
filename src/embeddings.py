"""OpenAI Embeddings API 封装。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from openai import OpenAI

from chunking import chunk_by_headings
from llm import get_openai_client
from pdf_extract import extract_pdf_text_ocr


def embed_texts(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    batch_size: int = 64,
    client: Optional[OpenAI] = None,
) -> List[List[float]]:
    """
    对一批文本调用 OpenAI Embeddings，返回与 ``texts`` 同序的向量列表。

    - ``model`` 默认读环境变量 ``OPENAI_EMBEDDING_MODEL``，未设置则为 ``text-embedding-3-small``。
    - ``dimensions``：仅部分模型（如 ``text-embedding-3-*``）支持缩短维度。
    - ``batch_size``：单次请求条数，避免一次提交过长列表。

    使用与 ``llm.get_openai_client`` 相同的鉴权：``OPENAI_API_KEY``、可选 ``OPENAI_BASE_URL``。
    """
    if not texts:
        return []

    c = client or get_openai_client()
    m = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    texts_list = list(texts)
    out: List[List[float]] = []

    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i : i + batch_size]
        kwargs: dict[str, Any] = {"model": m, "input": batch}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        resp = c.embeddings.create(**kwargs)
        ordered = sorted(resp.data, key=lambda d: d.index)
        for item in ordered:
            out.append(list(item.embedding))

    return out


def embed_text(
    text: str,
    *,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    client: Optional[OpenAI] = None,
) -> List[float]:
    """单条文本嵌入，等价于 ``embed_texts([text])[0]``。"""
    vec = embed_texts([text], model=model, dimensions=dimensions, client=client)
    return vec[0]


def _default_sample_pdf() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "GBT 1568-2008 键 技术条件.pdf"


def _first_chunk_as_text(chunk: dict[str, Any]) -> str:
    title = (chunk.get("title") or "").strip()
    body = (chunk.get("content") or "").strip()
    if title and body:
        return f"{title}\n{body}"
    return title or body


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR → 按标题分 chunk → 仅对第一个 chunk 调用 OpenAI Embedding 试跑"
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=None,
        help="PDF 路径；省略则使用 data/GBT 1568-2008 键 技术条件.pdf",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lang", type=str, default="chi_sim+eng")
    parser.add_argument("--first-page", type=int, default=None)
    parser.add_argument("--last-page", type=int, default=None)
    parser.add_argument("--preamble-title", type=str, default="")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="覆盖环境变量 OPENAI_EMBEDDING_MODEL",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="text-embedding-3-* 可选缩短维度",
    )
    parser.add_argument(
        "--text-preview",
        type=int,
        default=200,
        metavar="N",
        help="打印送入 embedding 的文本前 N 字（0 表示不截断打印）",
    )
    args = parser.parse_args()

    pdf_path = (args.pdf or _default_sample_pdf()).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"找不到 PDF：{pdf_path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        pages = extract_pdf_text_ocr(
            pdf_path,
            dpi=args.dpi,
            lang=args.lang,
            first_page=args.first_page,
            last_page=args.last_page,
        )
    except Exception as e:
        print(f"OCR 失败：{e}", file=sys.stderr)
        raise SystemExit(1) from e

    chunks = chunk_by_headings(pages, preamble_title=args.preamble_title)
    if not chunks:
        print("没有分出任何 chunk。", file=sys.stderr)
        raise SystemExit(1)

    c0 = chunks[0]
    text = _first_chunk_as_text(c0)
    if not text.strip():
        print("第一个 chunk 无可用文本（标题与正文皆空）。", file=sys.stderr)
        raise SystemExit(1)

    preview = text if args.text_preview == 0 else text[: args.text_preview]
    suffix = "" if args.text_preview == 0 or len(text) <= args.text_preview else "…"

    print(f"PDF：{pdf_path}")
    print(f"chunk 总数：{len(chunks)}（仅对第 1 个做 embedding）")
    print(f"第 1 节 title 字段：{c0.get('title')!r}")
    print(f"页码：{c0['start_page']} – {c0['end_page']}")
    print("-" * 40)
    print(f"送入 embedding 的文本（预览）：\n{preview}{suffix}")
    print("-" * 40)

    try:
        vec = embed_text(
            text,
            model=args.embedding_model,
            dimensions=args.dimensions,
        )
    except Exception as e:
        print(f"Embedding 失败：{e}", file=sys.stderr)
        raise SystemExit(1) from e

    head = vec[:8]
    print(f"向量维度：{len(vec)}")
    print(f"前 8 维：{head}")


if __name__ == "__main__":
    main()
