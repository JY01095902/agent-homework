"""RAG：问题向量化 → Chroma 相似检索 → 上下文注入 LLM → 返回答案与来源。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from openai import OpenAI

from embeddings import embed_text
from llm import chat, get_openai_client


DEFAULT_RAG_SYSTEM = (
    "你是严谨的技术文档助手。请仅根据用户消息中给出的「检索片段」回答问题；"
    "不要使用片段外的臆测。若片段不足以回答，请明确说明无法从所给材料中得出答案。"
    "回答尽量简洁，可引用页码。"
)


def _distance_to_confidence(distance: float) -> float:
    """将 Chroma 返回的距离粗略映射到 (0,1]，仅作展示用。"""
    d = float(distance)
    if d < 0:
        d = 0.0
    return max(0.05, min(1.0, 1.0 / (1.0 + d)))


def _meta_page(meta: Dict[str, Any], key: str) -> int:
    v = meta.get(key)
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def rag_answer(
    question: str,
    *,
    persist_directory: Union[str, Path],
    collection_name: str = "pdf_chunks",
    top_k: int = 4,
    embed_model: Optional[str] = None,
    dimensions: Optional[int] = None,
    llm_model: Optional[str] = None,
    temperature: float = 0.3,
    client: Optional[OpenAI] = None,
    system_prompt: Optional[str] = None,
    preview_chars: int = 220,
) -> Dict[str, Any]:
    """
    基本 RAG 流程：问题 Embedding → Chroma 向量检索 → 将片段与用户问题拼成提示 → ``llm.chat`` 生成答案。

    参数
    ----
    question
        用户自然语言问题。
    persist_directory
        Chroma 持久化目录（与 ``save_chunks_to_chroma`` 一致）。
    collection_name
        集合名称。
    top_k
        检索返回的片段条数上限（受集合内文档数限制）。
    embed_model / dimensions
        须与入库时使用的 Embedding 模型及维度一致（见 ``embed_text``）。
    llm_model / temperature / client
        传给 ``llm.chat``；``client`` 同时用于 Embedding 与对话（未传则 ``get_openai_client()``）。
    system_prompt
        系统提示；默认 ``DEFAULT_RAG_SYSTEM``。
    preview_chars
        ``sources`` 里 ``content_preview`` 的最大字符数。

    返回
    ----
    ``{"answer": str, "sources": [...], "confidence": float|None}``。

    ``sources`` 每项含 ``page``（取 ``start_page``）、``content_preview``、可选 ``title``、``distance``。
    ``confidence`` 由最优命中距离粗略换算，仅作参考。
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("问题不能为空")

    c = client or get_openai_client()
    persist = Path(persist_directory).expanduser().resolve()

    ch_client = chromadb.PersistentClient(path=str(persist))
    try:
        collection = ch_client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(
            f"无法打开 Chroma 集合「{collection_name}」（路径 {persist}）：{e}"
        ) from e

    n_docs = collection.count()
    if n_docs == 0:
        return {
            "answer": "知识库中暂无可用文档片段，无法基于检索回答问题。请先写入向量库（如运行 chroma_store）。",
            "sources": [],
            "confidence": None,
        }

    q_emb = embed_text(q, model=embed_model, dimensions=dimensions, client=c)
    k = max(1, min(int(top_k), n_docs))

    raw = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (raw.get("documents") or [[]])[0] or []
    metas = (raw.get("metadatas") or [[]])[0] or []
    dists = (raw.get("distances") or [[]])[0] or []

    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs):
        text = doc or ""
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None

        title = (meta.get("title") or "") if isinstance(meta, dict) else ""
        sp = _meta_page(meta, "start_page") if isinstance(meta, dict) else 0
        ep = _meta_page(meta, "end_page") if isinstance(meta, dict) else sp

        if sp and ep and ep != sp:
            page_label = f"第{sp}–{ep}页"
        elif sp:
            page_label = f"第{sp}页"
        else:
            page_label = "页码未知"

        ttl = f"，标题：{title}" if title else ""
        context_blocks.append(f"【片段{i + 1}】（{page_label}{ttl}）\n{text.strip()}")

        preview = text.strip().replace("\n", " ")
        if len(preview) > preview_chars:
            preview = preview[:preview_chars] + "…"

        src: Dict[str, Any] = {
            "page": sp if sp else 0,
            "content_preview": preview,
        }
        if title:
            src["title"] = title
        if dist is not None:
            src["distance"] = float(dist)
        sources.append(src)

    context = "\n\n".join(context_blocks)
    user_content = (
        f"以下是检索到的文档片段（按相关度排序）：\n\n{context}\n\n"
        f"用户问题：{q}\n\n请基于上述片段作答。"
    )

    sys_p = system_prompt if system_prompt is not None else DEFAULT_RAG_SYSTEM
    answer = chat(
        [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_content},
        ],
        model=llm_model,
        temperature=temperature,
        client=c,
    )

    confidence: Optional[float] = None
    dist_vals = [float(x) for x in dists if x is not None]
    if dist_vals:
        confidence = _distance_to_confidence(min(dist_vals))

    return {
        "answer": answer.strip(),
        "sources": sources,
        "confidence": confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG 试跑：问题向量化 → Chroma 检索 → LLM 作答"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="这份文档主要内容是什么？",
        help="用户问题",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Chroma 持久化目录；默认 项目/data/chroma_db",
    )
    parser.add_argument("--collection", type=str, default="pdf_chunks")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--json",
        action="store_true",
        help="整段结果以 JSON 打印（便于脚本解析）",
    )
    args = parser.parse_args()

    chroma_dir = args.chroma_path or (
        Path(__file__).resolve().parent.parent / "data" / "chroma_db"
    )

    try:
        out = rag_answer(
            args.question,
            persist_directory=chroma_dir,
            collection_name=args.collection,
            top_k=args.top_k,
            embed_model=args.embedding_model,
            dimensions=args.dimensions,
            llm_model=args.llm_model,
            temperature=args.temperature,
        )
    except ValueError as e:
        print(f"参数错误：{e}", file=sys.stderr)
        raise SystemExit(1) from e
    except Exception as e:
        print(f"RAG 失败：{e}", file=sys.stderr)
        raise SystemExit(1) from e

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"Chroma：{chroma_dir.resolve()}")
    print(f"集合：{args.collection}  |  top_k={args.top_k}")
    print("=" * 50)
    print(out["answer"])
    print("=" * 50)
    print(f"confidence（参考）: {out['confidence']}")
    print("来源：")
    for i, s in enumerate(out["sources"], start=1):
        title = s.get("title", "")
        dist = s.get("distance")
        extra = f"  |  {title}" if title else ""
        dtxt = f"  distance={dist:.4f}" if dist is not None else ""
        print(f"  [{i}] 第{s.get('page', 0)}页{dtxt}{extra}")
        print(f"      {s.get('content_preview', '')}")


if __name__ == "__main__":
    main()
