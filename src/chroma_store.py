"""将分块文本与 OpenAI 向量写入 Chroma 持久化向量库。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from chunking import chunk_by_headings
from embeddings import embed_texts
from pdf_extract import extract_pdf_text_ocr


def chunk_to_document_text(chunk: Dict[str, Any]) -> str:
    """与 embedding 一致的拼接：有标题则 ``标题\\n正文``。"""
    title = (chunk.get("title") or "").strip()
    body = (chunk.get("content") or "").strip()
    if title and body:
        return f"{title}\n{body}"
    return title or body


def export_parsed_text(
    pages: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    export_dir: Path,
    *,
    pdf_stem: str,
) -> tuple[Path, Path]:
    """将 OCR 分页与 chunk 落成纯文本，仅供人工查看或调试。"""
    export_dir = Path(export_dir).expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pdf_stem)[:200]

    ocr_path = export_dir / f"{stem}_ocr_pages.txt"
    ocr_parts: List[str] = []
    for p in pages:
        page_no = int(p["page"])
        body = (p.get("text") or "").strip()
        ocr_parts.append(f"\n\n--- 第 {page_no} 页 ---\n\n")
        ocr_parts.append(body)
    ocr_path.write_text("".join(ocr_parts).lstrip() + "\n", encoding="utf-8")

    chunks_path = export_dir / f"{stem}_chunks.txt"
    chunk_parts: List[str] = []
    for i, c in enumerate(chunks):
        title_bit = (c.get("title") or "").strip() or "(无标题)"
        chunk_parts.append(
            f"\n\n===== chunk {i} | "
            f"p.{c['start_page']}-{c['end_page']} | {title_bit} =====\n\n"
        )
        chunk_parts.append(chunk_to_document_text(c).strip())
    chunks_path.write_text("".join(chunk_parts).lstrip() + "\n", encoding="utf-8")

    return ocr_path, chunks_path


def save_chunks_to_chroma(
    chunks: List[Dict[str, Any]],
    *,
    persist_directory: Path,
    collection_name: str = "pdf_chunks",
    source_pdf: Optional[str] = None,
    reset: bool = False,
    embed_model: Optional[str] = None,
    dimensions: Optional[int] = None,
    batch_size: int = 64,
) -> int:
    """
    对 ``chunks`` 逐条做 OpenAI Embedding，写入 Chroma（磁盘持久化）。

    - ``persist_directory``：Chroma 数据目录（不存在会自动创建）。
    - ``reset=True``：若集合已存在则先删除再重建。
    - 元数据字段：``title``, ``start_page``, ``end_page``, ``chunk_index``, ``source``（PDF 路径字符串）。

    返回成功写入的条数（跳过无文本 chunk）。
    """
    if not chunks:
        return 0

    persist_directory = Path(persist_directory).expanduser().resolve()
    persist_directory.mkdir(parents=True, exist_ok=True)

    rows: List[tuple[int, Dict[str, Any], str]] = []
    for i, c in enumerate(chunks):
        text = chunk_to_document_text(c).strip()
        if not text:
            continue
        rows.append((i, c, text))

    if not rows:
        return 0

    texts = [t for _, _, t in rows]
    embeddings = embed_texts(
        texts,
        model=embed_model,
        dimensions=dimensions,
        batch_size=batch_size,
    )

    ids = [f"chunk_{i:05d}" for i, _, _ in rows]
    documents = texts
    metadatas: List[Dict[str, Any]] = []
    src = source_pdf or ""
    for idx, (orig_i, c, _) in enumerate(rows):
        metadatas.append(
            {
                "title": (c.get("title") or "")[:512],
                "start_page": int(c["start_page"]),
                "end_page": int(c["end_page"]),
                "chunk_index": int(orig_i),
                "source": src[:1024],
            }
        )

    client = chromadb.PersistentClient(path=str(persist_directory))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(ids)


def _default_sample_pdf() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "GBT 1568-2008 键 技术条件.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR → 分 chunk → OpenAI Embedding → 写入 Chroma 持久化目录"
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
        "--chroma-path",
        type=Path,
        default=None,
        help="Chroma 持久化目录；默认项目下 data/chroma_db",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="pdf_chunks",
        help="集合名称",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="启动前删除同名集合再重建",
    )
    parser.add_argument(
        "--export-text-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "可选：仅为方便本地查看/调试，导出 OCR 分页与分块纯文本到 DIR "
            "（文件名含 PDF 主名后缀 _ocr_pages.txt / _chunks.txt），"
            "不影响向量库内容；容器内需挂卷到宿主才能在本机看到这些文件"
        ),
    )
    parser.add_argument(
        "--export-text",
        action="store_true",
        help="与「--export-text-dir 不写路径」等价：导出到默认目录项目下 data/parsed",
    )
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--dimensions", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    export_dir: Optional[Path] = args.export_text_dir
    if export_dir is None and args.export_text:
        export_dir = project_root / "data" / "parsed"

    pdf_path = (args.pdf or _default_sample_pdf()).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"找不到 PDF：{pdf_path}", file=sys.stderr)
        raise SystemExit(1)

    chroma_dir = args.chroma_path or (
        Path(__file__).resolve().parent.parent / "data" / "chroma_db"
    )
    chroma_dir = chroma_dir.expanduser().resolve()

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

    if export_dir is not None:
        stem = pdf_path.stem
        ocr_f, ch_f = export_parsed_text(pages, chunks, export_dir, pdf_stem=stem)
        print(f"已导出 OCR 分页：{ocr_f}")
        print(f"已导出分块文本：{ch_f}")

    try:
        n = save_chunks_to_chroma(
            chunks,
            persist_directory=chroma_dir,
            collection_name=args.collection,
            source_pdf=str(pdf_path),
            reset=args.reset,
            embed_model=args.embedding_model,
            dimensions=args.dimensions,
        )
    except Exception as e:
        print(f"写入 Chroma 失败：{e}", file=sys.stderr)
        raise SystemExit(1) from e

    print(f"PDF：{pdf_path}")
    print(f"分块数：{len(chunks)}  |  已写入向量：{n}")
    print(f"Chroma 目录：{chroma_dir}")
    print(f"集合名：{args.collection}")


if __name__ == "__main__":
    main()
