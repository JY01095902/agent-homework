"""将 OCR 分页结果按「标题行」拆成 chunk（适合国标/技术文档常见编号标题）。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

from pdf_extract import extract_pdf_text_ocr

# ``extract_pdf_text_ocr`` 的单页结构
PageText = Dict[str, Any]


def default_heading_patterns() -> List[Pattern[str]]:
    """
    默认标题行规则（命中任一则开启新 chunk）：

    - ``1 范围`` / ``3.1 外观`` 等：数字节号 + 空格 + 标题
    - ``第一章`` / ``第 3 节`` 等
    - ``附录 A`` 等
    - Markdown 风格 ``# 标题``（1–6 个 #）
    """
    return [
        re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(\S.*)$"),
        re.compile(
            r"^\s*(第\s*[0-9一二三四五六七八九十百千万零〇两]+\s*[章节篇部])\s*(.*)$"
        ),
        re.compile(r"^\s*(附录\s*[A-Za-z0-9IVX]+)\s*(.*)$"),
        re.compile(r"^#{1,6}\s+\S.*$"),
    ]


def _flatten_lines_with_page(pages: List[PageText]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for block in pages:
        page_no = int(block["page"])
        text = (block.get("text") or "").replace("\r\n", "\n").replace("\r", "\n")
        for line in text.split("\n"):
            out.append((page_no, line))
    return out


def _is_heading(line_stripped: str, patterns: List[Pattern[str]]) -> bool:
    if not line_stripped:
        return False
    return any(p.match(line_stripped) for p in patterns)


def chunk_by_headings(
    pages: List[PageText],
    *,
    patterns: Optional[List[Pattern[str]]] = None,
    preamble_title: str = "",
) -> List[Dict[str, Any]]:
    """
    按标题行把 ``extract_pdf_text_ocr`` 的结果切成多块。

    每个 chunk：

    - ``title``：当前节标题（命中标题规则的那一行原文，去掉首尾空白）
    - ``content``：该节正文（**不含**标题行本身；允许为空字符串）
    - ``start_page`` / ``end_page``：标题行与正文行涉及的最小/最大页码

    第一个标题出现前的文字归入一节，``title`` 为 ``preamble_title``（默认空字符串）。

    OCR 断行、编号识别错误会导致分节不准，可通过 ``patterns`` 传入自定义正则微调。
    """
    pats = patterns if patterns is not None else default_heading_patterns()
    lines = _flatten_lines_with_page(pages)

    chunks: List[Dict[str, Any]] = []

    pending_title = preamble_title
    title_page: Optional[int] = None
    buf: List[Tuple[int, str]] = []

    def finalize() -> None:
        nonlocal pending_title, title_page, buf
        if title_page is None and not buf:
            return
        page_nums: List[int] = []
        if title_page is not None:
            page_nums.append(title_page)
        page_nums.extend(p for p, _ in buf)
        if not page_nums:
            return
        body = "\n".join(ln for _, ln in buf).strip()
        if pending_title == preamble_title and not body.strip():
            return
        chunks.append(
            {
                "title": pending_title,
                "content": body,
                "start_page": min(page_nums),
                "end_page": max(page_nums),
            }
        )
        buf = []

    for page_no, line in lines:
        stripped = line.strip()
        if _is_heading(stripped, pats):
            finalize()
            pending_title = stripped
            title_page = page_no
            continue
        buf.append((page_no, line))

    finalize()

    if not chunks and lines:
        sp = min(p for p, _ in lines)
        ep = max(p for p, _ in lines)
        body = "\n".join(ln for _, ln in lines).strip()
        chunks.append(
            {
                "title": preamble_title,
                "content": body,
                "start_page": sp,
                "end_page": ep,
            }
        )

    return chunks


def _default_sample_pdf() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "GBT 1568-2008 键 技术条件.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF OCR（PyMuPDF+Tesseract）后按标题分 chunk 并打印"
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=None,
        help="PDF 路径；省略则使用 data/GBT 1568-2008 键 技术条件.pdf",
    )
    parser.add_argument("--dpi", type=int, default=200, help="渲染 DPI")
    parser.add_argument("--lang", type=str, default="chi_sim+eng", help="Tesseract 语言")
    parser.add_argument("--first-page", type=int, default=None, help="OCR 起始页（1-based）")
    parser.add_argument("--last-page", type=int, default=None, help="OCR 结束页（1-based）")
    parser.add_argument(
        "--preamble-title",
        type=str,
        default="",
        metavar="STR",
        help="第一个标题前内容的节标题（默认空）",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="每个 chunk 的正文只打印前 N 字，0 表示全文",
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

    print(f"文件：{pdf_path}")
    print(f"OCR 页数：{len(pages)}  |  chunk 数：{len(chunks)}")
    print("=" * 60)

    for i, c in enumerate(chunks, start=1):
        title = c["title"] or "（无标题/文头）"
        body = c["content"]
        if args.preview and args.preview > 0:
            body = body[: args.preview] + ("…" if len(c["content"]) > args.preview else "")
        print(f"\n>>> Chunk {i}")
        print(f"标题：{title}")
        print(f"页码：{c['start_page']} – {c['end_page']}")
        print("-" * 40)
        print(body or "（正文为空）")
        print()


if __name__ == "__main__":
    main()
