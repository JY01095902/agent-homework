"""扫描版 PDF：PyMuPDF 渲页为图 + Tesseract OCR 提取文本（无需 Poppler）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


def _render_pdf_pages(
    pdf_path: str,
    *,
    dpi: int,
    first_page: Optional[int],
    last_page: Optional[int],
) -> List[Tuple[int, Image.Image]]:
    """
    用 PyMuPDF 按 DPI 光栅化指定页面。

    返回 ``[(页码 1-based, PIL.Image), ...]``。
    """
    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        if n == 0:
            return []

        start0 = (first_page - 1) if first_page is not None else 0
        end0 = (last_page - 1) if last_page is not None else n - 1
        start0 = max(0, min(start0, n - 1))
        end0 = max(start0, min(end0, n - 1))

        out: List[Tuple[int, Image.Image]] = []
        for idx in range(start0, end0 + 1):
            page = doc.load_page(idx)
            try:
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            except TypeError:
                zoom = dpi / 72.0
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom),
                    colorspace=fitz.csRGB,
                    alpha=False,
                )

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out.append((idx + 1, img))
        return out
    finally:
        doc.close()


def extract_pdf_text_ocr(
    pdf_path: Union[str, Path],
    *,
    dpi: int = 200,
    lang: str = "chi_sim+eng",
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    tesseract_config: str = "",
) -> List[Dict[str, Any]]:
    """
    将 PDF 每页渲染为图片后做 OCR，返回按页文本。

    依赖：

    - **PyMuPDF**（``pymupdf``）：将 PDF 页渲染为位图，**不需要** Poppler / pdf2image。
    - **Tesseract** + 语言包：pytesseract（macOS 可用 conda-forge 或系统包管理器安装）。

    参数
    ----
    pdf_path
        PDF 文件路径。
    dpi
        渲染分辨率；过低影响识别，过高变慢、占内存。
    lang
        Tesseract 语言，默认简体中文 + 英文（``chi_sim+eng``）。
    first_page / last_page
        仅处理闭区间页码，均为 **1-based**；省略则处理全部页面。
    tesseract_config
        传给 Tesseract 的额外配置串，例如 ``--psm 6``。

    返回
    ----
    ``[{"page": 1, "text": "..."}, ...]``，``text`` 已 ``strip``。
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {path}")

    rendered = _render_pdf_pages(
        str(path), dpi=dpi, first_page=first_page, last_page=last_page
    )

    out: List[Dict[str, Any]] = []
    for page_no, pil in rendered:
        text = pytesseract.image_to_string(pil, lang=lang, config=tesseract_config)
        out.append({"page": page_no, "text": text.strip()})
    return out


def extract_pdf_plain_text_ocr(
    pdf_path: Union[str, Path],
    *,
    page_separator: str = "\n\n",
    **kwargs: Any,
) -> str:
    """
    与 :func:`extract_pdf_text_ocr` 相同管线，返回整份 PDF 拼接成的一段纯文本。
    ``kwargs`` 会原样传给 :func:`extract_pdf_text_ocr`。
    """
    pages = extract_pdf_text_ocr(pdf_path, **kwargs)
    return page_separator.join(p["text"] for p in pages if p["text"])


def _default_sample_pdf() -> Path:
    """仓库内示例 PDF：``data/GBT 1568-2008 键 技术条件.pdf``。"""
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "GBT 1568-2008 键 技术条件.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyMuPDF 渲页 + Tesseract：从 PDF 抽取 OCR 文本并打印（无需 Poppler）"
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=None,
        help="PDF 路径；省略则使用 data/GBT 1568-2008 键 技术条件.pdf",
    )
    parser.add_argument("--dpi", type=int, default=200, help="渲染 DPI（默认 200）")
    parser.add_argument("--lang", type=str, default="chi_sim+eng", help="Tesseract 语言")
    parser.add_argument("--first-page", type=int, default=None, help="起始页（1-based）")
    parser.add_argument("--last-page", type=int, default=None, help="结束页（1-based）")
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="每页只打印前 N 个字符，0 表示整页打印",
    )
    args = parser.parse_args()

    pdf_path = (args.pdf or _default_sample_pdf()).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"找不到 PDF：{pdf_path}", file=sys.stderr)
        print("请将文件放到上述路径，或通过参数传入 pdf 路径。", file=sys.stderr)
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
        print(f"解析失败：{e}", file=sys.stderr)
        raise SystemExit(1) from e

    print(f"文件：{pdf_path}")
    print(f"页数：{len(pages)}")
    print("-" * 60)
    for row in pages:
        text = row["text"]
        if args.preview and args.preview > 0:
            text = text[: args.preview] + ("…" if len(row["text"]) > args.preview else "")
        print(f"\n=== 第 {row['page']} 页 ===\n{text}\n")


if __name__ == "__main__":
    main()
