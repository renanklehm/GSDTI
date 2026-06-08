from __future__ import annotations

import argparse
from pathlib import Path

from markitdown import MarkItDown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF file to Markdown using MarkItDown."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the source PDF file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf_path.expanduser().resolve()

    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Expected a .pdf file, got: {pdf_path}")

    if not pdf_path.is_file():
        raise SystemExit(f"PDF file not found: {pdf_path}")

    markdown_path = pdf_path.with_suffix(".md")

    result = MarkItDown().convert(str(pdf_path))
    markdown_path.write_text(result.text_content, encoding="utf-8")

    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
