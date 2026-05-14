import io
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """从 PDF 字节流中提取纯文本"""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_metadata(file_bytes: bytes) -> dict:
    """提取 PDF 元数据（页数、作者等）"""
    reader = PdfReader(io.BytesIO(file_bytes))
    return {
        "pages": len(reader.pages),
        "metadata": dict(reader.metadata) if reader.metadata else {}
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            content = f.read()
        info = extract_metadata(content)
        text = extract_text_from_pdf(content)
        print(f"页数: {info['pages']}")
        print(f"文本长度: {len(text)} 字符")
        print("--- 前 500 字符 ---")
        print(text[:500])
    else:
        print("用法: python pdf_parser.py <pdf_file_path>")
