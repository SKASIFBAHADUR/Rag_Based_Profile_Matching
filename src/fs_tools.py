from pathlib import Path
from typing import List
from PyPDF2 import PdfReader
from docx import Document

def read_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {ext}")


def list_files(folder_path: str, extensions=None) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        return []

    extensions = extensions or [".pdf", ".docx", ".txt"]
    extensions = {e.lower() for e in extensions}

    return [
        str(p)
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def write_file(file_path: str, content: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def search_in_file(file_path: str, keyword: str) -> List[str]:
    text = read_file(file_path)
    return [
        line for line in text.splitlines()
        if keyword.lower() in line.lower()
    ]