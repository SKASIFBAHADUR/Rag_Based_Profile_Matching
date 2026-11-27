import re
from pathlib import Path
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

from fs_tools import list_files, read_file

SECTION_HEADERS = [
    "summary", "professional summary", "skills", "technical skills",
    "experience", "professional experience", "work experience",
    "education", "projects", "certifications"
]

KNOWN_SKILLS = [
    "python", "java", "spring boot", "sql", "machine learning",
    "deep learning", "nlp", "aws", "azure", "docker", "kubernetes",
    "selenium", "playwright", "rest assured", "javascript",
    "typescript", "react", "node.js", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "power bi", "tableau"
]


def normalize_header(line: str) -> str:
    return re.sub(r"[^a-z ]", "", line.lower()).strip()


def split_into_sections(text: str) -> List[Dict]:
    sections = []
    current_section = "general"
    current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized = normalize_header(line)

        if normalized in SECTION_HEADERS:
            if current_lines:
                sections.append({
                    "section": current_section,
                    "text": "\n".join(current_lines).strip()
                })
            current_section = normalized
            current_lines = []
        else:
            current_lines.append(raw_line)

    if current_lines:
        sections.append({
            "section": current_section,
            "text": "\n".join(current_lines).strip()
        })

    return [s for s in sections if s["text"]]


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def extract_name(text: str, file_path: str) -> str:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        first = lines[0]
        if 1 <= len(first.split()) <= 5 and len(first) < 80:
            return first
    return Path(file_path).stem.replace("_", " ").title()


def extract_skills(text: str) -> List[str]:
    lower = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        if skill in lower:
            found.append(skill.title())
    return sorted(set(found))


def extract_experience_years(text: str) -> int:
    patterns = [
        r"(\d+)\+?\s+years?\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+)\+?\s+years?"
    ]
    values = []
    for pattern in patterns:
        for match in re.findall(pattern, text.lower()):
            try:
                values.append(int(match))
            except ValueError:
                pass
    return max(values) if values else 0


def extract_education(text: str) -> str:
    candidates = [
        line.strip() for line in text.splitlines()
        if any(k in line.lower() for k in [
            "b.tech", "bachelor", "master", "m.tech", "mba",
            "computer science", "engineering", "university", "college"
        ])
    ]
    return candidates[0][:200] if candidates else "Not specified"


def extract_metadata(text: str, file_path: str) -> Dict:
    return {
        "candidate_name": extract_name(text, file_path),
        "skills": ", ".join(extract_skills(text)),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
        "resume_path": file_path
    }


class ResumeRAG:
    def __init__(
        self,
        db_path: str = "chroma_db",
        collection_name: str = "resumes",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_resumes(self, resume_folder: str = "data/resumes") -> int:
        files = list_files(resume_folder)
        ids, documents, embeddings, metadatas = [], [], [], []

        counter = 0
        for file_path in files:
            text = read_file(file_path)
            metadata = extract_metadata(text, file_path)
            sections = split_into_sections(text)

            for section in sections:
                for chunk in chunk_text(section["text"]):
                    chunk_id = f"{Path(file_path).stem}_{counter}"
                    chunk_metadata = dict(metadata)
                    chunk_metadata["section"] = section["section"]

                    ids.append(chunk_id)
                    documents.append(chunk)
                    embeddings.append(self.model.encode(chunk).tolist())
                    metadatas.append(chunk_metadata)
                    counter += 1

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        return len(ids)


if __name__ == "__main__":
    rag = ResumeRAG()
    count = rag.index_resumes()
    print(f"Indexed {count} resume chunks.")