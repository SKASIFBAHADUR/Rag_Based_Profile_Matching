import json
import re
import time
from collections import defaultdict
from typing import Dict, List

from sentence_transformers import SentenceTransformer
import chromadb

CRITICAL_SKILLS = [
    "python", "java", "spring boot", "sql", "machine learning", "aws",
    "azure", "docker", "kubernetes", "selenium", "playwright",
    "rest assured", "javascript", "typescript", "react", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch"
]


def extract_required_skills(job_description: str) -> List[str]:
    lower = job_description.lower()
    return [s for s in CRITICAL_SKILLS if s in lower]


def extract_min_years(job_description: str) -> int:
    matches = re.findall(r"(\d+)\+?\s*years?", job_description.lower())
    return max([int(x) for x in matches], default=0)


class JobMatcher:
    def __init__(
        self,
        db_path: str = "chroma_db",
        collection_name: str = "resumes",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def _keyword_score(self, jd_skills: List[str], resume_skills: str) -> float:
        if not jd_skills:
            return 1.0
        resume_skills_lower = resume_skills.lower()
        matched = [s for s in jd_skills if s in resume_skills_lower]
        return len(matched) / len(jd_skills)

    def search(self, job_description: str, top_k: int = 10) -> Dict:
        start = time.perf_counter()

        query_embedding = self.model.encode(job_description).tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(50, max(top_k * 5, 20))
        )

        required_skills = extract_required_skills(job_description)
        min_years = extract_min_years(job_description)

        grouped = defaultdict(lambda: {
            "semantic_scores": [],
            "documents": [],
            "sections": [],
            "metadata": None
        })

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]

        for doc, meta, distance in zip(docs, metas, distances):
            candidate = meta["candidate_name"]
            semantic_similarity = max(0.0, min(1.0, 1.0 - float(distance)))

            grouped[candidate]["semantic_scores"].append(semantic_similarity)
            grouped[candidate]["documents"].append(doc)
            grouped[candidate]["sections"].append(meta.get("section", "general"))
            grouped[candidate]["metadata"] = meta

        matches = []

        for candidate, info in grouped.items():
            meta = info["metadata"]
            years = int(meta.get("experience_years", 0))

            if min_years and years < min_years:
                continue

            keyword_score = self._keyword_score(required_skills, meta.get("skills", ""))
            semantic_score = max(info["semantic_scores"]) if info["semantic_scores"] else 0

            final_score = (semantic_score * 0.70) + (keyword_score * 0.30)
            matched_skills = [
                s.title() for s in required_skills
                if s in meta.get("skills", "").lower()
            ]

            best_idx = sorted(
                range(len(info["semantic_scores"])),
                key=lambda i: info["semantic_scores"][i],
                reverse=True
            )[:3]

            excerpts = [info["documents"][i][:350] for i in best_idx]
            matched_sections = sorted(set(info["sections"][i] for i in best_idx))

            reasoning = (
                f"Semantic similarity was strong in sections: "
                f"{', '.join(matched_sections)}. "
                f"Matched critical skills: "
                f"{', '.join(matched_skills) if matched_skills else 'None explicitly detected'}."
            )

            matches.append({
                "candidate_name": candidate,
                "resume_path": meta.get("resume_path"),
                "match_score": round(final_score * 100, 2),
                "matched_skills": matched_skills,
                "relevant_excerpts": excerpts,
                "reasoning": reasoning
            })

        matches.sort(key=lambda x: x["match_score"], reverse=True)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "job_description": job_description,
            "top_matches": matches[:top_k],
            "latency_ms": latency_ms
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        jd_path = Path(sys.argv[1])
        jd = jd_path.read_text(encoding="utf-8")
    else:
        jd = input("Paste job description: ")

    matcher = JobMatcher()
    output = matcher.search(jd, top_k=10)
    print(json.dumps(output, indent=2))