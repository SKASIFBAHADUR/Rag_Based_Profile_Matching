import json
from pathlib import Path
import sys

sys.path.append("src")

from resume_rag import ResumeRAG
from job_matcher import JobMatcher

def main():
    rag = ResumeRAG()
    indexed = rag.index_resumes("data/resumes")
    print(f"Indexed chunks: {indexed}")

    jd_files = list(Path("data/job_descriptions").glob("*.txt"))
    if not jd_files:
        raise FileNotFoundError("No job descriptions found.")

    jd_path = jd_files[0]
    job_description = jd_path.read_text(encoding="utf-8")

    matcher = JobMatcher()
    result = matcher.search(job_description, top_k=10)

    Path("results").mkdir(exist_ok=True)
    output_path = Path("results/sample_results.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()