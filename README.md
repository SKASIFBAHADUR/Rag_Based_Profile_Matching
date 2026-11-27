# Resume RAG Job Matching Engine

A Retrieval-Augmented Generation style resume search and job-matching project.

## Features

- Resume loading from TXT, PDF and DOCX
- Section-aware chunking
- Metadata extraction:
  - Candidate name
  - Skills
  - Experience years
  - Education
- HuggingFace embeddings using `all-MiniLM-L6-v2`
- ChromaDB vector storage
- Semantic resume retrieval
- Hybrid semantic + keyword skill matching
- Must-have experience filtering
- 0-100 ranking
- Relevant excerpts and match reasoning
- Retrieval latency measurement

## Project Structure

```text
resume-rag-job-matcher/
├── data/
│   ├── resumes/
│   └── job_descriptions/
├── src/
│   ├── fs_tools.py
│   ├── resume_rag.py
│   └── job_matcher.py
├── notebooks/
│   └── rag_experiments.ipynb
├── results/
├── chroma_db/
├── main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Build Vector Database

```bash
python src/resume_rag.py
```

## Run Job Matching

Example:

```bash
python src/job_matcher.py data/job_descriptions/python_developer.txt
```

Or run the complete demo:

```bash
python main.py
```

## Scoring

```text
Final Score =
70% Semantic Similarity
+
30% Keyword Skill Match
```

## Dataset

The starter project includes:
- 30 synthetic resumes
- 5 sample job descriptions

Replace the synthetic files with your assignment dataset if required.

## Metrics

The notebook demonstrates:
- Precision@10
- Recall@10
- Retrieval latency

## Suggested Pull Request

### Branch

```bash
git checkout -b feature/resume-rag-job-matcher
```

### Commit

```bash
git add .
git commit -m "Implement resume RAG and job matching engine"
git push origin feature/resume-rag-job-matcher
```

### PR Title

`Implement Resume RAG and Semantic Job Matching Engine`