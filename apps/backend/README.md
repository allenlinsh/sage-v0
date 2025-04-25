# Resume Screening & Candidate Ranking Backend

Backend system that automates resume screening and candidate ranking based on job descriptions.

## Features

- **Resume Parsing**: Extracts skills, education, experience from resumes
- **Ranking**: Uses BM25 algorithm to score and rank candidates
- **Re-ranking**: LLM-based reranking of top candidates (via OpenAI API)
- **Evaluation**: Sophisticated evaluation of ranking quality

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit the `.env` file with your OpenAI API key.

## Usage

Start the API server:
```bash
uvicorn main:app --reload
```

Test the pipeline:
```bash
curl -X POST http://localhost:8000/run -F 'job_description=Frontend Developer, React, 3+ years, remote-friendly'
```

## API Endpoints

- `POST /parse-resumes`: Parse uploaded resumes
- `POST /rank-candidates`: Rank candidates based on job description
- `POST /rerank-candidates`: Re-rank top candidates using LLM
- `POST /evaluate-ranking`: Evaluate ranking quality
- `POST /complete-pipeline`: Run the full pipeline 