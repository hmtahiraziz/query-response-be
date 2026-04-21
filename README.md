# Portfolio cover letter — backend

FastAPI service that ingests portfolio PDFs into Pinecone (OpenAI embeddings), serves RAG-backed cover letter generation, and optionally stores history in MongoDB.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set at least `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME`. See comments in `.env.example` for optional MongoDB and CORS settings.

3. Create the Pinecone serverless index (cosine, dimension = `OPENAI_EMBED_DIMENSIONS`, default 3072):

   ```bash
   python scripts/create_pinecone_index.py
   ```

4. Run the API (default port `8000`):

   ```bash
   uvicorn app.main:app --reload
   ```

   OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Layout

- `scripts/` — utilities (e.g. `create_pinecone_index.py`)
- `app/main.py` — FastAPI app factory (middleware, router mount)
- `app/core/` — settings, paths, application lifespan
- `app/schemas/` — Pydantic API contracts (requests/responses)
- `app/api/` — HTTP routers (`router.py`, `deps.py`, `routes/` per domain)
- `app/services/` — business logic, integrations (OpenAI, Pinecone, Mongo, files)
- `app/rules/cover_letter_assistant.default.json` — bundled cover-letter assistant rules (policy + generation bullets); edit in-repo, no DB/UI persistence
- `data/` — local PDF storage and JSON fallback for history (gitignored)
- `requirements.txt` — Python dependencies

The Next.js UI expects the API at `http://127.0.0.1:8000` unless `NEXT_PUBLIC_API_URL` is set in the frontend.
