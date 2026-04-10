# Portfolio cover letter — backend

FastAPI service that ingests portfolio PDFs into Pinecone (Gemini embeddings), serves RAG-backed cover letter generation, and optionally stores history in MongoDB.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set at least `GEMINI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME`. See comments in `.env.example` for optional MongoDB and CORS settings.

3. Run the API (default port `8000`):

   ```bash
   uvicorn app.main:app --reload
   ```

   OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Layout

- `app/` — FastAPI app, config, and services
- `data/` — Local PDF storage and JSON fallback for history (gitignored)
- `requirements.txt` — Python dependencies

The Next.js UI expects the API at `http://127.0.0.1:8000` unless `NEXT_PUBLIC_API_URL` is set in the frontend.
