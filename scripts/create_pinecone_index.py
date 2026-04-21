#!/usr/bin/env python3
"""Create a Pinecone serverless index for this app (cosine, dim = OpenAI embedding size).

Run from the backend folder (uses ``.env`` in that directory):

    cd backend
    python scripts/create_pinecone_index.py

Uses ``PINECONE_API_KEY``, ``PINECONE_INDEX_NAME``, and ``OPENAI_EMBED_DIMENSIONS`` from
``.env`` unless overridden with flags. Safe to re-run: exits cleanly if the index already exists.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    load_dotenv(BACKEND_ROOT / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=os.environ.get("PINECONE_INDEX_NAME", "").strip(),
        help="Index name (default: PINECONE_INDEX_NAME from .env)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=int(os.environ.get("OPENAI_EMBED_DIMENSIONS", "3072")),
        help="Vector dimension (default: OPENAI_EMBED_DIMENSIONS, usually 3072 for text-embedding-3-large)",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=("cosine", "dotproduct", "euclidean"),
        help="Distance metric (default: cosine, matches the app)",
    )
    parser.add_argument(
        "--cloud",
        default=os.environ.get("PINECONE_SERVERLESS_CLOUD", "aws").strip().lower(),
        help="Serverless cloud (default: aws or PINECONE_SERVERLESS_CLOUD)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("PINECONE_SERVERLESS_REGION", "us-east-1").strip(),
        help="Serverless region (default: us-east-1 or PINECONE_SERVERLESS_REGION)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds to wait for the index to become ready (default: 300)",
    )
    args = parser.parse_args()

    api_key = (os.environ.get("PINECONE_API_KEY") or "").strip()
    if not api_key:
        print("error: PINECONE_API_KEY is missing. Set it in backend/.env", file=sys.stderr)
        return 1
    if not args.name:
        print(
            "error: index name missing. Set PINECONE_INDEX_NAME in .env or pass --name",
            file=sys.stderr,
        )
        return 1

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=api_key)
    existing = set(pc.list_indexes().names())
    if args.name in existing:
        print(f"Index '{args.name}' already exists — nothing to do.")
        return 0

    print(
        f"Creating serverless index '{args.name}' "
        f"(dimension={args.dimension}, metric={args.metric}, "
        f"cloud={args.cloud}, region={args.region})…"
    )
    pc.create_index(
        name=args.name,
        dimension=args.dimension,
        metric=args.metric,
        spec=ServerlessSpec(cloud=args.cloud, region=args.region),
        timeout=args.timeout,
    )
    print(f"Ready. Set PINECONE_INDEX_NAME={args.name} in .env if it is not already.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
