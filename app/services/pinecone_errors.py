"""User-facing hints when Pinecone calls fail (DNS, network, etc.)."""

from __future__ import annotations


def pinecone_connection_user_hint(exc: BaseException) -> str | None:
    """Return a short operational hint if this looks like a Pinecone reachability failure."""
    s = str(exc).lower()
    if "pinecone.io" not in s and "pinecone" not in s:
        return None
    markers = (
        "failed to resolve",
        "nodename nor servname",
        "name resolution",
        "name or service not known",
        "connection refused",
        "timed out",
        "timeout",
        "max retries exceeded",
        "temporary failure",
        "network is unreachable",
    )
    if not any(m in s for m in markers):
        return None
    return (
        "Cannot reach Pinecone (network/DNS or wrong index). "
        "Check internet and VPN; confirm PINECONE_INDEX_NAME in .env matches an index in the Pinecone console "
        "(same project as PINECONE_API_KEY). Restart the API after changing Pinecone settings."
    )
