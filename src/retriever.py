import re
from src.embeddings import get_embeddings
from src.config import COLLECTION_NAME
from src.vectorstores import get_qdrant_client

TOP_SCORE_THRESHOLD = 0.70

# Minimum absolute score every individual chunk must meet.
# Prevents low-scoring chunks from a WRONG PDF sneaking in
# just because they're within 85% of a mediocre best score.
MIN_CHUNK_SCORE = 0.70


def normalize_query(query: str) -> str:
    """
    Normalize common spacing/spelling variants so the embedding
    model sees a clean, consistent string.

    Examples
    --------
    'srilanka'    → 'sri lanka'
    'newzealand'  → 'new zealand'
    'southafrica' → 'south africa'
    'uae'         → 'united arab emirates'
    """
    replacements = {
        r'\bsrilanka\b':    'sri lanka',
        r'\bnewzealand\b':  'new zealand',
        r'\bsouthkorea\b':  'south korea',
        r'\bnorthkorea\b':  'north korea',
        r'\bsouthafrica\b': 'south africa',
        r'\bcostariica\b':  'costa rica',
        r'\bcostarica\b':   'costa rica',
        r'\bpuertorico\b':  'puerto rico',
        r'\buae\b':         'united arab emirates',
        r'\busa\b':         'united states',
        r'\buk\b':          'united kingdom',
    }
    normalized = query.lower().strip()
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    if normalized != query.lower().strip():
        print(f"[Retriever] Query normalized: '{query}' → '{normalized}'")
    return normalized


def retrieve_docs(query: str, top_k: int = 10) -> list[str]:
    client = get_qdrant_client()

    # ── Step 1: Normalize query spelling ─────────────────
    normalized_query = normalize_query(query)

    query_vector = get_embeddings([normalized_query])[0]

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    print("Retriever module loaded successfully.")

    hits = search_result.points

    if not hits:
        print("[Retriever] No results found in Qdrant.")
        return []

    # ── Step 2: Check best score against threshold ────────
    best_score = hits[0].score
    print(f"[Retriever] Best score: {best_score:.3f} | Threshold: {TOP_SCORE_THRESHOLD}")

    if best_score < TOP_SCORE_THRESHOLD:
        print(f"[Retriever] Best score {best_score:.3f} below threshold — no relevant context")
        return []

    # ── Step 3: Keep chunks that pass BOTH filters ────────
    # Filter 1 — relative : within 85% of best score
    # Filter 2 — absolute : must also be >= MIN_CHUNK_SCORE
    # Both must pass. This prevents chunks from a WRONG PDF
    # (e.g. india.pdf for a Sri Lanka query) from slipping
    # through just because they're close to a mediocre best.
    keep_threshold = max(best_score * 0.85, MIN_CHUNK_SCORE)

    relevant = []
    for hit in hits:
        score  = hit.score
        text   = hit.payload.get("text", "")
        source = hit.payload.get("source", "?")
        if score >= keep_threshold:
            print(f"[Retriever] Kept    {score:.3f} | {source}")
            relevant.append(text)
        else:
            print(f"[Retriever] Dropped {score:.3f} | {source}")

    print(f"[Retriever] Final: {len(relevant)} chunks kept")
    return relevant