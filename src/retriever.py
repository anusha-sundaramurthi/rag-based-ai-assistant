import re
from src.embeddings import get_query_embedding   # ← uses input_type='query'
from src.config import COLLECTION_NAME
from src.vectorstores import get_qdrant_client

TOP_SCORE_THRESHOLD = 0.60
MIN_CHUNK_SCORE     = 0.60


def normalize_query(query: str) -> str:
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

    normalized_query = normalize_query(query)
    # Use query-specific embedding for better retrieval accuracy
    query_vector = get_query_embedding(normalized_query)

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    hits = search_result.points

    if not hits:
        print("[Retriever] No results found in Qdrant.")
        return []

    best_score = hits[0].score
    print(f"[Retriever] Best score: {best_score:.3f} | Threshold: {TOP_SCORE_THRESHOLD}")

    if best_score < TOP_SCORE_THRESHOLD:
        print(f"[Retriever] Best score below threshold — no relevant context")
        return []

    keep_threshold = max(best_score * 0.80, MIN_CHUNK_SCORE)

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
