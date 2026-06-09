from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.config import QDRANT_HOST, QDRANT_API_KEY, COLLECTION_NAME


def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY
    )


def init_qdrant():
    client = get_qdrant_client()

    existing_collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    return client


def clear_qdrant():
    """
    Deletes and recreates the collection — wipes ALL stored chunks.
    Call this from the /clear-db endpoint to do a fresh start.
    """
    client = get_qdrant_client()
    print(f"[Qdrant] Deleting collection '{COLLECTION_NAME}'...")
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"[Qdrant] Recreating collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print(f"[Qdrant] Collection cleared and ready.")
    return client