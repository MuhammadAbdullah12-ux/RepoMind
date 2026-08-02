import os
from qdrant_client import QdrantClient

_global_client = None

def get_qdrant_client(path: str = None) -> QdrantClient:
    """
    Returns a shared, process-wide QdrantClient instance for the local storage.
    Ensures that only one client instance accesses the database folder at a time,
    preventing concurrent file access locks and permission exceptions.
    """
    global _global_client
    if _global_client is None:
        if path is None:
            path = "/tmp/qdrant_db" if os.getenv("VERCEL") else "data/qdrant_db"
        elif path == "data/qdrant_db" and os.getenv("VERCEL"):
            path = "/tmp/qdrant_db"
        os.makedirs(path, exist_ok=True)
        _global_client = QdrantClient(path=path)
    return _global_client

def close_global_client():
    """
    Closes the process-wide QdrantClient connection.
    """
    global _global_client
    if _global_client is not None:
        try:
            _global_client.close()
        except Exception:
            pass
        _global_client = None
