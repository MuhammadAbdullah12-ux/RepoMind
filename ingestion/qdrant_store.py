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
            
        if os.getenv("VERCEL") and path.startswith("/tmp/"):
            if not os.path.exists(path) or len(os.listdir(path) if os.path.exists(path) else []) == 0:
                for candidate in ["data/qdrant_db", "qdrant_storage"]:
                    if os.path.exists(candidate):
                        import shutil
                        print(f"[INFO] Copying pre-built Qdrant vectors '{candidate}' to '{path}'...")
                        try:
                            if os.path.exists(path):
                                shutil.rmtree(path)
                            shutil.copytree(candidate, path)
                        except Exception as e:
                            print(f"[WARNING] Failed to copy Qdrant seed: {e}")
                        break
        os.makedirs(path, exist_ok=True)
        lock_file = os.path.join(path, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass
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
