import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session

load_dotenv()

# ==========================================
# Day 4-5 Task 3: Database Connection & Initialization
# ==========================================

# Default to local SQLite storage for zero-dependency execution
# If DATABASE_URL is set in .env (e.g. postgresql://repomind:password123@localhost:5432/repomind_db), it uses Postgres.
IS_VERCEL = bool(os.getenv("VERCEL"))
DEFAULT_DB_URL = "sqlite:////tmp/repomind.db" if IS_VERCEL else "sqlite:///data/repomind.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# For SQLite, check_same_thread=False allows FastAPI multi-threaded requests
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)

def init_db():
    """
    Initializes the database and creates all defined SQLModel tables.
    """
    if DATABASE_URL.startswith("sqlite"):
        try:
            if "///" in DATABASE_URL:
                db_path = DATABASE_URL.split("///")[-1]
                db_dir = os.path.dirname(db_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)
        except Exception:
            pass
        
    print(f"[RUNNING] Initializing SQLModel Database at: {DATABASE_URL}...")
    # Import models to ensure they are registered with SQLModel metadata
    from backend import models
    SQLModel.metadata.create_all(engine)
    print("[SUCCESS] Database tables created successfully!")

def get_session():
    """
    FastAPI dependency yielding a database session per request.
    """
    with Session(engine) as session:
        yield session
