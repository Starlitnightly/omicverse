"""Database utilities and initialization."""

import logging
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import Pool

from ..config.config import settings

# Optional models import: make this module import-safe when DB models are absent
try:
    # When packaged standalone, models may live under datacollect.models
    from ..models.base import Base, init_db  # type: ignore
except Exception:  # pragma: no cover - soft fallback for environments without DB
    class _PlaceholderMeta:
        tables = {}

    class _BasePlaceholder:
        metadata = _PlaceholderMeta()

    Base = _BasePlaceholder()  # type: ignore

    def init_db():  # type: ignore
        return None


logger = logging.getLogger(__name__)


def create_database_engine(database_url: Optional[str] = None):
    """Create database engine with optimized settings.
    
    Args:
        database_url: Database URL (uses settings if not provided)
    
    Returns:
        SQLAlchemy engine
    """
    url = database_url or settings.storage.database_url
    
    # SQLite-specific optimizations
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},  # Allow multi-threading
            pool_pre_ping=True,  # Verify connections
        )
        
        # Enable foreign keys and optimizations for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            cursor.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            cursor.close()
    else:
        # PostgreSQL or other databases
        engine = create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
    
    return engine


def initialize_database():
    """Initialize database tables and indexes."""
    logger.info("Initializing database...")
    
    try:
        # Create all tables
        init_db()
        
        # Create additional indexes for better performance
        engine = create_database_engine()
        
        with engine.connect() as conn:
            # Add indexes for common queries
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_proteins_gene_name ON proteins(gene_name)",
                "CREATE INDEX IF NOT EXISTS idx_proteins_organism ON proteins(organism)",
                "CREATE INDEX IF NOT EXISTS idx_structures_uniprot ON structures(uniprot_accession)",
                "CREATE INDEX IF NOT EXISTS idx_variants_gene ON variants(gene_symbol)",
                "CREATE INDEX IF NOT EXISTS idx_go_terms_namespace ON go_terms(namespace)",
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def check_database_connection():
    """Check if database is accessible."""
    try:
        engine = create_database_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_table_stats(session: Session) -> dict:
    """Get statistics about database tables.
    
    Args:
        session: Database session
    
    Returns:
        Dictionary with table names and row counts
    """
    stats = {}
    
    # Get all table names
    for table in Base.metadata.tables.values():
        try:
            count = session.execute(
                text(f"SELECT COUNT(*) FROM {table.name}")
            ).scalar()
            stats[table.name] = count
        except Exception as e:
            logger.error(f"Could not get stats for {table.name}: {e}")
            stats[table.name] = -1
    
    return stats


def vacuum_database():
    """Optimize database (SQLite specific)."""
    if not settings.storage.database_url.startswith("sqlite"):
        logger.info("Database optimization is SQLite specific, skipping")
        return
    
    try:
        engine = create_database_engine()
        with engine.connect() as conn:
            conn.execute(text("VACUUM"))
            conn.commit()
        logger.info("Database optimized successfully")
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")


def backup_database(backup_path: Optional[str] = None):
    """Create database backup (SQLite specific).
    
    Args:
        backup_path: Path for backup file
    """
    if not settings.storage.database_url.startswith("sqlite"):
        logger.warning("Database backup is SQLite specific")
        return
    
    import shutil
    from datetime import datetime
    
    # Extract database file path
    db_path = settings.storage.database_url.replace("sqlite:///", "")
    
    # Generate backup path if not provided
    if not backup_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.backup_{timestamp}"
    
    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise
