from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ..config.settings import settings

# Configure connection pool for concurrent document processing
# pool_size: number of persistent connections
# max_overflow: additional connections that can be created on demand
# pool_timeout: seconds to wait before giving up on getting a connection
engine = create_async_engine(
    settings.sql_database_url,
    pool_size=settings.sql_pool_size,
    max_overflow=settings.sql_max_overflow,
    pool_timeout=settings.sql_pool_timeout,
    pool_pre_ping=True,  # Verify connections before using them
)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)


@asynccontextmanager
async def get_db():
    if settings.use_sqlite and not settings.path_data_sql.exists():
        from .schema import init_db

        settings.path_data_sql.parent.mkdir(parents=True, exist_ok=True)
        await init_db(settings.sql_database_url)

    async with AsyncSessionLocal() as session:
        yield session
