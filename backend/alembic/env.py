from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from app.explanation_engine.repository import metadata as explanation_metadata
from app.recommendation_engine.repository import metadata as recommendation_metadata
# Make backend/app importable when Alembic runs from backend/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.models_base import Base

# Import all ORM model modules so Base.metadata contains all tables.
import app.cache.models
import app.instruments.models
import app.instrument_master.models
import app.market_data.models
import app.market_data_history.models
import app.portfolio.models
import app.profiles.models

config = context.config

# Use DATABASE_URL from app settings instead of hardcoding in alembic.ini.
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = [
    Base.metadata,
    explanation_metadata,
    recommendation_metadata,
]

def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()