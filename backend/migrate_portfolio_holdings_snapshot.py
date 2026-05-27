from sqlalchemy import text

from app.db import engine


MIGRATION_SQL = """
ALTER TABLE portfolio_holdings
ADD COLUMN IF NOT EXISTS source_upload_id VARCHAR;

ALTER TABLE portfolio_holdings
ADD COLUMN IF NOT EXISTS snapshot_date DATE;

ALTER TABLE portfolio_holdings
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

CREATE INDEX IF NOT EXISTS ix_portfolio_holdings_source_upload_id
ON portfolio_holdings(source_upload_id);

CREATE INDEX IF NOT EXISTS ix_portfolio_holdings_snapshot_date
ON portfolio_holdings(snapshot_date);
"""


def main() -> None:
    with engine.begin() as connection:
        connection.execute(text(MIGRATION_SQL))

    print("Portfolio holdings snapshot migration completed successfully.")


if __name__ == "__main__":
    main()