from sqlalchemy import text

from app.db import engine


MIGRATION_SQL = """
ALTER TABLE recommendations
ADD COLUMN IF NOT EXISTS research_context JSON;
"""


def main() -> None:
    with engine.begin() as connection:
        connection.execute(text(MIGRATION_SQL))

    print("Recommendations research_context migration completed successfully.")


if __name__ == "__main__":
    main()
