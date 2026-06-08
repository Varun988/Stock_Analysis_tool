from app.instrument_master.service import seed_default_instrument_master


if __name__ == "__main__":
    inserted_count = seed_default_instrument_master()
    print(f"Seed completed. Inserted rows: {inserted_count}")