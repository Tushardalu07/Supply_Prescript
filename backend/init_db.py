from backend.database import get_db_connection


def create_tables():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shipments (
                    id SERIAL PRIMARY KEY,
                    shipment_id VARCHAR(50) UNIQUE NOT NULL,
                    shipping_days INTEGER NOT NULL CHECK (shipping_days > 0),
                    expected_days INTEGER NOT NULL CHECK (expected_days > 0),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    print("Shipments table created successfully")


if __name__ == "__main__":
    create_tables()