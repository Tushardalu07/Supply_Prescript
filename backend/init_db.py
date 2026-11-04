from backend.database import get_db_connection


def create_tables():
    with get_db_connection() as conn:
        with conn.cursor() as cur:

            # 1. Shipments table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shipments (
                    id SERIAL PRIMARY KEY,
                    shipment_id VARCHAR(50) UNIQUE NOT NULL,
                    shipping_days INTEGER NOT NULL CHECK (shipping_days > 0),
                    expected_days INTEGER NOT NULL CHECK (expected_days > 0),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Decisions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id SERIAL PRIMARY KEY,
                    shipment_id VARCHAR(50) NOT NULL
                        REFERENCES shipments(shipment_id) ON DELETE CASCADE,
                    decision_id INTEGER NOT NULL,
                    decision_type VARCHAR(100) NOT NULL,
                    predicted_days INTEGER NOT NULL,
                    predicted_cost NUMERIC(12, 2) NOT NULL,
                    score NUMERIC(5, 2) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Outcomes table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id SERIAL PRIMARY KEY,
                    decision_record_id INTEGER UNIQUE NOT NULL
                        REFERENCES decisions(id) ON DELETE CASCADE,
                    actual_days INTEGER,
                    actual_cost NUMERIC(12, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    print("All three tables created successfully")


if __name__ == "__main__":
    create_tables()