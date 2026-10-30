import os
import psycopg


def get_db_connection():
    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname="supply_prescript",
        user="postgres",
        password=os.environ["DB_PASSWORD"],
        
    )