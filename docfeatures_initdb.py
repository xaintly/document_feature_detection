#!/usr/bin/env python3
"""
docfeatures_initdb.py — One-time database setup for docfeatures.

Run this once before using docfeatures.py or docfeatures_web.py.
It creates the database (if needed), tables, and indexes.

Usage:
    python docfeatures_initdb.py              # create/verify schema
    python docfeatures_initdb.py --check      # verify only, no changes
    python docfeatures_initdb.py --reset      # drop and recreate (destructive!)

Reads DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME from .env file.
"""

import argparse
import os
import sys

import pymysql
from dotenv import load_dotenv

load_dotenv()

# ===========================================================================
# Schema definition (single source of truth)
# ===========================================================================

TABLES = [
    (
        "runs",
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_name        VARCHAR(255) PRIMARY KEY,
            config_hash     CHAR(64),
            config_yaml     MEDIUMTEXT,
            description     TEXT,
            llm_host        VARCHAR(512),
            llm_model       VARCHAR(255),
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                      ON UPDATE CURRENT_TIMESTAMP
        )
        """,
    ),
    (
        "documents",
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id          INT AUTO_INCREMENT PRIMARY KEY,
            run_name        VARCHAR(255) NOT NULL,
            file_path       VARCHAR(2048) NOT NULL,
            file_hash       CHAR(64),
            file_size_bytes INT,
            total_chunks    INT DEFAULT 1,
            status          ENUM('processing','complete','error')
                                DEFAULT 'processing',
            error_message   TEXT,
            processing_secs FLOAT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_run_path (run_name, file_path),
            FOREIGN KEY (run_name) REFERENCES runs(run_name)
                ON DELETE CASCADE
        )
        """,
    ),
    (
        "chunk_results",
        """
        CREATE TABLE IF NOT EXISTS chunk_results (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            doc_id          INT NOT NULL,
            chunk_index     INT NOT NULL,
            raw_json        MEDIUMTEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_chunk (doc_id, chunk_index),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                ON DELETE CASCADE
        )
        """,
    ),
    (
        "document_features",
        """
        CREATE TABLE IF NOT EXISTS document_features (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            doc_id          INT NOT NULL,
            feature_name    VARCHAR(255) NOT NULL,
            value_text      VARCHAR(1024),
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_feature (doc_id, feature_name),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                ON DELETE CASCADE
        )
        """,
    ),
]

INDEXES = [
    ("idx_df_feature_value", "document_features", "(feature_name, value_text(128))"),
    ("idx_doc_run_status", "documents", "(run_name, status)"),
]


# ===========================================================================
# Helpers
# ===========================================================================

def get_connection(database=None):
    """Connect to MySQL. If database is None, connects without selecting a DB."""
    return pymysql.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def index_exists(cursor, table_name, index_name):
    cursor.execute(
        "SELECT 1 FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() "
        "AND table_name = %s AND index_name = %s LIMIT 1",
        (table_name, index_name),
    )
    return cursor.fetchone() is not None


def table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = DATABASE() "
        "AND table_name = %s LIMIT 1",
        (table_name,),
    )
    return cursor.fetchone() is not None


def database_exists(cursor, db_name):
    cursor.execute(
        "SELECT 1 FROM information_schema.schemata "
        "WHERE schema_name = %s LIMIT 1",
        (db_name,),
    )
    return cursor.fetchone() is not None


def get_table_row_count(cursor, table_name):
    try:
        cursor.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}")
        return cursor.fetchone()["cnt"]
    except Exception:
        return 0


# ===========================================================================
# Actions
# ===========================================================================

def do_check(db_name):
    """Verify the schema without making changes."""
    print(f"Checking database: {db_name}")
    print()

    # Check database exists
    conn = get_connection(database=None)
    with conn.cursor() as cur:
        if not database_exists(cur, db_name):
            print(f"  ✗ Database '{db_name}' does not exist.")
            print(f"    Run without --check to create it.")
            conn.close()
            return False
    conn.close()

    print(f"  ✓ Database '{db_name}' exists.")

    conn = get_connection(database=db_name)
    ok = True

    with conn.cursor() as cur:
        # Check tables
        for tname, _ in TABLES:
            if table_exists(cur, tname):
                count = get_table_row_count(cur, tname)
                print(f"  ✓ Table '{tname}' exists ({count:,} rows)")
            else:
                print(f"  ✗ Table '{tname}' is missing.")
                ok = False

        # Check indexes
        for iname, tname, _ in INDEXES:
            if table_exists(cur, tname) and index_exists(cur, tname, iname):
                print(f"  ✓ Index '{iname}' on '{tname}' exists")
            elif table_exists(cur, tname):
                print(f"  ✗ Index '{iname}' on '{tname}' is missing.")
                ok = False

    conn.close()
    print()
    if ok:
        print("Schema is up to date.")
    else:
        print("Schema has issues. Run without --check to fix.")
    return ok


def do_init(db_name):
    """Create the database, tables, and indexes."""
    # Create database if needed
    conn = get_connection(database=None)
    with conn.cursor() as cur:
        if database_exists(cur, db_name):
            print(f"  Database '{db_name}' already exists.")
        else:
            cur.execute(
                f"CREATE DATABASE {db_name} "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            print(f"  Created database '{db_name}'.")
    conn.close()

    # Create tables and indexes
    conn = get_connection(database=db_name)
    with conn.cursor() as cur:
        for tname, ddl in TABLES:
            existed = table_exists(cur, tname)
            cur.execute(ddl)
            if existed:
                count = get_table_row_count(cur, tname)
                print(f"  Table '{tname}' — already exists ({count:,} rows)")
            else:
                print(f"  Table '{tname}' — created")

        for iname, tname, columns_sql in INDEXES:
            if index_exists(cur, tname, iname):
                print(f"  Index '{iname}' — already exists")
            else:
                print(f"  Index '{iname}' — creating on '{tname}'...")
                cur.execute(f"CREATE INDEX {iname} ON {tname} {columns_sql}")
                print(f"  Index '{iname}' — created")

    conn.close()
    print()
    print("Database is ready. You can now run docfeatures.py.")


def do_reset(db_name):
    """Drop and recreate everything. Destructive!"""
    confirm = input(
        f"WARNING: This will DROP the database '{db_name}' and all data.\n"
        f"Type the database name to confirm: "
    )
    if confirm.strip() != db_name:
        print("Cancelled.")
        return

    conn = get_connection(database=None)
    with conn.cursor() as cur:
        if database_exists(cur, db_name):
            cur.execute(f"DROP DATABASE {db_name}")
            print(f"  Dropped database '{db_name}'.")
        else:
            print(f"  Database '{db_name}' did not exist.")
    conn.close()

    do_init(db_name)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Initialize the docfeatures database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s                    # create database, tables, and indexes
  %(prog)s --check            # verify schema, report issues
  %(prog)s --reset            # drop and recreate (asks for confirmation)
  %(prog)s --db my_corpus     # override DB_NAME from .env
        """,
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify the schema without making changes.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop and recreate the database (destructive!).",
    )
    parser.add_argument(
        "--db", metavar="NAME",
        help="Database name (overrides DB_NAME from .env).",
    )
    args = parser.parse_args()

    db_name = args.db or os.environ.get("DB_NAME", "docfeatures")

    print(f"docfeatures database initialization")
    print(f"  Host: {os.environ.get('DB_HOST', 'localhost')}:{os.environ.get('DB_PORT', '3306')}")
    print(f"  User: {os.environ.get('DB_USER', 'root')}")
    print(f"  Database: {db_name}")
    print()

    try:
        if args.reset:
            do_reset(db_name)
        elif args.check:
            ok = do_check(db_name)
            sys.exit(0 if ok else 1)
        else:
            do_init(db_name)
    except pymysql.err.OperationalError as e:
        code, msg = e.args
        if code == 1045:
            print(f"\n  Connection failed: Access denied. Check DB_USER and DB_PASSWORD in .env.")
        elif code == 2003:
            print(f"\n  Connection failed: Cannot reach MySQL at "
                  f"{os.environ.get('DB_HOST', 'localhost')}:{os.environ.get('DB_PORT', '3306')}.")
        else:
            print(f"\n  MySQL error {code}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
