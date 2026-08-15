#!/usr/bin/env python3
"""
docfeatures_initdb.py — One-time database setup for docfeatures.

Run this once before using docfeatures.py or docfeatures_web.py.
It creates the database (if needed), tables, and indexes.

Schema shape:
    files           — stable file identity (file_path, file_hash, file_size_bytes).
                       One row per unique file_path, regardless of how many runs
                       have processed it.
    runs            — one row per named run (feature-definition version).
    document_runs   — one row per (file, run): status, chunk count, timing, errors.
                       This is what used to be called 'documents'.
    chunk_results   — raw per-chunk LLM output for a document_runs row.
    document_features — extracted feature values for a document_runs row, with
                       a denormalized file_id so "all features for this file
                       across every run" doesn't require joining through runs.

Usage:
    python docfeatures_initdb.py              # create/verify schema (fresh DB)
    python docfeatures_initdb.py --check      # verify only, no changes
    python docfeatures_initdb.py --migrate    # migrate an old 'documents'-table
                                               # schema to the files/document_runs split
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
# Schema definition (single source of truth for a FRESH install)
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
            llm_temperature FLOAT DEFAULT 0.0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                      ON UPDATE CURRENT_TIMESTAMP
        )
        """,
    ),
    (
        "files",
        """
        CREATE TABLE IF NOT EXISTS files (
            file_id         INT AUTO_INCREMENT PRIMARY KEY,
            file_path       VARCHAR(255) NOT NULL,
            file_hash       CHAR(64),
            file_size_bytes INT,
            first_seen_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                      ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_file_path (file_path)
        )
        """,
    ),
    # status is a plain VARCHAR, not a MySQL ENUM: it stores AWS Bedrock's own
    # job status strings verbatim (Submitted, Validating, Scheduled,
    # InProgress, Completed, PartiallyCompleted, Failed, Expired, Stopping,
    # Stopped -- see docfeatures_batch.py) plus two local-only bookend states
    # ('preparing' before submit, 'imported'/'cancelled' once we're done with
    # it). Bedrock's status set isn't ours to lock a DB constraint to.
    (
        "batch_jobs",
        """
        CREATE TABLE IF NOT EXISTS batch_jobs (
            batch_job_id    INT AUTO_INCREMENT PRIMARY KEY,
            run_name        VARCHAR(255) NOT NULL,
            job_name        VARCHAR(255) NOT NULL,
            job_arn         VARCHAR(512),
            model_id        VARCHAR(255),
            model_invocation_type ENUM('InvokeModel','Converse') DEFAULT 'Converse',
            s3_input_uri    VARCHAR(1024),
            s3_output_uri   VARCHAR(1024),
            role_arn        VARCHAR(512),
            status          VARCHAR(32) NOT NULL DEFAULT 'preparing',
            total_records   INT,
            error_message   TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            submitted_at    TIMESTAMP NULL,
            last_checked_at TIMESTAMP NULL,
            imported_at     TIMESTAMP NULL,
            UNIQUE KEY uq_job_name (job_name),
            FOREIGN KEY (run_name) REFERENCES runs(run_name)
                ON DELETE CASCADE
        )
        """,
    ),
    (
        "document_runs",
        """
        CREATE TABLE IF NOT EXISTS document_runs (
            doc_id          INT AUTO_INCREMENT PRIMARY KEY,
            run_name        VARCHAR(255) NOT NULL,
            file_id         INT NOT NULL,
            batch_job_id    INT NULL,
            total_chunks    INT DEFAULT 1,
            status          ENUM('processing','complete','error','batch_pending')
                                DEFAULT 'processing',
            error_message   TEXT,
            processing_secs FLOAT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_run_file (run_name, file_id),
            FOREIGN KEY (run_name) REFERENCES runs(run_name)
                ON DELETE CASCADE,
            FOREIGN KEY (file_id) REFERENCES files(file_id)
                ON DELETE CASCADE,
            FOREIGN KEY (batch_job_id) REFERENCES batch_jobs(batch_job_id)
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
            FOREIGN KEY (doc_id) REFERENCES document_runs(doc_id)
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
            file_id         INT NOT NULL,
            feature_name    VARCHAR(255) NOT NULL,
            value_text      VARCHAR(1024),
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_feature (doc_id, feature_name),
            FOREIGN KEY (doc_id) REFERENCES document_runs(doc_id)
                ON DELETE CASCADE,
            FOREIGN KEY (file_id) REFERENCES files(file_id)
                ON DELETE CASCADE
        )
        """,
    ),
    (
        "feature_verifications",
        """
        CREATE TABLE IF NOT EXISTS feature_verifications (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            doc_id          INT NOT NULL,
            feature_name    VARCHAR(255) NOT NULL,
            original_value  VARCHAR(1024),
            verified_value  VARCHAR(1024) NOT NULL,
            verified_by     VARCHAR(255),
            verified_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                      ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_doc_feature (doc_id, feature_name),
            FOREIGN KEY (doc_id) REFERENCES document_runs(doc_id)
                ON DELETE CASCADE
        )
        """,
    ),
]

INDEXES = [
    ("idx_df_feature_value", "document_features", "(feature_name, value_text(128))"),
    ("idx_doc_run_status", "document_runs", "(run_name, status)"),
    ("idx_file_hash", "files", "(file_hash)"),
    ("idx_batch_jobs_status", "batch_jobs", "(status)"),
]

# Additive columns for existing databases that predate a schema change.
# (table, column, ALTER TABLE ... ADD COLUMN ... statement)
COLUMNS = [
    ("runs", "llm_temperature",
     "ALTER TABLE runs ADD COLUMN llm_temperature FLOAT DEFAULT 0.0 AFTER llm_model"),
    ("document_runs", "batch_job_id",
     "ALTER TABLE document_runs ADD COLUMN batch_job_id INT NULL AFTER file_id, "
     "ADD CONSTRAINT fk_document_runs_batch_job FOREIGN KEY (batch_job_id) "
     "REFERENCES batch_jobs(batch_job_id) ON DELETE CASCADE"),
]

# Additive ENUM values for existing databases that predate a schema change.
# ALTER TABLE ... MODIFY doesn't fit the ADD-COLUMN-shaped COLUMNS list above,
# so this gets its own idempotent (table, column, value, ALTER statement) list.
ENUM_ADDITIONS = [
    ("document_runs", "status", "batch_pending",
     "ALTER TABLE document_runs MODIFY status "
     "ENUM('processing','complete','error','batch_pending') DEFAULT 'processing'"),
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


def column_exists(cursor, table_name, column_name):
    cursor.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = DATABASE() "
        "AND table_name = %s AND column_name = %s LIMIT 1",
        (table_name, column_name),
    )
    return cursor.fetchone() is not None


def enum_has_value(cursor, table_name, column_name, value):
    """Check whether *value* is already a member of a MySQL ENUM column,
    by inspecting its COLUMN_TYPE (e.g. "enum('a','b','c')")."""
    cursor.execute(
        "SELECT COLUMN_TYPE FROM information_schema.columns "
        "WHERE table_schema = DATABASE() "
        "AND table_name = %s AND column_name = %s LIMIT 1",
        (table_name, column_name),
    )
    row = cursor.fetchone()
    if row is None:
        return False
    return f"'{value}'" in row["COLUMN_TYPE"]


def fk_exists(cursor, table_name, column_name, ref_table_name):
    cursor.execute(
        "SELECT 1 FROM information_schema.KEY_COLUMN_USAGE "
        "WHERE table_schema = DATABASE() "
        "AND table_name = %s AND column_name = %s "
        "AND referenced_table_name = %s LIMIT 1",
        (table_name, column_name, ref_table_name),
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


def detect_schema_state(cursor):
    """Classify the DB as 'fresh', 'legacy', 'migrated', or 'partial'.

    'legacy'   — old single-table shape: documents.file_path exists, files
                 does not.
    'migrated' — files + document_runs exist, documents is gone, and
                 document_features already has file_id. Nothing to do.
    'partial'  — a previous --migrate run was interrupted partway through.
                 --migrate can be re-run safely; every step is idempotent.
    'fresh'    — none of the old or new tables exist yet.
    """
    has_files = table_exists(cursor, "files")
    has_document_runs = table_exists(cursor, "document_runs")
    has_documents = table_exists(cursor, "documents")
    has_legacy_file_path = has_documents and column_exists(cursor, "documents", "file_path")

    if not has_files and not has_document_runs and not has_documents:
        return "fresh"
    if (
        has_files and has_document_runs and not has_documents
        and column_exists(cursor, "document_features", "file_id")
    ):
        return "migrated"
    if has_legacy_file_path and not has_files:
        return "legacy"
    return "partial"


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
        state = detect_schema_state(cur)
        if state in ("legacy", "partial"):
            print(f"  ✗ Schema is in a '{state}' state (old-style 'documents' table "
                  f"with file_path found, or a migration was interrupted).")
            print(f"    Run with --migrate to convert to the files/document_runs schema.")
            conn.close()
            return False

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

        # Check additive columns
        for tname, cname, _ in COLUMNS:
            if table_exists(cur, tname) and column_exists(cur, tname, cname):
                print(f"  ✓ Column '{tname}.{cname}' exists")
            elif table_exists(cur, tname):
                print(f"  ✗ Column '{tname}.{cname}' is missing.")
                ok = False

        # Check additive ENUM values
        for tname, cname, value, _ in ENUM_ADDITIONS:
            if table_exists(cur, tname) and enum_has_value(cur, tname, cname, value):
                print(f"  ✓ Enum value '{tname}.{cname}' includes '{value}'")
            elif table_exists(cur, tname):
                print(f"  ✗ Enum value '{tname}.{cname}' is missing '{value}'.")
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

    conn = get_connection(database=db_name)
    with conn.cursor() as cur:
        state = detect_schema_state(cur)
        if state in ("legacy", "partial"):
            print()
            print(f"  This database has an old-style schema ('documents' table "
                  f"with file_path) that needs migrating.")
            print(f"  Run: python {os.path.basename(sys.argv[0])} --migrate")
            conn.close()
            return

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

        for tname, cname, alter_sql in COLUMNS:
            if column_exists(cur, tname, cname):
                print(f"  Column '{tname}.{cname}' — already exists")
            else:
                print(f"  Column '{tname}.{cname}' — adding...")
                cur.execute(alter_sql)
                print(f"  Column '{tname}.{cname}' — added")

        for tname, cname, value, alter_sql in ENUM_ADDITIONS:
            if enum_has_value(cur, tname, cname, value):
                print(f"  Enum value '{tname}.{cname}'='{value}' — already exists")
            else:
                print(f"  Enum value '{tname}.{cname}'='{value}' — adding...")
                cur.execute(alter_sql)
                print(f"  Enum value '{tname}.{cname}'='{value}' — added")

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


def do_migrate(db_name):
    """Migrate a legacy single-table 'documents' schema to the
    files / document_runs split, in place. Every step checks its own
    precondition first, so this is safe to re-run if interrupted.
    """
    conn = get_connection(database=db_name)

    with conn.cursor() as cur:
        state = detect_schema_state(cur)
        if state == "migrated":
            print("  Already migrated (files + document_runs present). Nothing to do.")
            conn.close()
            return
        if state == "fresh":
            print("  No existing schema found. Run without --migrate to create "
                  "a fresh install of the current schema.")
            conn.close()
            return

        print(f"  Detected schema state: {state}")

    confirm = input(
        f"\nThis will restructure live tables in database '{db_name}':\n"
        f"  - split 'documents' into 'files' + 'document_runs'\n"
        f"  - add a 'file_id' column (+ FK) to 'document_features'\n"
        f"  - 'chunk_results' and 'document_features' keep their existing rows;\n"
        f"    their FK to 'documents' is repointed at 'document_runs' automatically\n"
        f"Back up the database first if you have not already.\n"
        f"Type the database name to confirm: "
    )
    if confirm.strip() != db_name:
        print("Cancelled.")
        conn.close()
        return

    print()
    with conn.cursor() as cur:
        # --- Step 1: files table ---
        if not table_exists(cur, "files"):
            print("  [1/9] Creating 'files' table...")
            cur.execute("""
                CREATE TABLE files (
                    file_id         INT AUTO_INCREMENT PRIMARY KEY,
                    file_path       VARCHAR(255) NOT NULL,
                    file_hash       CHAR(64),
                    file_size_bytes INT,
                    first_seen_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                              ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_file_path (file_path)
                )
            """)
        else:
            print("  [1/9] 'files' table already exists — skipping create.")

        # --- Step 2: populate files (earliest row per file_path wins) ---
        cur.execute("SELECT COUNT(*) AS cnt FROM files")
        if cur.fetchone()["cnt"] == 0 and table_exists(cur, "documents"):
            print("  [2/9] Populating 'files' from 'documents' "
                  "(earliest row per file_path is canonical)...")
            cur.execute("""
                INSERT INTO files (file_path, file_hash, file_size_bytes, first_seen_at)
                SELECT d.file_path, d.file_hash, d.file_size_bytes, d.created_at
                FROM documents d
                INNER JOIN (
                    SELECT file_path, MIN(doc_id) AS min_doc_id
                    FROM documents
                    GROUP BY file_path
                ) fd ON d.file_path = fd.file_path AND d.doc_id = fd.min_doc_id
            """)
            print(f"        Inserted {cur.rowcount:,} file(s).")
        else:
            print("  [2/9] 'files' already populated — skipping.")

        # --- Step 3: anomaly report (informational only, non-blocking) ---
        if table_exists(cur, "documents"):
            cur.execute("""
                SELECT file_path, COUNT(DISTINCT file_hash) AS n
                FROM documents GROUP BY file_path HAVING COUNT(DISTINCT file_hash) > 1
            """)
            anomalies = cur.fetchall()
            if anomalies:
                print(f"  [3/9] ANOMALY: {len(anomalies)} file_path(s) have a "
                      f"different file_hash in different runs (the file changed "
                      f"on disk between runs). The EARLIEST version was kept as "
                      f"canonical in 'files'; later runs' extracted features are "
                      f"unaffected, but their file_hash no longer matches 'files'. "
                      f"Review:")
                for a in anomalies[:25]:
                    print(f"        {a['file_path']}  ({a['n']} distinct hashes)")
                if len(anomalies) > 25:
                    print(f"        ... and {len(anomalies) - 25} more.")
            else:
                print("  [3/9] No file_hash anomalies across runs.")

        # --- Step 4-7: migrate 'documents' -> 'document_runs' ---
        if table_exists(cur, "documents"):
            if not column_exists(cur, "documents", "file_id"):
                print("  [4/9] Adding 'file_id' column to 'documents'...")
                cur.execute("ALTER TABLE documents ADD COLUMN file_id INT NULL")
            else:
                print("  [4/9] 'documents.file_id' already exists — skipping.")

            cur.execute("SELECT COUNT(*) AS cnt FROM documents WHERE file_id IS NULL")
            if cur.fetchone()["cnt"] > 0:
                print("  [5/9] Backfilling 'documents.file_id' from 'files'...")
                cur.execute("""
                    UPDATE documents d
                    JOIN files f ON d.file_path = f.file_path
                    SET d.file_id = f.file_id
                    WHERE d.file_id IS NULL
                """)
                print(f"        Updated {cur.rowcount:,} row(s).")
            else:
                print("  [5/9] 'documents.file_id' already populated — skipping.")

            cur.execute("SELECT COUNT(*) AS cnt FROM documents WHERE file_id IS NULL")
            remaining = cur.fetchone()["cnt"]
            if remaining:
                raise RuntimeError(
                    f"{remaining} row(s) in 'documents' still have NULL file_id "
                    f"after backfill — aborting. This should not happen; investigate "
                    f"before re-running --migrate."
                )

            print("  [6/9] Enforcing NOT NULL + foreign key + unique key on 'documents'...")
            cur.execute("""
                SELECT IS_NULLABLE FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = 'documents'
                AND column_name = 'file_id'
            """)
            if cur.fetchone()["IS_NULLABLE"] == "YES":
                cur.execute("ALTER TABLE documents MODIFY file_id INT NOT NULL")

            if not fk_exists(cur, "documents", "file_id", "files"):
                cur.execute("""
                    ALTER TABLE documents
                    ADD CONSTRAINT fk_documents_file
                    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
                """)

            if index_exists(cur, "documents", "uq_run_path"):
                cur.execute("ALTER TABLE documents DROP INDEX uq_run_path")
            if not index_exists(cur, "documents", "uq_run_file"):
                cur.execute("""
                    ALTER TABLE documents
                    ADD UNIQUE KEY uq_run_file (run_name, file_id)
                """)

            drop_cols = [
                c for c in ("file_path", "file_hash", "file_size_bytes")
                if column_exists(cur, "documents", c)
            ]
            if drop_cols:
                print(f"  [7/9] Dropping redundant columns from 'documents': "
                      f"{', '.join(drop_cols)}...")
                cur.execute(
                    "ALTER TABLE documents " +
                    ", ".join(f"DROP COLUMN {c}" for c in drop_cols)
                )
            else:
                print("  [7/9] Redundant columns already dropped — skipping.")

            print("  [8/9] Renaming 'documents' -> 'document_runs'...")
            cur.execute("RENAME TABLE documents TO document_runs")
            print("        Done. chunk_results/document_features FKs now point "
                  "at document_runs automatically (verified: MySQL updates FK "
                  "metadata on RENAME TABLE).")
        else:
            print("  [4-8/9] 'documents' already migrated to 'document_runs' — skipping.")

        # --- Step 9: document_features gets a denormalized file_id ---
        if not column_exists(cur, "document_features", "file_id"):
            print("  [9/9] Adding 'file_id' to 'document_features'...")
            cur.execute("ALTER TABLE document_features ADD COLUMN file_id INT NULL")
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM document_features WHERE file_id IS NULL"
        )
        if cur.fetchone()["cnt"] > 0:
            print("        Backfilling 'document_features.file_id' from 'document_runs'...")
            cur.execute("""
                UPDATE document_features df
                JOIN document_runs dr ON df.doc_id = dr.doc_id
                SET df.file_id = dr.file_id
                WHERE df.file_id IS NULL
            """)
            print(f"        Updated {cur.rowcount:,} row(s).")

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM document_features WHERE file_id IS NULL"
        )
        remaining = cur.fetchone()["cnt"]
        if remaining:
            raise RuntimeError(
                f"{remaining} row(s) in 'document_features' still have NULL "
                f"file_id after backfill — aborting."
            )

        cur.execute("""
            SELECT IS_NULLABLE FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = 'document_features'
            AND column_name = 'file_id'
        """)
        if cur.fetchone()["IS_NULLABLE"] == "YES":
            cur.execute("ALTER TABLE document_features MODIFY file_id INT NOT NULL")

        if not fk_exists(cur, "document_features", "file_id", "files"):
            cur.execute("""
                ALTER TABLE document_features
                ADD CONSTRAINT fk_document_features_file
                FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
            """)
        print("        'document_features.file_id' is populated and constrained.")

    conn.close()
    print()
    print("Migration complete.")
    print("Next: update docfeatures.py (and any other scripts) to use the new")
    print("files/document_runs tables — see docfeatures_web.py, docfeatures_dedupe.py.")


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
  %(prog)s --migrate          # migrate an old 'documents'-table schema in place
  %(prog)s --reset            # drop and recreate (asks for confirmation)
  %(prog)s --db my_corpus     # override DB_NAME from .env
        """,
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify the schema without making changes.",
    )
    parser.add_argument(
        "--migrate", action="store_true",
        help="Migrate a legacy 'documents' schema to the files/document_runs split.",
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
        elif args.migrate:
            do_migrate(db_name)
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
