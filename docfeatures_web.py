#!/home/saintly/projects/ollama/venv/bin/python3
"""
docfeatures_web.py — Web interface for searching docfeatures results.

Usage:
    python docfeatures_web.py
    # or with gunicorn:
    gunicorn docfeatures_web:app -b 0.0.0.0:5000
"""

import csv
import io
import os
from pathlib import Path
# for print to stderr
import sys

import pymysql
import yaml
from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

CORPUS_BASE_PATH = os.environ.get("CORPUS_BASE_PATH", "")
MAX_PREVIEW_BYTES = 2 * 1024 * 1024  # 2 MB cap on file preview


# ===========================================================================
# Database
# ===========================================================================

def get_db():
    return pymysql.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "docfeatures"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


# ===========================================================================
# Helpers
# ===========================================================================

def safe_file_path(file_path):
    """Validate that a file path is under CORPUS_BASE_PATH to prevent
    path traversal. Returns the resolved path or None."""
    if not CORPUS_BASE_PATH:
        return file_path  # no restriction configured
    try:
        resolved = Path(file_path).resolve()
        print(resolved)
        base = Path(CORPUS_BASE_PATH).resolve()
        if str(resolved).startswith(str(base)):
            return str(resolved)
    except (ValueError, OSError):
        pass
    return None


def build_search_query(runs, filters):
    """Build a JOIN-based SQL query from search filters.

    Returns (select_sql, count_sql, params) where params is shared
    between both queries.
    """
    run_ph = ", ".join(["%s"] * len(runs))
    params = []
    run_queries = []

    for run_number, run_name in enumerate(runs):
        run_alias = f"q{run_number}"
        joins = []
        where = [f"d.run_name = %s", "d.status = 'complete'"]
        where_params = [run_name]
        
        for i, f in enumerate([filter for filter in filters if filter["run"] == run_name]):
            alias = f"f{i}"
            fname = f["feature"]
            fmode = f["mode"]

            if fmode == "present":
                joins.append(
                    f"INNER JOIN document_features {alias} "
                    f"ON d.doc_id = {alias}.doc_id "
                    f"AND {alias}.feature_name = %s"
                )
                params.append(fname)

            elif fmode == "absent":
                joins.append(
                    f"LEFT JOIN document_features {alias} "
                    f"ON d.doc_id = {alias}.doc_id "
                    f"AND {alias}.feature_name = %s"
                )
                params.append(fname)
                where.append(f"{alias}.id IS NULL")

            elif fmode == "enum_any":
                values = f.get("values", [])
                if not values:
                    continue
                null_wanted = False
                if "NULL" in values:
                    null_wanted = True
                    values = [value for value in values if value != 'NULL']
                    joins.append(
                        f"LEFT JOIN document_features {alias} "
                        f"ON d.doc_id = {alias}.doc_id "
                        f"AND {alias}.feature_name = %s"
                    )
                    params.append(fname)
                    if len(values) > 0:
                        val_ph = ", ".join(["%s"] * len(values))
                        where.append(f"({alias}.id IS NULL OR {alias}.value_text IN ({val_ph}))")
                        where_params.extend(values)
                    else:
                        where.append(f"{alias}.id IS NULL")
                    
                else:                    
                    val_ph = ", ".join(["%s"] * len(values))
                    joins.append(
                        f"INNER JOIN document_features {alias} "
                        f"ON d.doc_id = {alias}.doc_id "
                        f"AND {alias}.feature_name = %s "
                        f"AND {alias}.value_text IN ({val_ph})"
                    )
                    params.append(fname)
                    params.extend(values)

            elif fmode == "text_search":
                term = f.get("search", "").strip()
                if not term:
                    continue
                joins.append(
                    f"INNER JOIN document_features {alias} "
                    f"ON d.doc_id = {alias}.doc_id "
                    f"AND {alias}.feature_name = %s "
                    f"AND {alias}.value_text LIKE %s"
                )
                params.append(fname)
                params.append(f"%{term}%")

            elif fmode == "int_range":
                joins.append(
                    f"INNER JOIN document_features {alias} "
                    f"ON d.doc_id = {alias}.doc_id "
                    f"AND {alias}.feature_name = %s"
                )
                params.append(fname)
                if f.get("min") is not None:
                    where.append(
                        f"CAST({alias}.value_text AS SIGNED) >= %s"
                    )
                    params.append(int(f["min"]))
                if f.get("max") is not None:
                    where.append(
                        f"CAST({alias}.value_text AS SIGNED) <= %s"
                    )
                    params.append(int(f["max"]))

        params.extend(where_params)
        join_sql = "\n  ".join(joins)
        where_sql = " AND ".join(where)
        run_select_sql = ( "FROM " if run_number == 0 else "INNER JOIN " ) + (
            f"(\n  SELECT d.doc_id, d.file_path, d.file_hash, d.run_name, d.file_size_bytes\n"
            f"  FROM documents d\n  {join_sql}\n"
            f"  WHERE {where_sql}\n) AS {run_alias}"
        )
        if run_number > 0:
            run_select_sql += f" ON {run_alias}.file_path = q0.file_path"
        run_queries.append(run_select_sql)
        
    run_join_sql = "\n".join(run_queries)

    select_sql = (
        f"SELECT q0.doc_id, q0.file_path, q0.file_hash, q0.run_name, q0.file_size_bytes\n"
        f"{run_join_sql}\n"
        f"ORDER BY q0.file_path, q0.run_name"
    )
    count_sql = (
        f"SELECT COUNT(*) AS total\n"
        f"{run_join_sql}\n"
    )
    # print(select_sql, params, file=sys.stderr)
    return select_sql, count_sql, params


def get_features_for_docs(conn, doc_ids):
    """Fetch all features for a list of doc_ids. Returns dict of
    doc_id → list of {feature_name, value_text}."""
    if not doc_ids:
        return {}
    ph = ", ".join(["%s"] * len(doc_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT doc_id, feature_name, value_text "
            f"FROM document_features WHERE doc_id IN ({ph}) "
            f"ORDER BY feature_name",
            doc_ids,
        )
        result = {}
        for row in cur.fetchall():
            result.setdefault(row["doc_id"], []).append(
                {"name": row["feature_name"], "value": row["value_text"]}
            )
        return result


def get_feature_configs_for_runs(conn, run_names):
    """Merge feature definitions across selected runs. Returns a dict
    of feature_name → {type, options, description}."""
    if not run_names:
        return {}
    ph = ", ".join(["%s"] * len(run_names))
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT config_yaml FROM runs WHERE run_name IN ({ph})",
            run_names,
        )
        merged = {}
        for row in cur.fetchall():
            cfg = yaml.safe_load(row["config_yaml"]) if row["config_yaml"] else {}
            for fname, fdef in cfg.get("features", {}).items():
                if fname not in merged:
                    merged[fname] = {
                        "type": fdef.get("type", "boolean"),
                        "options": fdef.get("options", []),
                        "description": fdef.get("description", ""),
                    }
        return merged


# ===========================================================================
# Routes
# ===========================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/runs")
def api_runs():
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.run_name, r.description, r.config_yaml,
                       r.llm_model, r.created_at,
                       COUNT(d.doc_id) AS total_docs,
                       SUM(CASE WHEN d.status='complete' THEN 1 ELSE 0 END)
                           AS completed
                FROM runs r
                LEFT JOIN documents d ON r.run_name = d.run_name
                GROUP BY r.run_name
                ORDER BY r.created_at DESC
            """)
            runs = []
            for row in cur.fetchall():
                cfg = (
                    yaml.safe_load(row["config_yaml"])
                    if row["config_yaml"]
                    else {}
                )
                features = {}
                for fname, fdef in cfg.get("features", {}).items():
                    features[fname] = {
                        "run": row["run_name"],
                        "type": fdef.get("type", "boolean"),
                        "description": fdef.get("description", ""),
                        "options": fdef.get("options", []),
                    }
                # Check for filter info
                filter_info = cfg.get("filter")
                runs.append({
                    "run_name": row["run_name"],
                    "description": row["description"] or "",
                    "model": row["llm_model"] or "",
                    "created_at": str(row["created_at"]),
                    "total_docs": row["total_docs"] or 0,
                    "completed": int(row["completed"] or 0),
                    "features": features,
                    "has_filter": bool(filter_info),
                    "filter_from": (
                        filter_info.get("from_run", "")
                        if filter_info
                        else ""
                    ),
                })
        return jsonify(runs)
    finally:
        conn.close()


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.json or {}
    runs = data.get("runs", [])
    filters = data.get("filters", [])
    page = max(1, data.get("page", 1))
    page_size = min(200, max(1, data.get("page_size", 25)))

    if not runs:
        return jsonify({"error": "Select at least one run."}), 400

    select_sql, count_sql, params = build_search_query(runs, filters)

    conn = get_db()
    try:
        with conn.cursor() as cur:
            # Total count
            cur.execute(count_sql, params)
            total = cur.fetchone()["total"]

            # Page of results
            offset = (page - 1) * page_size
            page_params = params + [page_size, offset]
            cur.execute(select_sql + "\nLIMIT %s OFFSET %s", page_params)
            rows = cur.fetchall()

        # Fetch features for matched docs
        doc_ids = [r["doc_id"] for r in rows]
        features_map = get_features_for_docs(conn, doc_ids)

        # Fetch feature configs for type info
        feat_configs = get_feature_configs_for_runs(conn, runs)

        results = []
        for row in rows:
            fname = Path(row["file_path"]).name
            parent = Path(row["file_path"]).parent.name
            display_path = f"{parent}/{fname}" if parent != "." else fname

            doc_features = features_map.get(row["doc_id"], [])

            results.append({
                "doc_id": row["doc_id"],
                "file_path": row["file_path"],
                "display_path": display_path,
                "file_hash": row["file_hash"],
                "run_name": row["run_name"],
                "file_size": row["file_size_bytes"],
                "features": doc_features,
            })

        # Build the display SQL for "Show SQL"
        display_sql = select_sql
        for p in params:
            display_sql = display_sql.replace("%s", repr(str(p)), 1)

        return jsonify({
            "results": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, -(-total // page_size)),
            "feature_configs": feat_configs,
            "sql": display_sql,
        })
    finally:
        conn.close()


@app.route("/api/document/<int:doc_id>/content")
def api_document_content(doc_id):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_path FROM documents WHERE doc_id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row:
                abort(404)

        file_path = row["file_path"]
        safe = safe_file_path(file_path)
        if safe is None:
            abort(403)

        p = Path(safe)
        if not p.is_file():
            return jsonify({"error": "File not found on disk.", "path": file_path}), 404

        # Try UTF-8 first; fall back to cp1252 (Windows-1252), which is
        # the most common non-UTF-8 encoding in clinical systems and can
        # decode any byte sequence without errors.
        raw = p.read_bytes()
        if len(raw) > MAX_PREVIEW_BYTES:
            raw = raw[:MAX_PREVIEW_BYTES]
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("cp1252")
        if len(raw) == MAX_PREVIEW_BYTES:
            content += "\n\n[... truncated ...]"

        suffix = p.suffix.lower()
        if suffix in (".html", ".htm"):
            content_type = "html"
        elif suffix == ".md":
            content_type = "markdown"
        else:
            content_type = "text"

        return jsonify({
            "content": content,
            "content_type": content_type,
            "filename": p.name,
            "size": p.stat().st_size,
        })
    finally:
        conn.close()


@app.route("/api/document/<int:doc_id>/features")
def api_document_features(doc_id):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            # Get all features for this doc
            cur.execute(
                "SELECT feature_name, value_text FROM document_features "
                "WHERE doc_id = %s ORDER BY feature_name",
                (doc_id,),
            )
            features = [
                {"name": r["feature_name"], "value": r["value_text"]}
                for r in cur.fetchall()
            ]

            # Get run info
            cur.execute(
                "SELECT run_name, file_path FROM documents "
                "WHERE doc_id = %s",
                (doc_id,),
            )
            doc = cur.fetchone()

            # Also get features from other runs of the same file
            if doc:
                cur.execute(
                    "SELECT d.run_name, df.feature_name, df.value_text "
                    "FROM documents d "
                    "JOIN document_features df ON d.doc_id = df.doc_id "
                    "WHERE d.file_path = %s AND d.doc_id != %s "
                    "AND d.status = 'complete' "
                    "ORDER BY d.run_name, df.feature_name",
                    (doc["file_path"], doc_id),
                )
                other_runs = {}
                for r in cur.fetchall():
                    other_runs.setdefault(r["run_name"], []).append(
                        {"name": r["feature_name"], "value": r["value_text"]}
                    )
            else:
                other_runs = {}

        return jsonify({
            "doc_id": doc_id,
            "run_name": doc["run_name"] if doc else "",
            "file_path": doc["file_path"] if doc else "",
            "features": features,
            "other_runs": other_runs,
        })
    finally:
        conn.close()


@app.route("/api/export", methods=["POST"])
def api_export():
    """Export current search results as CSV (all pages)."""
    data = request.json or {}
    runs = data.get("runs", [])
    filters = data.get("filters", [])

    if not runs:
        abort(400)

    select_sql, _, params = build_search_query(runs, filters)
    # Remove the ORDER BY / add a limit for safety
    export_sql = select_sql + "\nLIMIT 100000"

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(export_sql, params)
            rows = cur.fetchall()

        doc_ids = [r["doc_id"] for r in rows]
        features_map = get_features_for_docs(conn, doc_ids)

        # Collect all feature names
        all_feat_names = sorted(
            {f["name"] for feats in features_map.values() for f in feats}
        )

        output = io.StringIO()
        writer = csv.writer(output)
        header = ["file_path", "run_name", "file_hash"] + all_feat_names
        writer.writerow(header)

        for row in rows:
            feats = {
                f["name"]: f["value"]
                for f in features_map.get(row["doc_id"], [])
            }
            csv_row = [
                row["file_path"],
                row["run_name"],
                row["file_hash"],
            ] + [feats.get(fn, "") for fn in all_feat_names]
            writer.writerow(csv_row)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=docfeatures_export.csv"
            },
        )
    finally:
        conn.close()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
