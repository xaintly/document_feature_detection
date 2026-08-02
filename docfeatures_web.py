#!/usr/bin/env python3
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
    """Resolve file_path against CORPUS_BASE_PATH and validate the result
    stays under it, to prevent path traversal escaping the corpus.

    A file_path that looks rooted (contains a ':' -- a Windows drive
    letter or URI scheme -- or starts with '/' or '\\') is resolved as-is
    and must already live under CORPUS_BASE_PATH. Everything else is
    treated as relative to CORPUS_BASE_PATH and joined against it, so
    docfeatures_web doesn't need to be launched from the corpus directory.

    Returns the resolved absolute path, or None if disallowed.
    """
    if not CORPUS_BASE_PATH:
        return file_path  # no restriction configured

    base = Path(CORPUS_BASE_PATH).resolve()
    is_rooted = (
        ":" in file_path
        or file_path.startswith("/")
        or file_path.startswith("\\")
    )

    try:
        candidate = Path(file_path) if is_rooted else base / file_path
        resolved = candidate.resolve()
        resolved.relative_to(base)  # raises ValueError if outside base
    except (ValueError, OSError):
        return None

    return str(resolved)


def build_path_filter(term):
    """Build a slash-agnostic, substring 'file_path' filter.

    Matches anywhere in the path and treats '/' and '\\' as equivalent
    separators, so a search for "study_1005/lung" matches both
    "my_studies/study_1005/lung.txt" and "c:\\my_studies\\study_1005\\lung\\3.txt".

    Returns (sql_fragment, param) referencing the 'f' (files) alias, or
    None if term is blank.
    """
    term = (term or "").strip()
    if not term:
        return None
    normalized = term.replace("\\", "/")
    escaped = normalized.replace("%", r"\%").replace("_", r"\_")
    sql = r"REPLACE(f.file_path, '\\', '/') LIKE %s ESCAPE '\\'"
    return sql, f"%{escaped}%"


def build_search_query(runs, filters, file_path_search=None):
    """Build a JOIN-based SQL query from search filters.

    file_path_search, if given, is applied once at the outer level (it's
    a property of the file, not any particular run) via build_path_filter.

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
            f"(\n  SELECT d.doc_id, d.file_id, d.run_name\n"
            f"  FROM document_runs d\n  {join_sql}\n"
            f"  WHERE {where_sql}\n) AS {run_alias}"
        )
        if run_number > 0:
            run_select_sql += f" ON {run_alias}.file_id = q0.file_id"
        run_queries.append(run_select_sql)

    run_join_sql = "\n".join(run_queries)

    outer_where_sql = ""
    path_filter = build_path_filter(file_path_search)
    if path_filter:
        sql_fragment, param = path_filter
        outer_where_sql = f"\nWHERE {sql_fragment}"
        params.append(param)

    select_sql = (
        f"SELECT q0.doc_id, q0.file_id, q0.run_name, f.file_path, f.file_hash, f.file_size_bytes\n"
        f"{run_join_sql}\n"
        f"JOIN files f ON f.file_id = q0.file_id"
        f"{outer_where_sql}\n"
        f"ORDER BY f.file_path, q0.run_name"
    )
    count_sql = (
        f"SELECT COUNT(*) AS total\n"
        f"{run_join_sql}\n"
        f"JOIN files f ON f.file_id = q0.file_id"
        f"{outer_where_sql}\n"
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


def get_feature_configs_per_run(conn, run_names):
    """Return {run_name: {feature_name: {type, options, description}}},
    each run's config read independently (NOT merged like
    get_feature_configs_for_runs). Needed anywhere the exact declared
    option order / default value matters, since two runs can define the
    same feature name with different enum options."""
    if not run_names:
        return {}
    ph = ", ".join(["%s"] * len(run_names))
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT run_name, config_yaml FROM runs WHERE run_name IN ({ph})",
            run_names,
        )
        result = {}
        for row in cur.fetchall():
            cfg = yaml.safe_load(row["config_yaml"]) if row["config_yaml"] else {}
            feats = {}
            for fname, fdef in cfg.get("features", {}).items():
                feats[fname] = {
                    "type": fdef.get("type", "boolean"),
                    "options": fdef.get("options", []),
                    "description": fdef.get("description", ""),
                }
            result[row["run_name"]] = feats
        return result


def compute_chart_stats(conn, runs, features):
    """For each (run, feature) where feature is boolean/enum in that run's
    own config, compute a full category breakdown -- including the
    never-stored default value (False / first enum option), inferred by
    subtracting stored counts from the run's total completed-document
    count.

    Returns {run_name: {feature_name: {type, total, counts}}} where counts
    is [{label, count, is_default}, ...] in declared option order (for
    enums) or [True, False] (for booleans).
    """
    if not runs or not features:
        return {}

    per_run_configs = get_feature_configs_per_run(conn, runs)

    with conn.cursor() as cur:
        run_ph = ", ".join(["%s"] * len(runs))
        cur.execute(
            f"SELECT run_name, COUNT(*) AS cnt FROM document_runs "
            f"WHERE run_name IN ({run_ph}) AND status='complete' "
            f"GROUP BY run_name",
            runs,
        )
        totals = {row["run_name"]: row["cnt"] for row in cur.fetchall()}

        feat_ph = ", ".join(["%s"] * len(features))
        cur.execute(
            f"SELECT dr.run_name, df.feature_name, df.value_text, COUNT(*) AS cnt\n"
            f"FROM document_features df\n"
            f"JOIN document_runs dr ON df.doc_id = dr.doc_id\n"
            f"WHERE dr.run_name IN ({run_ph}) AND dr.status = 'complete'\n"
            f"AND df.feature_name IN ({feat_ph})\n"
            f"GROUP BY dr.run_name, df.feature_name, df.value_text",
            runs + features,
        )
        stored = {}  # (run_name, feature_name) -> {value_text: count}
        for row in cur.fetchall():
            key = (row["run_name"], row["feature_name"])
            stored.setdefault(key, {})[row["value_text"]] = row["cnt"]

    result = {}
    for run_name in runs:
        total = totals.get(run_name, 0)
        run_feats = per_run_configs.get(run_name, {})
        out_feats = {}
        for fname in features:
            fdef = run_feats.get(fname)
            if not fdef or fdef["type"] not in ("boolean", "enum"):
                continue
            stored_counts = stored.get((run_name, fname), {})

            if fdef["type"] == "boolean":
                # Only 'True' is ever stored; False is inferred.
                true_count = sum(stored_counts.values())
                false_count = max(total - true_count, 0)
                counts = [
                    {"label": "True", "count": true_count, "is_default": False},
                    {"label": "False", "count": false_count, "is_default": True},
                ]
            else:  # enum
                options = fdef.get("options") or []
                option_counts = {opt: 0 for opt in options}
                leftover = {}
                for val, c in stored_counts.items():
                    if val in option_counts:
                        option_counts[val] += c
                    else:
                        # Stored value doesn't match this run's declared
                        # options (config drift) -- surface it rather than
                        # silently dropping the count.
                        leftover[val] = leftover.get(val, 0) + c

                if options:
                    non_default_total = sum(option_counts[opt] for opt in options[1:])
                    option_counts[options[0]] = max(
                        total - non_default_total - sum(leftover.values()), 0
                    )

                counts = [
                    {"label": opt, "count": option_counts[opt], "is_default": i == 0}
                    for i, opt in enumerate(options)
                ]
                for val, c in leftover.items():
                    counts.append({"label": f"{val} (other)", "count": c, "is_default": False})

            out_feats[fname] = {
                "type": fdef["type"],
                "total": total,
                "counts": counts,
            }
        if out_feats:
            result[run_name] = out_feats
    return result


def normalize_browse_path(path):
    """Normalize a browse path to '/' separators, no leading/trailing slash."""
    return (path or "").strip().replace("\\", "/").strip("/")


def pick_representative_doc(rows):
    """Choose one document_runs row to represent a file for preview when it
    was touched by multiple runs. Prefer status='complete', tie-break on
    the lowest doc_id -- the same keeper rule docfeatures_dedupe.py uses."""
    complete = [r for r in rows if r["status"] == "complete"]
    pool = complete if complete else rows
    return min(pool, key=lambda r: r["doc_id"])


def list_folder(conn, path, run_name=None):
    """List immediate subfolders and files directly under `path`.

    If run_name is given, scoped to files that run touched (any status --
    browsing should surface errored docs too, not just complete ones).
    Otherwise the union of every file any run has ever touched.

    Returns {"path", "parent", "folders": [...], "files": [...]}.
    """
    prefix = normalize_browse_path(path)
    like_prefix = f"{prefix}/" if prefix else ""
    escaped_prefix = like_prefix.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")

    joins = []
    join_params = []
    if run_name:
        joins.append("JOIN document_runs dr ON dr.file_id = f.file_id AND dr.run_name = %s")
        join_params.append(run_name)

    sql = (
        "SELECT DISTINCT f.file_id, f.file_path, f.file_size_bytes\n"
        "FROM files f\n"
        + "\n".join(joins) + "\n"
        "WHERE REPLACE(f.file_path, '\\\\', '/') LIKE %s ESCAPE '\\\\'"
    )
    params = join_params + [f"{escaped_prefix}%"]

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

        folders = set()
        direct_files = []
        for row in rows:
            norm = row["file_path"].replace("\\", "/")
            remainder = norm[len(like_prefix):]
            if "/" in remainder:
                folders.add(remainder.split("/", 1)[0])
            else:
                direct_files.append(row)

        doc_map = {}
        file_ids = [f["file_id"] for f in direct_files]
        if file_ids:
            ph = ", ".join(["%s"] * len(file_ids))
            if run_name:
                cur.execute(
                    f"SELECT doc_id, file_id, status FROM document_runs "
                    f"WHERE run_name = %s AND file_id IN ({ph})",
                    [run_name] + file_ids,
                )
            else:
                cur.execute(
                    f"SELECT doc_id, file_id, status FROM document_runs "
                    f"WHERE file_id IN ({ph})",
                    file_ids,
                )
            by_file = {}
            for r in cur.fetchall():
                by_file.setdefault(r["file_id"], []).append(r)
            for fid, fid_rows in by_file.items():
                doc_map[fid] = pick_representative_doc(fid_rows)

    files_out = []
    for f in direct_files:
        rep = doc_map.get(f["file_id"])
        norm = f["file_path"].replace("\\", "/")
        files_out.append({
            "doc_id": rep["doc_id"] if rep else None,
            "status": rep["status"] if rep else None,
            "file_path": f["file_path"],
            "name": norm.rsplit("/", 1)[-1],
            "file_size": f["file_size_bytes"],
        })
    files_out.sort(key=lambda f: f["name"].lower())

    parent = prefix.rsplit("/", 1)[0] if "/" in prefix else ("" if prefix else None)

    return {
        "path": prefix,
        "parent": parent,
        "folders": sorted(folders, key=str.lower),
        "files": files_out,
    }


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
                       r.llm_model, r.llm_temperature, r.created_at,
                       COUNT(d.doc_id) AS total_docs,
                       SUM(CASE WHEN d.status='complete' THEN 1 ELSE 0 END)
                           AS completed
                FROM runs r
                LEFT JOIN document_runs d ON r.run_name = d.run_name
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
                    "temperature": row["llm_temperature"] if row["llm_temperature"] is not None else 0.0,
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


@app.route("/api/browse")
def api_browse():
    path = request.args.get("path", "")
    run_name = request.args.get("run") or None

    conn = get_db()
    try:
        if run_name:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM runs WHERE run_name = %s", (run_name,))
                if not cur.fetchone():
                    return jsonify({"error": f"Run '{run_name}' not found."}), 404
        return jsonify(list_folder(conn, path, run_name))
    finally:
        conn.close()


@app.route("/api/chart_stats", methods=["POST"])
def api_chart_stats():
    data = request.json or {}
    runs = data.get("runs", [])
    features = data.get("features", [])
    if not runs or not features:
        return jsonify({"error": "Select at least one run and one feature."}), 400

    conn = get_db()
    try:
        return jsonify(compute_chart_stats(conn, runs, features))
    finally:
        conn.close()


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.json or {}
    runs = data.get("runs", [])
    filters = data.get("filters", [])
    file_path_search = data.get("file_path_search", "")
    page = max(1, data.get("page", 1))
    page_size = min(200, max(1, data.get("page_size", 25)))

    if not runs:
        return jsonify({"error": "Select at least one run."}), 400

    select_sql, count_sql, params = build_search_query(runs, filters, file_path_search)

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
                "SELECT f.file_path FROM document_runs dr "
                "JOIN files f ON dr.file_id = f.file_id "
                "WHERE dr.doc_id = %s",
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

            # Get run info + file identity
            cur.execute(
                "SELECT dr.run_name, dr.file_id, f.file_path "
                "FROM document_runs dr JOIN files f ON dr.file_id = f.file_id "
                "WHERE dr.doc_id = %s",
                (doc_id,),
            )
            doc = cur.fetchone()

            # Also get features from other runs of the same file -- a plain
            # file_id filter now, no file_path self-join needed.
            if doc:
                cur.execute(
                    "SELECT dr.run_name, df.feature_name, df.value_text "
                    "FROM document_features df "
                    "JOIN document_runs dr ON df.doc_id = dr.doc_id "
                    "WHERE df.file_id = %s AND df.doc_id != %s "
                    "AND dr.status = 'complete' "
                    "ORDER BY dr.run_name, df.feature_name",
                    (doc["file_id"], doc_id),
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
    file_path_search = data.get("file_path_search", "")

    if not runs:
        abort(400)

    select_sql, _, params = build_search_query(runs, filters, file_path_search)
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
