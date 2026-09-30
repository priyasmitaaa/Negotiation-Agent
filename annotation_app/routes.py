from pathlib import Path

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    session,
)

from .db import close_db, get_db, init_db
from .import_items import import_all
from . import services

bp = Blueprint("annotation", __name__)


@bp.record_once
def setup(state):
    app = state.app
    app.teardown_appcontext(close_db)

    @app.cli.command("init-db")
    def init_db_command():
        with app.app_context():
            init_db()
            print("Initialized database.")

    @app.cli.command("import-items")
    def import_items_command():
        with app.app_context():
            count = import_all(app.config["DATASET_ROOT"])
            print(f"Imported {count} items.")


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/health")
def health():
    get_db().execute("SELECT 1").fetchone()
    return {"ok": True}


@bp.post("/api/start")
def start():
    data = request.get_json(force=True)
    try:
        annotator = services.ensure_annotator(
            data.get("rater_id"),
            guide_acknowledged=bool(data.get("guide_acknowledged")),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    session["annotator_id"] = annotator["id"]
    return jsonify({"annotator": public_annotator(annotator), "progress": services.progress_list(annotator["id"])})


@bp.get("/api/session")
def current_session():
    annotator = require_annotator()
    return jsonify({"annotator": public_annotator(annotator), "progress": services.progress_list(annotator["id"])})


@bp.get("/api/progress")
def progress():
    annotator = require_annotator()
    return jsonify({"progress": services.progress_list(annotator["id"])})


@bp.get("/api/items/<item_code>")
def item(item_code):
    annotator = require_annotator()
    item_row = require_item(item_code)
    if not services.can_open_item(annotator["id"], item_row):
        return jsonify({"error": "Complete or report the current item before moving forward."}), 409
    ann = services.get_or_create_annotation(annotator["id"], item_row["id"])
    include_post = bool(ann["revealed_at"])
    return jsonify(services.serialize_item(item_row, ann, include_post=include_post))


@bp.post("/api/items/<item_code>/d0")
def save_d0(item_code):
    annotator = require_annotator()
    item_row = require_item(item_code)
    data = request.get_json(force=True)
    try:
        ann = services.save_d0(
            annotator["id"],
            item_row,
            data.get("d0_strategy"),
            data.get("d0_confidence"),
        )
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 423
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"annotation": services.serialize_annotation(ann), "progress": services.progress_list(annotator["id"])})


@bp.post("/api/items/<item_code>/reveal")
def reveal(item_code):
    annotator = require_annotator()
    item_row = require_item(item_code)
    try:
        ann = services.reveal(annotator["id"], item_row)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 423
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(services.serialize_item(item_row, ann, include_post=True))


@bp.patch("/api/items/<item_code>/ratings")
def ratings(item_code):
    annotator = require_annotator()
    item_row = require_item(item_code)
    try:
        ann = services.autosave(annotator["id"], item_row, request.get_json(force=True))
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 423
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"annotation": services.serialize_annotation(ann), "progress": services.progress_list(annotator["id"])})


@bp.post("/api/items/<item_code>/technical-issue")
def technical_issue(item_code):
    annotator = require_annotator()
    item_row = require_item(item_code)
    data = request.get_json(force=True)
    try:
        ann = services.report_technical_issue(annotator["id"], item_row, data.get("description"))
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 423
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"annotation": services.serialize_annotation(ann), "progress": services.progress_list(annotator["id"])})


@bp.post("/api/submit")
def submit():
    annotator = require_annotator()
    try:
        services.submit_all(annotator["id"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True})


@bp.get("/admin/export.csv")
def export_csv():
    require_admin_secret()
    return Response(
        services.csv_export(audit=False),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotation_export.csv"},
    )


@bp.get("/admin/audit_export.csv")
def audit_export_csv():
    require_admin_secret()
    return Response(
        services.csv_export(audit=True),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotation_audit_export.csv"},
    )


@bp.get("/reference/<path:name>")
def reference_file(name):
    allowed = {
        "human_eval_annotator_guide.docx",
        "human_eval_rating_sheet_REF.xlsx",
        "original_index.html",
        "package_README.txt",
    }
    if name not in allowed:
        abort(404)
    return send_file(Path(current_app.config["DATASET_ROOT"]) / "reference_material" / name, as_attachment=False)


@bp.get("/assets/<item_code>/<path:asset_path>")
def item_asset(item_code, asset_path):
    annotator = require_annotator()
    item_row = require_item(item_code)
    if ".." in Path(asset_path).parts:
        abort(400)
    if asset_path == "reply.wav":
        ann = services.annotation_for(annotator["id"], item_row["id"])
        if not ann or not ann["revealed_at"]:
            abort(404)
    elif not asset_path.startswith("context/"):
        abort(404)
    base = Path(current_app.config["DATASET_ROOT"]) / item_row["source_directory"]
    target = (base / asset_path).resolve()
    if not str(target).startswith(str(base.resolve())) or not target.exists():
        abort(404)
    return send_file(target)


def require_annotator():
    annotator_id = session.get("annotator_id")
    if not annotator_id:
        abort(401)
    annotator = services.annotator_by_id(annotator_id)
    if annotator is None:
        abort(401)
    return annotator


def require_item(item_code):
    item_row = services.item_by_code(item_code)
    if item_row is None:
        abort(404)
    return item_row


def require_admin_secret():
    expected = current_app.config.get("ADMIN_EXPORT_SECRET")
    supplied = request.headers.get("X-Admin-Secret") or request.args.get("secret")
    if not expected or supplied != expected:
        abort(403)


def public_annotator(annotator):
    return {
        "rater_id": annotator["rater_code"],
        "guide_acknowledged": bool(annotator["guide_acknowledged_at"]),
        "submitted": bool(annotator["submitted_at"]),
        "submission_status": annotator["submission_status"],
    }
