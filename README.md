# Negotiation Reply Annotation App

This repository contains a lightweight web application for human evaluation of shopkeeper replies in second-hand electronics negotiations. It replaces the original static `index.html` plus spreadsheet workflow with one integrated annotation interface while preserving the two-stage reveal protocol.

## Source Materials

The authoritative research materials are preserved in `reference_material/`:

- `human_eval_annotator_guide.docx`: rubric meanings, workflow, examples, warning flags, and FAQ.
- `human_eval_rating_sheet_REF.xlsx`: export column names, item order, allowed values, and D6 flip-item mapping.
- `original_index.html`: original item order and traceability index.
- `package_README.txt`: original package note.

The 100 original item folders remain at repository root as `item_001/` through `item_100/`.

## Stack

Flask, vanilla JavaScript, plain CSS, Python `sqlite3`, pytest, Gunicorn, and Docker. This keeps the app easy for a research team to inspect and maintain.

## Workflow

Each item has two stages. Before reveal, the browser receives only the product card, previous conversation transcripts/audio, and D0 controls. After D0 Strategy and D0 Confidence are saved, the backend locks D0, records `revealed_at`, and returns seller reply text/audio plus D1-D13, flags, and comment fields.

D0 lock is enforced by the backend. Seller reply text and `reply.wav` are not returned or served before reveal.

## Rubric and Export

Researcher export preserves the reference spreadsheet order: `Rater ID`, `Item`, `D6 applies? (flip item)`, D0 fields, D1-D13, four explicit warning flags, and `Comment (optional)`.

D1-D12 use 1-5 segmented controls. D13 and all flags require explicit Yes/No answers.

## D6 Flip Items

D6 is mandatory only for:

`item_003`, `item_005`, `item_010`, `item_011`, `item_013`, `item_027`, `item_034`, `item_043`, `item_047`, `item_054`, `item_059`, `item_060`, `item_062`, `item_068`, `item_069`, `item_076`, `item_090`, `item_097`.

For all other items, D6 is automatically stored and exported as `N/A`.

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:FLASK_APP = "app:create_app"
flask init-db
flask import-items
flask run
```

Open `http://127.0.0.1:5000`.

## Environment

- `DATABASE_PATH`: SQLite file path. Defaults to `instance/annotations.sqlite3`.
- `ADMIN_EXPORT_SECRET`: required secret for researcher exports.
- `SECRET_KEY`: Flask session signing key. Use a long random value in production.
- `FLASK_ENV`: use `production` outside local development.

Copy `.env.example` for deployment. Do not commit `.env`.

## Tests

```bash
pytest
```

The tests cover item import/order, D6 mapping, session separation, D0 validation/locking, reveal secrecy, rating validation, D6 behavior, explicit flags, technical issues, final submission, and CSV export order.

## Production

```bash
gunicorn -b 0.0.0.0:8000 "app:create_app()"
```

Health check: `/health`.

## Docker

```bash
docker build -t negotiation-annotation .
docker run --rm -p 8000:8000 ^
  -e DATABASE_PATH=/data/annotations.sqlite3 ^
  -e ADMIN_EXPORT_SECRET=replace-me ^
  -v negotiation_annotation_data:/data ^
  negotiation-annotation
```

Run `flask init-db` and `flask import-items` once against the persistent volume. SQLite must live on durable storage such as `/data`.

## Deployment

Use a Docker-capable host with a persistent volume. Set `DATABASE_PATH`, set `ADMIN_EXPORT_SECRET`, initialize/import the database, start Gunicorn, and verify `/health` plus the root URL. This repository includes deployment configuration but is not deployed unless a live URL is tested.

## Exports and Backup

```bash
curl -H "X-Admin-Secret: $ADMIN_EXPORT_SECRET" https://your-site/admin/export.csv -o annotation_export.csv
curl -H "X-Admin-Secret: $ADMIN_EXPORT_SECRET" https://your-site/admin/audit_export.csv -o annotation_audit_export.csv
sqlite3 /data/annotations.sqlite3 ".backup '/data/annotations-backup.sqlite3'"
```

## Interface Behavior

- Sidebar shows untouched, partial, complete, and technical-issue states using color plus symbols.
- Forward navigation is blocked unless the current item is complete or recorded as a technical issue.
- Earlier visited items can be reopened.
- Autosave writes each answer to SQLite.
- Refreshing or reopening with the same Rater ID restores progress.
- Final submission is blocked until all 100 items are complete or technical issues.
- After final submission, data is read-only.

## Technical Issues

Use **Report technical problem** for broken text, missing assets, or other non-audio problems. It requires a short description, records `technical_issue`, and keeps the item distinguishable in exports.

## Research Safeguards

- Seller reply data is excluded from pre-reveal API responses.
- `reply.wav` is blocked before reveal.
- D0 cannot be changed after reveal.
- Rater IDs remain separate.
- The app does not show correct D0 answers, aggregate scores, leaderboards, or speed rewards.
- Dataset text is escaped in the frontend.
- Asset serving validates paths and prevents traversal.
- SQLite queries use parameters.
- Completeness is recalculated server-side.

## Updating Items or Rubric Text

Keep the `item_XXX/item.json`, `item.html`, `context/*.wav`, and `reply.wav` convention, then run `flask import-items`. Update rubric text carefully from the authoritative guide and spreadsheet; do not silently change column names or rating semantics.

## Troubleshooting

- Seller reply audio returning 404 before reveal is expected.
- If export returns 403, set and provide `ADMIN_EXPORT_SECRET`.
- If progress disappears after deployment restart, SQLite is not on a persistent volume.
