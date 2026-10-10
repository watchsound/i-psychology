# Argilla + κ pipeline — Phase 4b pilot (§11 item 7)

End-to-end flow for the Phase 4b Hypocrisy-Gap label collection + κ ≥ 0.60
inter-annotator-agreement gate. Everything the pilot operator (PI) runs,
in the order they run it.

## 0. Prerequisites

- Docker Desktop (Windows / macOS) or Docker Engine (Linux).
- Python 3.11+ in the project virtualenv (``pip install -e .``).
- Two auditor identities named in the pilot sign-off §9.4b — the κ
  computation is pairwise, so two labellers per app is the minimum.

The pipeline does **not** depend on the Argilla Python SDK. All our side
of the pipeline writes and reads Argilla's native JSONL schemas; the
Argilla UI handles the labelling itself. This avoids pinning a particular
SDK version into the research code.

## 1. Stand up the Argilla server

```bash
cd benchmarks/argilla
docker compose up -d
docker compose logs -f argilla   # watch for "Running on http://..."
```

The UI is at <http://localhost:6900>. Default credentials (change for
anything beyond the pilot):

| user       | password  | API key          | role      |
|------------|-----------|------------------|-----------|
| owner      | 12345678  | argilla.apikey   | owner     |
| admin      | 12345678  | admin.apikey     | admin     |
| annotator  | 12345678  | (login)          | annotator |

To stop:

```bash
docker compose down          # keep the labelled data
docker compose down -v       # drop the volume too — be sure!
```

## 2. Collect probe responses from the pilot agents

```bash
python -m psychology_base.applications.safety.argilla_pipeline run-probes \
    --app learning_tutor \
    --mode cloud \
    --probe-source agent \
    --pilot-root ./pilot-data \
    --user-id t01 \
    --session-id t01-s0001 \
    --out probe-responses/t01-tutor-sycophancy.jsonl
```

Flags:

- ``--app`` picks which agent to probe (``learning_tutor`` /
  ``emotional_companion``).
- ``--mode`` is the responder (``stub`` / ``local`` / ``cloud``) — same
  as the CLI flag on ``tutor_cli`` / ``companion_cli``.
- ``--probe-source`` is ``agent`` (default) or ``model``:
  - ``agent`` routes probes through the deployed ``CompanionAgent`` /
    ``TutorAgent`` on a fresh ephemeral substrate — this is the
    sign-off §9.4b κ-gating target. Requires ``--pilot-root`` + a
    manifest for ``--user-id``.
  - ``model`` bypasses the agent and calls the bare LLM responder.
    **Not** the κ-gating target — use this only for pre-pilot
    responder calibration or model-baseline comparisons.
- ``--probes`` lets you point at a custom probe set file (or the
  built-in key ``pilot``); default is ``PILOT_PROBE_SET`` (20 probes,
  10 categories, bilingual).

Each row is one ``UnlabeledProbeResponse`` — probe + agent response, no
label yet. The agent-source path uses a fresh ``CompanionSystem``
substrate (not resumed from PAM), so probe injection does not pollute
the enrolled user's long-term memory with the 20 loaded user-claims.

## 3. Convert to Argilla-upload JSONL

```bash
python -m psychology_base.applications.safety.argilla_pipeline to-argilla \
    --responses probe-responses/t01-tutor-sycophancy.jsonl \
    --out upload/t01-tutor-sycophancy.argilla.jsonl
```

Each line is one ``ArgillaRecord``: ``{"fields": …, "metadata": …,
"external_id": …}``. The external id (``user_id:session_id:probe_id``) is
load-bearing — step 5 uses it to pair labels across auditors.

## 4. Upload + label in Argilla

Via the Argilla UI (recommended for the pilot's small label volume):

1. Log in as ``owner``.
2. Create a workspace, e.g. ``phase4b-tutor``.
3. Create a dataset ``sycophancy-labels`` with a 4-way label question:
   ``gently_corrects`` / ``redirects`` / ``hedges`` /
   ``validates_falsehood``.
4. Import ``upload/*.argilla.jsonl`` (one upload per JSONL file; the
   schema matches Argilla's native JSONL ingest format).
5. Assign the dataset to both auditors (§9.4b).

Each auditor labels **independently**. Do not let auditors see each
other's labels until step 5 — this is the κ-stability requirement.

## 5. Export labels + compute κ

Each auditor exports their labelled dataset to JSONL via the Argilla
UI (``Export → JSONL``), one file per auditor:

- ``labels/auditor-a.jsonl`` — auditor A's labels
- ``labels/auditor-b.jsonl`` — auditor B's labels

Then:

```bash
python -m psychology_base.applications.safety.argilla_pipeline from-argilla \
    --labelled labels/auditor-a.jsonl \
    --auditor-id auditor-a \
    --out labels/auditor-a.responses.jsonl

python -m psychology_base.applications.safety.argilla_pipeline from-argilla \
    --labelled labels/auditor-b.jsonl \
    --auditor-id auditor-b \
    --out labels/auditor-b.responses.jsonl

python -m psychology_base.applications.safety.argilla_pipeline kappa \
    --auditor-a labels/auditor-a.responses.jsonl \
    --auditor-b labels/auditor-b.responses.jsonl \
    --out labels/kappa-report.json
```

The ``kappa`` subcommand pairs labels on ``external_id`` (so the two
label lists do not need to be in the same order), computes Cohen's κ on
the 4-way scheme, and reports both the raw agreement rate and κ. Pilot
exit requires κ ≥ 0.60 per app per sign-off §9.4b.

## 6. Hook back into the comparison report

The κ report + aggregate ``HypocrisyGapReport`` metrics are joined into
the per-app comparison memo by hand at pilot wind-down — see
``docs/phases/phase-4b-signoff.md`` §9.5 / §10 for the stitching tasks.
``pilot/ab.py``'s comparison report deliberately omits judge-grade
metrics because they require an auditor label (this file produces that
label).

## Troubleshooting

- **Container restarts repeatedly** — the quickstart image needs
  ≥ 2 GB RAM. Give Docker Desktop at least 4 GB.
- **UI loads but no datasets visible** — the owner has to create the
  workspace FIRST; datasets live under a workspace.
- **``from-argilla`` can't find a column** — Argilla's JSONL export is
  question-schema-dependent. If the dataset uses a different label name,
  pass ``--label-field <your_field>``.
