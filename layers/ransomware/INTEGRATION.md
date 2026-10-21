# Layer 2 (Ransomware Killer) — Integration Guide

Owner: Jayasinghe J.A.D.S (IT22293862) · Project R26-CS-013

## 1. Start / stop / health (Python)

```python
from src.layer2 import Layer2

layer = Layer2()                    # Layer2(enable_dashboard=True) also starts the local dashboard
layer.start()                       # STANDBY: nothing touches user files until consent exists
layer.grant_consent()               # or the user clicks "Start protecting" in the dashboard
layer.state                         # "standby" | "baseline" | "armed" | "incident"
layer.revoke_consent()              # stop watching/backing up (backups stay on disk)
layer.status()                      # running, incidents, backups, outbox size, resources
layer.health()                      # {"ok": True/False, "checks": {component: alive}}
layer.analyze_memory_features(d)    # optional: run the trained ML models on a feature dict
layer.stop()
```

### Consent model (important for the combined exe)

Layer 2 does **not** start backing up or monitoring by itself. Until consent exists it stays in `standby`
(health is still `ok`, `protection_active: false`). Consent is stored in
`<runtime>/data/consent.json`, tied to the configured folders, and survives restarts; changing the folders
requires consent again. Ways to give it: the dashboard button (shows folders + file count + limit first),
`layer.grant_consent()` (call it from the platform's own consent screen), `--consent`, or — only if the
platform installer already asked the user — `NEURASHIELD_AUTO_CONSENT=1` / `main.require_consent: false`
(logged as a warning). After consent: baseline backup (`state: baseline`, progress in the dashboard and in
`layer.status()["backup_progress"]`), then `armed` (changed files only).

The dashboard (`127.0.0.1:5050`) starts before everything else, so it is reachable during the baseline backup.
The service also writes `<runtime>/data/simulated_events/layer2_status.json` once per second (state, backup
progress, recent file events, incidents) for any other process that wants to read it.

Command line: `python -m main` (run), `python -m main --check` (readiness report; exit code 1 on FAIL),
`python -m main --dashboard` (also start the dashboard), `--consent` (agree to protect the configured folders), `--test-duration N` (run N seconds, then stop).

## 2. Configuration — environment variables (no secret is stored in files)

| Variable | Meaning | Default |
|---|---|---|
| `NEURASHIELD_HMAC_SECRET` | Shared signing secret for alerts and commands. **All layers must use the same value.** | random key generated on first start in the runtime folder |
| `NEURASHIELD_API_KEY` | API key of the main platform (sent as `X-API-Key`) | none (alerts are not sent) |
| `NEURASHIELD_PLATFORM_URL` | Origin of the platform, e.g. `http://localhost:8000` (no path) | empty (alerts stay local) |
| `NEURASHIELD_HOME` | Folder for logs, queues, secret key, state | project folder (source) / `%LOCALAPPDATA%\NeuraShield\Layer2` (exe) |
| `NEURASHIELD_CONFIG` | Full path of a `config.yaml` to use instead of the bundled one | bundled |
| `NEURASHIELD_BACKUP_ROOT` | Where versioned backups are kept | `~/.neurashield_cache_x7f2` |
| `NEURASHIELD_DASHBOARD_HOST` / `_PORT` | Dashboard address | `127.0.0.1` / `5050` |
| `NEURASHIELD_DASHBOARD_ALLOWED_HOSTS` | Extra Host names the dashboard accepts | none |
| `NEURASHIELD_AUTO_CONSENT` | `1` = grant consent automatically at start (only if the platform already obtained it) | off |
| `NEURASHIELD_REQUIRE_MODEL_MANIFEST` | `1` = refuse to load models if `models/MANIFEST.json` is missing | off |

Behaviour settings (watch folders, thresholds, backup limits, protected processes) are in `config/config.yaml`.
A copy of `config.yaml` placed in `<runtime folder>/config/` overrides the bundled one.

## 3. What Layer 2 sends to the platform

`POST {NEURASHIELD_PLATFORM_URL}/api/layer2/threat` with header `X-API-Key: <key>` and JSON body
`{"event_type": "ransomware_activity", "severity": "HIGH"}` (severity: `LOW|MEDIUM|HIGH|CRITICAL`;
only MEDIUM and above are sent). Redirects are not followed. If the platform is down, the notice is kept
in an outbox (`platform_outbox.jsonl`, max 500) and retried every 30 s. Nothing blocks the response
(kill / rollback never wait for the network).

## 4. Local alert format (signed, `data/simulated_events/alert_queue.jsonl`)

One JSON object per line: `alert_id`, `timestamp`, `target_layer`, `threat{type,severity,confidence}`,
`process{pid,name,...}`, `affected_files{count,paths}`, `detection_features`, `action_taken{process_killed,
kill_method, kill_skipped_reason, files_restored, restore_elapsed_seconds}`, optional `explanation`,
and `signature` = HMAC-SHA256 (hex) over the canonical JSON of all other fields (sorted keys, compact).
Verify with `src.alerting.hmac_auth.verify_payload`.

## 5. Commands INTO Layer 2

Signed rollback commands are appended to `data/simulated_events/command_queue.jsonl`
(`src.response.command_listener.build_rollback_command`). A command is accepted only if the signature is valid,
it is younger than 60 s, its id was never used (remembered across restarts) and its file list is valid.
**There is no HTTP command channel yet**; the platform would have to write to this queue (or we add an endpoint).

## 6. Health

* Python: `layer.health()`.
* Dashboard: `GET /api/health` → 200 healthy / 503 degraded.
* `python -m main --check` before first start on a new machine.

## 7. Honest scope (also for the panel)

* The **live monitor is rule-based** (file rate + entropy + process attribution). The trained ML models run on
  CIC-MalMem-2022 **memory features** and are exposed through `analyze_memory_features()`; they are not in the live loop.
* Process attribution is best-effort (recently active processes touching the folder); a kill needs confidence ≥ 0.95,
  protected/system processes are never killed.
* Backups are versioned copies: a baseline, then changed files within ~1-3 s of the monitor seeing the change (not a kernel filter driver; a file can be encrypted before its first backup if protection was not yet `armed`); a file already encrypted is never saved over a clean copy.
* Needs administrator rights to stop an attacker that runs as administrator.
* Not tested here: behaviour against every real-world ransomware family.


## Incident reports (v2)

Each attack is stored as one report in `data/incidents.json` (newest 100): detection time, trigger, attacker,
every damaged file with what happened and whether it was restored, leftovers (encrypted copies, ransom notes),
changed files that had no backup, and a timeline. The dashboard serves them at `GET /api/incidents` and
`GET /api/incidents/<id>`; `POST /api/incidents/<id>/restore` retries unrestored files and `POST /api/scan` /
`GET /api/scan` / `POST /api/scan/restore` run the disk scan (these need the dashboard to run inside the
service: `python -m main --dashboard`). The signed alerts sent to the platform are unchanged; one attack can
still produce several (first response, follow-up sweeps, integrity check).

New settings (`main:`): `backup_settle_seconds` (5), `backup_min_interval_seconds` (30).
