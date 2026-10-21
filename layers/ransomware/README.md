# NeuraShield — Layer 2: Ransomware Killer

Endpoint ransomware defence for the NeuraShield platform. It watches the protected folders, keeps versioned
backups, detects ransomware-like behaviour (fast file changes + high-entropy / random-looking data), stops the
responsible process when it is confident, restores **every** damaged file from before the attack began (and keeps sweeping for up to 60 s in case the attack continues), and
sends a signed alert (and a notice to the main platform).

Author: Jayasinghe J.A.D.S (IT22293862) · SLIIT B.Sc. Cyber Security · R26-CS-013

## Quick start (Windows, PowerShell, run as Administrator for full protection)

```powershell
pip install -r requirements.txt
python -m main --check        # readiness report: fix every [FAIL]
python -m main --dashboard    # start in STANDBY + local dashboard; click "Start protecting" there
python -m main --dashboard --consent   # same, but agree to protect the configured folders right away
python -m pytest tests -q     # automated tests
```

Local dashboard (this computer only): **http://127.0.0.1:5050** (type exactly this address).

### Consent first (nothing happens to your files before you agree)

Layer 2 starts in **STANDBY**: the dashboard and health check work, but no file is read, watched or copied.
The dashboard lists the folders and how many files/MB are in them; after you tick the box and press
**Start protecting**, Layer 2 makes a one-time *baseline* backup (progress bar), then keeps only
*changed* files up to date. Consent is saved, so later starts resume without asking; it must be given
again if the configured folders change. **Stop protecting** switches everything off (existing backups stay).
The dashboard also shows live file activity, so you can watch an attack happen.

## How it works

1. **Monitor** — `watchdog` events on the folders in `config/config.yaml`. Every event is judged on speed, content (entropy) **and names**: many files renamed by adding a new extension (`a.docx` → `a.docx.locked`) and ransom-note file names are detected even when file contents cannot be read. A watchdog restarts the monitor if it ever stops.
2. **Backup** — after consent: one baseline copy of every file, then only *changed* files. A changed file is copied after it has been quiet for `main.backup_settle_seconds` (5 s) and at most once per `main.backup_min_interval_seconds` (30 s) per file, plus a slow full sweep (5 min). Versions: max 10 per file, 5 GB total.
3. **Never back up an encrypted copy** — a new copy is refused (the clean backup is kept) if the file (a) turned from normal data into random-looking data, (b) lost the file signature its type requires (a `.docx`, `.pdf`, `.jpg`, `.zip` … is always high-entropy, so entropy alone cannot judge it; the signature can), or (c) is a new file that is the name of a protected file plus an extra extension, or has a well-known ransomware extension.
4. **Detect** — fast path: file rate + entropy, extension/ransom-note names. Slow path: after every full sweep an *integrity check* compares every protected file with its clean backup; if 3 or more turned into encrypted-looking data (a slow or stealthy attack that stays under the rate limit), it is treated as an attack too.
5. **Respond** — pause backups → find the attacking process (by the bytes it wrote, then by open files) → kill its tree if confidence ≥ 0.95 (protected/system processes are never killed) → restore every damaged file from the newest clean version made *before* the attack started → keep sweeping for late damage → signed alert.
6. **Report** — one **incident report per attack** (dashboard "Incident Reports"): what was damaged (encrypted / renamed / emptied), what was restored, what could not be, what the attack left behind, changed files that had no backup, the attacker, and a timeline. "Restore again" retries anything not restored. The **Disk scan** button checks the protected folders at any time for damaged files, ransomware-style names and ransom notes, and restores what has a clean backup. Signed alerts still go to the platform (see `INTEGRATION.md`).

The trained Isolation Forest + Random Forest models (CIC-MalMem-2022, memory features) are available through
`Layer2.analyze_memory_features()`; see `INTEGRATION.md` for what is and is not wired into the live loop.

## Files

| Path | Purpose |
|---|---|
| `main.py` | service entry point (`--check`, `--dashboard`, `--test-duration`) |
| `src/layer2.py` | the class the platform calls: start/stop/status/health |
| `src/monitor`, `src/response`, `src/alerting` | detection, rollback/kill, alerts |
| `src/dashboard` | hardened local dashboard |
| `src/detection` | model loading + integrity check |
| `models/` + `MANIFEST.json` | trained models and their SHA-256 fingerprints |
| `INTEGRATION.md` | environment variables, platform endpoint, alert format, health |
| `docs/WINDOWS_VERIFICATION.md` | checks to run on Windows before release |
| `layer2.spec` | PyInstaller build (`pyinstaller layer2.spec`) |

## Security notes

No secrets in files (see `INTEGRATION.md`). Alerts and commands are HMAC-SHA256 signed; commands expire after 60 s and
cannot be replayed. Models refuse to load if changed. The dashboard binds to 127.0.0.1, uses a strict CSP and
checks Host/Origin on every POST.
