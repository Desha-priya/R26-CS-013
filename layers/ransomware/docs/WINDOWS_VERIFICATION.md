# Windows verification checklist (things the Linux build environment cannot prove)

Run on the Windows VM / target machine, PowerShell, in the project folder.

1. `pip install -r requirements.txt` then `python -m main --check` → no `[FAIL]` lines.
2. `python -m pytest tests -q` → all pass.
3. Real kill: start `python -m main --dashboard` as **Administrator**, open http://127.0.0.1:5050, tick the box and press *Start protecting*, wait for the state **ARMED**; run a test attacker (the demo set's simulator or a real sample in the VM). Confirm the dashboard/log shows `Terminating suspicious process` and the files are restored.
4. Protected process: confirm the log never kills `explorer.exe`, `svchost.exe`, `lsass.exe` (`REFUSED to kill` lines if attribution points at them).
5. Antivirus: run as a normal user and as Administrator with Windows Defender ON; note any quarantine of `main.exe`.
6. Exe build: `pyinstaller layer2.spec` → `dist\NeuraShieldLayer2\NeuraShieldLayer2.exe --check` → all OK; run it for 2 minutes; confirm logs and `secret.key` appear under `%LOCALAPPDATA%\NeuraShield\Layer2`.
7. Reboot test: start, stop, start again; confirm it resumes as ARMED without asking again, unchanged files are not re-copied (log: baseline copied 0), and a previously used command id is still rejected.
8. Resource check: Task Manager over 10 minutes idle (target CPU < 5 %, memory < 50 MB for the monitor; the dashboard and ML add more).
9. Long path / locked file: encrypt a file that another program has open; confirm restore reports it and does not crash.


## Added checks (backup policy, reports, attribution)

1. **Attribution on Windows.** Run the fake attack as Administrator (`python fake_attack.py ...`) with the layer
   running as Administrator. In the incident report, "Attacker" should name `python.exe`/`powershell.exe` and the
   attack should stop when confidence is >= 0.95. Not Administrator: other users' processes cannot be inspected,
   so attribution may be "unidentified" (files are still restored).
2. **Office / PDF files.** Put a few `.docx`, `.xlsx`, `.pdf`, `.jpg` in a protected folder, wait for the backup,
   run an attack that overwrites them in place. They must be restored and the dashboard shows "encrypted copies
   blocked" increase.
3. **Backup timing.** Edit a text file; the dashboard line "Last backup" must update about 5 s after you stop editing,
   and saving the same file again within 30 s waits (shown as "waiting").
4. **Disk scan.** After an attack, press **Scan now**: leftovers (`.locked`, ransom note) are listed as
   "leftover"; no protected file should be listed as damaged once the restore finished.
5. **OneDrive.** If Documents/Desktop are redirected into OneDrive, `~/Documents` may not be the folder you think;
   check the folders listed in the dashboard before testing, and never test attacks inside a synced folder.
