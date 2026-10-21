import psutil, shutil, os, time
from alert_dispatcher import send_alert

SNAPSHOT_DIR = r"src\ransomware_killer\snapshots"

def take_snapshot(filepath: str):
    """Back up a file before it gets encrypted."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    fname = os.path.basename(filepath)
    dest  = os.path.join(SNAPSHOT_DIR, fname + ".bak")
    try:
        shutil.copy2(filepath, dest)
        print(f"[SNAPSHOT] Backed up {fname}")
    except Exception as e:
        print(f"[!] Snapshot failed for {filepath}: {e}")

def kill_process(pid: int, process_name: str):
    """Kill the malicious process and its children."""
    killed = []
    try:
        proc = psutil.Process(pid)
        children = proc.children(recursive=True)

        for child in children:
            child.kill()
            killed.append(child.pid)

        proc.kill()
        killed.append(pid)
        print(f"[KILLED] Process '{process_name}' (PID {pid}) and {len(children)} children.")
    except psutil.NoSuchProcess:
        print(f"[!] Process {pid} already gone.")
    except psutil.AccessDenied:
        print(f"[!] Access denied killing PID {pid}. Run as admin/root.")
    return killed

def rollback_files(files: list):
    """Restore files from snapshot directory."""
    restored = []
    for fpath in files:
        fname   = os.path.basename(fpath)
        backup  = os.path.join(SNAPSHOT_DIR, fname + ".bak")
        if os.path.exists(backup):
            shutil.copy2(backup, fpath)
            print(f"[ROLLBACK] Restored {fname}")
            restored.append(fpath)
        else:
            print(f"[!] No snapshot found for {fname}")
    return restored

def respond_to_ransomware(process_name: str, pid: int,
                           anomaly_score: float, affected_files: list):
    """Full automated response pipeline."""
    print(f"\n[RESPONSE] Ransomware detected: {process_name} (PID {pid})")
    print(f"[RESPONSE] Anomaly score: {anomaly_score:.4f}")

    # Step 1: Kill process
    kill_process(pid, process_name)

    # Step 2: Rollback files
    restored = rollback_files(affected_files)

    # Step 3: Send JSON alert to other layers
    action = f"Killed PID {pid}. Rolled back {len(restored)} files."
    send_alert(
        process_name   = process_name,
        pid            = pid,
        score          = anomaly_score,
        files_affected = affected_files,
        action_taken   = action
    )

    return {"killed_pid": pid, "restored_files": restored}