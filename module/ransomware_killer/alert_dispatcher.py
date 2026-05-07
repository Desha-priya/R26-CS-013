import json, datetime, os

ALERT_LOG = r"src\ransomware_killer\logs/alerts.json"

def send_alert(process_name: str, pid: int, score: float,
               files_affected: list, action_taken: str):
    """
    Sends a structured JSON alert to other NeuraShield layers.
    In PP1 demo: prints to console + saves to log file.
    In production: POST to authentication layer API endpoint.
    """
    alert = {
        "timestamp":      datetime.datetime.utcnow().isoformat() + "Z",
        "source_layer":   "RansomwareKiller",
        "severity":       "CRITICAL",
        "event_type":     "RANSOMWARE_DETECTED",
        "process": {
            "name":       process_name,
            "pid":        pid,
            "anomaly_score": round(float(score), 4)
        },
        "files_affected": files_affected,
        "action_taken":   action_taken,
        "notify_layers":  ["NetworkGuardian", "ZeroTrustAuth", "ContentThreat"],
        "recommended_action": "Suspend all sessions for affected user. Escalate to SOC."
    }

    os.makedirs(r"src\ransomware_killer\logs", exist_ok=True)

    # Append to log file
    with open(ALERT_LOG, "a") as f:
        f.write(json.dumps(alert) + "\n")

    # Print alert (for demo / supervisor)
    print("\n" + "="*60)
    print("  NEURASHIELD ALERT DISPATCHED")
    print("="*60)
    print(json.dumps(alert, indent=2))
    print("="*60 + "\n")

    return alert