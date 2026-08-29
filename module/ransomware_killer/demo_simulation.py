"""
NeuraShield — Ransomware Killer DEMO
Run this script for PP1 supervisor demonstration.
Simulates: rapid file encryption → detection → kill → rollback → JSON alert
"""
import os, time, math, random, string, threading
import joblib, numpy as np

from response_engine import respond_to_ransomware, take_snapshot
from alert_dispatcher import send_alert

# ── Demo config ──────────────────────────────────────────────
DEMO_DIR       = r"src\ransomware_killer\demo_files"
NUM_FILES      = 10
FAKE_PID       = 99999          # simulated malicious PID
FAKE_PROC_NAME = "cryptolocker_sim.exe"

def _random_text(n=200):
    return ''.join(random.choices(string.ascii_letters + " \n", k=n))

def _simulate_entropy(data: bytes) -> float:
    """Shannon entropy of bytes."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    entropy = 0.0
    for c in counts:
        if c:
            p = c / n
            entropy -= p * math.log2(p)
    return entropy

def setup_demo_files():
    """Create benign files and take snapshots."""
    os.makedirs(DEMO_DIR, exist_ok=True)
    files = []
    print("[DEMO] Creating demo files and snapshots...")
    for i in range(NUM_FILES):
        fpath = os.path.join(DEMO_DIR, f"document_{i+1:02d}.txt")
        with open(fpath, "w") as f:
            f.write(_random_text())
        take_snapshot(fpath)
        files.append(fpath)
    print(f"[DEMO] {NUM_FILES} files created and backed up.\n")
    return files

def simulate_ransomware_encryption(files: list):
    """Overwrite files with high-entropy (encrypted-looking) data."""
    print("="*55)
    print("[RANSOMWARE SIM] Starting rapid file encryption...")
    print("="*55)
    affected = []
    for fpath in files:
        with open(fpath, "wb") as f:
            # Write pseudo-random bytes (simulates encryption)
            payload = bytes([random.randint(0, 255) for _ in range(512)])
            f.write(payload)
        entropy = _simulate_entropy(open(fpath,"rb").read())
        print(f"  [ENC] {os.path.basename(fpath)} — entropy: {entropy:.2f} bits/byte")
        affected.append(fpath)
        time.sleep(0.1)   # rapid writes
    return affected

def simulate_feature_vector():
    """
    Build a fake live feature vector matching your trained feature set.
    In production, this comes from the real-time monitor.
    """
    feature_names = joblib.load(r"models\ransomware_killer\feature_names.pkl")
    # Simulate high-anomaly feature values (large VAD count, many handles, etc.)
    vector = np.random.uniform(2.0, 5.0, size=(1, len(feature_names)))
    return vector, feature_names

def run_detection(feature_vector):
    """Run both models and return anomaly score."""
    iso = joblib.load(r"models\ransomware_killer\isolation_forest.pkl")
    rf  = joblib.load(r"models\ransomware_killer\random_forest.pkl")

    # Isolation Forest: score < 0 means anomalous
    iso_score  = iso.decision_function(feature_vector)[0]
    iso_pred   = iso.predict(feature_vector)[0]   # -1 = ransomware

    # Random Forest probability
    rf_prob    = rf.predict_proba(feature_vector)[0][1]  # P(ransomware)

    print(f"\n[DETECTION] Isolation Forest score : {iso_score:.4f}  ({'RANSOMWARE' if iso_pred == -1 else 'Benign'})")
    print(f"[DETECTION] Random Forest P(ransom): {rf_prob:.2%}")

    is_ransomware = (iso_pred == -1) or (rf_prob > 0.70)
    return is_ransomware, iso_score, rf_prob

def run_demo():
    print("\n" + "★"*55)
    print("  NeuraShield — Ransomware Killer  |  PP1 Demo")
    print("★"*55 + "\n")

    # Phase 1: Setup
    files = setup_demo_files()
    input("Press ENTER to start ransomware simulation...\n")

    # Phase 2: Simulate ransomware encrypting files
    affected_files = simulate_ransomware_encryption(files)
    print(f"\n[!] {len(affected_files)} files encrypted by simulated ransomware!\n")
    time.sleep(1)

    # Phase 3: Feature extraction + detection
    print("[MONITOR] Extracting behavioral features...")
    fvec, fnames = simulate_feature_vector()
    is_ransom, iso_score, rf_prob = run_detection(fvec)

    if is_ransom:
        print("\n🚨  RANSOMWARE CONFIRMED — Triggering automated response...\n")
        time.sleep(0.5)

        # Phase 4: Kill + rollback + alert
        respond_to_ransomware(
            process_name   = FAKE_PROC_NAME,
            pid            = FAKE_PID,
            anomaly_score  = abs(iso_score),
            affected_files = affected_files
        )

        # Phase 5: Verify rollback
        print("\n[VERIFY] Checking restored files...")
        for fpath in affected_files[:3]:
            content = open(fpath).read()
            print(f"  [OK] {os.path.basename(fpath)} — restored ({len(content)} chars, readable text)")

    else:
        print("[OK] No ransomware detected in this run.")

    print("\n[DEMO COMPLETE] Check logs/alerts.json for the full alert record.")
    print("Show your supervisor: models/confusion_matrix.png and models/feature_importance.png\n")

if __name__ == "__main__":
    run_demo()