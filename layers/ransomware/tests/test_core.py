"""
test_core.py
--------------
Fast unit tests covering the core modules. Run with:
    python -m pytest tests/test_core.py -v
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.monitor.feature_extractor import shannon_entropy, SlidingWindowFeatureExtractor, FileEvent
from src.monitor.threshold_detector import is_threat, compute_dynamic_confidence
from src.alerting.hmac_auth import sign_payload, verify_payload
from src.response.rollback_engine import RollbackEngine


def test_entropy_distinguishes_plaintext_from_random():
    plain = b"the quick brown fox jumps over the lazy dog " * 20
    random_data = os.urandom(2000)
    assert shannon_entropy(plain) < 5.0
    assert shannon_entropy(random_data) > 7.5


def test_entropy_empty_input():
    assert shannon_entropy(b"") == 0.0


def test_threat_requires_both_rate_and_entropy():
    # High rate alone should NOT trigger
    assert not is_threat({"file_rate_5s": 10.0, "avg_entropy_5s": 3.0})
    # High entropy alone should NOT trigger
    assert not is_threat({"file_rate_5s": 0.5, "avg_entropy_5s": 7.9})
    # Both together SHOULD trigger
    assert is_threat({"file_rate_5s": 10.0, "avg_entropy_5s": 7.9})


def test_threat_resists_entropy_dilution_evasion():
    """Regression test for a real, proven evasion: an attacker
    interleaving genuine encrypted writes with low-entropy padding can
    drag avg_entropy_5s below threshold while still repeatedly writing
    high-entropy files. high_entropy_count_5s must catch this even
    when the average alone would miss it (confirmed via live testing
    with simulation/adversarial_simulator.py)."""
    diluted_but_real_attack = {
        "file_rate_5s": 3.0,
        "avg_entropy_5s": 1.99,          # diluted well below ENTROPY_THRESHOLD (7.2)
        "high_entropy_count_5s": 2,       # but 2+ distinct high-entropy writes occurred
    }
    assert is_threat(diluted_but_real_attack)


def test_threat_does_not_false_positive_on_single_high_entropy_file():
    """A single legitimate high-entropy file (e.g. copying one .zip or
    video) must NOT trigger on its own - only 2+ distinct high-entropy
    events count as a dilution-resistant signal."""
    single_legitimate_file = {
        "file_rate_5s": 3.0,
        "avg_entropy_5s": 1.5,
        "high_entropy_count_5s": 1,
    }
    assert not is_threat(single_legitimate_file)


def test_threat_catches_slow_drip_rate_throttling_evasion():
    """Regression test for a second real, proven evasion: an attacker
    that stays well under FILE_RATE_THRESHOLD at every window size is
    invisible to any rate-gated condition, no matter how much
    cumulative encryption has happened. Confirmed via live testing
    (20 files, 1 every 1.5s, rate never exceeded 2.4/s) that the
    original logic never fired. The rate-independent
    sustained_activity_path must catch this via accumulated
    high_entropy_count_60s."""
    slow_drip_attack = {
        "file_rate_5s": 2.4,          # never crosses FILE_RATE_THRESHOLD (3.0)
        "avg_entropy_5s": 2.0,
        "high_entropy_count_5s": 1,
        "high_entropy_count_60s": 15,  # but sustained over the longer window
    }
    assert is_threat(slow_drip_attack)


def test_threat_does_not_false_positive_on_spread_out_normal_use():
    """A handful of high-entropy files spread across 60s (e.g. a user
    downloading a few zip files) should not trigger, since it stays
    under SUSTAINED_HIGH_ENTROPY_COUNT."""
    spread_out_normal_use = {
        "file_rate_5s": 0.3,
        "avg_entropy_5s": 2.0,
        "high_entropy_count_5s": 1,
        "high_entropy_count_60s": 5,
    }
    assert not is_threat(spread_out_normal_use)


def test_confidence_increases_with_attack_severity():
    mild = compute_dynamic_confidence({"file_rate_5s": 3.0, "avg_entropy_5s": 7.2, "rename_ratio_5s": 0.0})
    aggressive = compute_dynamic_confidence({"file_rate_5s": 15.0, "avg_entropy_5s": 7.95, "rename_ratio_5s": 0.6})
    assert 0.0 <= mild <= 1.0
    assert 0.0 <= aggressive <= 1.0
    assert aggressive > mild, "A more severe attack should score higher confidence than a mild one"


def test_hmac_detects_tampering():
    secret = b"test-secret"
    payload = {"command": "rollback", "confidence": 0.9}
    sig = sign_payload(payload, secret)
    assert verify_payload(payload, sig, secret) is True

    tampered = dict(payload)
    tampered["confidence"] = 0.1
    assert verify_payload(tampered, sig, secret) is False


def test_hmac_rejects_wrong_secret():
    payload = {"a": 1}
    sig = sign_payload(payload, b"secret-a")
    assert verify_payload(payload, sig, b"secret-b") is False


def test_sliding_window_feature_extraction():
    import time
    extractor = SlidingWindowFeatureExtractor(window_seconds=(5, 30, 60))
    now = time.time()
    for i in range(10):
        extractor.add_event(FileEvent("modified", f"/tmp/f{i}.txt", timestamp=now - i * 0.2, entropy=7.5))
    features = extractor.compute_features(now=now)
    assert features["file_rate_5s"] > 0
    assert features["avg_entropy_5s"] > 7.0


def test_rollback_backup_and_restore_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "important.txt")
        with open(test_file, "w") as f:
            f.write("original clean content")

        engine = RollbackEngine(backup_root=os.path.join(tmp, "_backups"))
        backup_result = engine.backup_file(test_file)
        assert backup_result["status"] == "ok"

        # Simulate corruption
        with open(test_file, "wb") as f:
            f.write(os.urandom(200))

        restore_result = engine.restore_file(test_file)
        assert restore_result["status"] == "ok"

        with open(test_file, "r") as f:
            assert f.read() == "original clean content"


def test_rollback_detects_checksum_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "important.txt")
        with open(test_file, "w") as f:
            f.write("original content")

        engine = RollbackEngine(backup_root=os.path.join(tmp, "_backups"))
        backup_result = engine.backup_file(test_file)

        # Tamper with the backup file itself (simulating a corrupted/tampered backup)
        with open(backup_result["version_path"], "wb") as f:
            f.write(b"corrupted backup data")

        restore_result = engine.restore_file(test_file)
        assert restore_result["status"] == "failed"
        assert "checksum" in restore_result["reason"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
