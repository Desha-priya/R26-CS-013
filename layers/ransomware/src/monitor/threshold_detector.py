"""
threshold_detector.py
------------------------
Shared logic for the live, rule-based (non-ML) file-monitor detection
path. Used by BOTH demo_runner.py (scripted demo) and main.py (real
continuous operation) so there's exactly one implementation of "how do
we score and explain a live file-activity detection" - not two copies
that could quietly drift apart.
"""

FILE_RATE_THRESHOLD = 3.0      # files/sec over a 5s window
ENTROPY_THRESHOLD = 7.2        # matches config.yaml live_monitor.entropy_threshold

# Minimum number of distinct high-entropy events (see feature_extractor's
# HIGH_ENTROPY_EVENT_CUTOFF) within the window to treat as a real
# encryption signal even when the AVERAGE entropy has been diluted.
# Proven necessary via adversarial testing: an attacker interleaving
# real encrypted writes with low-entropy padding fully evaded the
# average-only check while repeatedly writing genuinely high-entropy
# files. Requiring 2+ (not 1) avoids flagging a single legitimate
# high-entropy file (e.g. copying one .zip or video file).
MIN_HIGH_ENTROPY_EVENTS = 2

# Second, independent evasion proven via testing: an attacker that
# stays well under FILE_RATE_THRESHOLD (e.g. one file every 1.5s) is
# NEVER caught by any rate-gated condition, at any window size - rate
# is normalized per-second, so a patient attacker's rate looks the
# same in a 5s, 30s, or 60s window. This is a fundamentally different
# evasion from entropy dilution and needs a fundamentally different
# fix: a RATE-INDEPENDENT check on cumulative high-entropy activity
# over the longest window. If enough distinct high-entropy writes
# accumulate over 60s - regardless of how slowly they arrived - that's
# still real, sustained encryption activity.
#
# Honest trade-off, stated plainly: this raises Time-to-Detect for a
# deliberately slow attacker compared to the <3s target for a loud
# one (it takes time for the count to accumulate), and sets a real
# false-positive risk if a legitimate user genuinely creates/copies
# many high-entropy files (e.g. batch-processing photos or archives)
# within 60 seconds. SUSTAINED_HIGH_ENTROPY_COUNT is deliberately set
# higher than MIN_HIGH_ENTROPY_EVENTS to keep that risk lower, but it
# is not eliminated - this is a tunable value, not a solved problem.
SUSTAINED_HIGH_ENTROPY_COUNT = 15

# File-NAME signals (many ransomware families rename what they encrypt):
#   - EXT_APPEND_MIN_COUNT files renamed by APPENDING an extension within 10 s
#     (a.docx -> a.docx.locked), using at most EXT_DISTINCT_MAX different new
#     extensions, is a mass-rename signature on its own - it works even when
#     the file contents cannot be read (locked or still being written).
#   - a ransom-note-looking file plus RANSOM_NOTE_MIN_HIGH_ENTROPY high-entropy
#     writes in 60 s.
# Honest trade-off: a user who bulk-renames 10+ files by adding the same
# extension within 10 s (for example adding ".bak") can trigger this; the
# restore only brings back files that are missing or look encrypted.
EXT_APPEND_MIN_COUNT = 10
EXT_DISTINCT_MAX = 3
RANSOM_NOTE_MIN_HIGH_ENTROPY = 3


def is_threat(features: dict) -> bool:
    rate = features.get("file_rate_5s", 0)
    avg_entropy = features.get("avg_entropy_5s", 0)
    high_entropy_count_5s = features.get("high_entropy_count_5s", 0)
    high_entropy_count_60s = features.get("high_entropy_count_60s", 0)

    rate_ok = rate >= FILE_RATE_THRESHOLD
    # Two independent paths to the entropy condition: either the
    # average is high (the original, simple case), OR enough distinct
    # high-entropy events happened even if padding diluted the average
    # (the dilution-resistant case, added after adversarial testing).
    entropy_ok = (avg_entropy >= ENTROPY_THRESHOLD) or (high_entropy_count_5s >= MIN_HIGH_ENTROPY_EVENTS)
    rate_and_entropy_path = rate_ok and entropy_ok

    # Rate-independent path: catches a patient attacker who never
    # crosses the rate threshold at all, proven necessary via testing.
    sustained_activity_path = high_entropy_count_60s >= SUSTAINED_HIGH_ENTROPY_COUNT

    extension_path = (features.get("appended_ext_count_10s", 0) >= EXT_APPEND_MIN_COUNT
                      and 0 < features.get("new_ext_distinct_10s", 0) <= EXT_DISTINCT_MAX)
    note_path = (features.get("ransom_note_count_60s", 0) >= 1
                 and high_entropy_count_60s >= RANSOM_NOTE_MIN_HIGH_ENTROPY)

    return rate_and_entropy_path or sustained_activity_path or extension_path or note_path


def compute_dynamic_confidence(features: dict) -> float:
    """
    Computes a REAL confidence score from the actual detected features,
    instead of a hardcoded number. A sample that barely crosses the
    thresholds gets a lower score; a sample that blows way past them
    (fast, high-entropy, lots of renames) gets a higher score - exactly
    like a real detector should behave.

    Not a trained model (that's HybridDetector's job on the
    CIC-MalMem-2022 side) - this is an explicit, auditable formula so
    it's easy to defend in a viva: "why is this score X?" has a clear
    answer, not a black box.
    """
    rate = features.get("file_rate_5s", 0)
    entropy = features.get("avg_entropy_5s", 0)
    rename_ratio = features.get("rename_ratio_5s", 0)

    rate_score = min(rate / (FILE_RATE_THRESHOLD * 3), 1.0)      # caps at 3x threshold
    entropy_score = min(entropy / 8.0, 1.0)                       # 8 bits/byte = max possible entropy
    rename_score = min(rename_ratio / 0.5, 1.0)                   # caps at 50% renames

    combined = 0.45 * entropy_score + 0.35 * rate_score + 0.20 * rename_score
    confidence = 0.75 + combined * 0.24
    return round(min(confidence, 0.99), 4)


def build_threshold_explanation(features: dict) -> dict:
    """
    A simple, honest, RULE-BASED explanation for the live file-monitor
    detection path. This is NOT SHAP - the live monitor uses a
    threshold rule (file_rate + entropy), not the trained Random
    Forest, so there's no model to run SHAP against here. Real SHAP
    explanations (method="shap_tree_explainer") come from
    HybridDetector.predict() in src/detection/predict.py, used on the
    CIC-MalMem-2022-trained classifier. Keeping the two clearly labeled
    (see "method" field) matters for the dissertation - don't overstate
    what generated which explanation.
    """
    rate = features.get("file_rate_5s", 0)
    entropy = features.get("avg_entropy_5s", 0)
    rename_ratio = features.get("rename_ratio_5s", 0)

    reasons = []
    if rate >= FILE_RATE_THRESHOLD:
        reasons.append(f"file modification rate of {rate:.1f}/sec (threshold {FILE_RATE_THRESHOLD}/sec)")
    if entropy >= ENTROPY_THRESHOLD:
        reasons.append(f"average write entropy of {entropy:.2f} bits/byte (threshold {ENTROPY_THRESHOLD})")
    if rename_ratio > 0.2:
        reasons.append(f"{rename_ratio*100:.0f}% of recent events were renames")
    if features.get("appended_ext_count_10s", 0) >= EXT_APPEND_MIN_COUNT:
        reasons.append(f"{features['appended_ext_count_10s']} files renamed by adding a new extension "
                       f"within 10 s")
    if features.get("ransom_note_count_60s", 0) >= 1:
        reasons.append("a file that looks like a ransom note appeared")

    summary = ("Flagged due to " + ", and ".join(reasons) + ".") if reasons else \
        "Flagged by combined file-activity signature."

    return {
        "method": "rule_based_threshold",
        "top_features": [
            {"feature": "file_rate_5s", "value": round(rate, 3)},
            {"feature": "avg_entropy_5s", "value": round(entropy, 3)},
            {"feature": "rename_ratio_5s", "value": round(rename_ratio, 3)},
        ],
        "summary": summary,
    }
