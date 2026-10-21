"""
alert_schema.py
------------------
Defines the JSON structure the Ransomware Killer sends to the Brain
component / other NeuraShield layers (especially the Zero-Trust Auth
layer, per the assignment). Kept as a plain dict builder (no external
schema-validation dependency) so it's easy to demo and easy to extend.
"""

import uuid
import time


def build_alert(
    threat_type: str,
    confidence: float,
    affected_files: list,
    process_info: dict,
    detection_features: dict,
    action_taken: dict,
    source_layer: str = "ransomware_killer",
    target_layer: str = "zero_trust_auth",
    explanation: dict = None,
) -> dict:
    """
    Builds a structured alert matching the NeuraShield Brain's expected
    format. `action_taken` should summarize what the response layer did
    (process kill result + rollback result), so downstream layers/Brain
    don't need to re-query anything.

    `explanation` (optional): output of ShapExplainer.explain() -
    {"top_features": [...], "summary": "..."}. When provided, this
    becomes the alert's human-readable "why" - what actually populates
    the "Brain Explanation" panel in the dashboard, generated from the
    real model rather than a hardcoded string.
    """
    return {
        "alert_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_layer": source_layer,
        "target_layer": target_layer,
        "threat": {
            "type": threat_type,               # e.g. "ransomware_activity"
            "confidence": round(confidence, 4),
            "severity": _severity_from_confidence(confidence),
        },
        "affected_files": {
            "count": len(affected_files),
            "paths": affected_files[:20],       # cap payload size; full list logged separately
            "truncated": len(affected_files) > 20,
        },
        "process": {
            "pid": process_info.get("pid"),
            "name": process_info.get("name"),
            "parent_name": process_info.get("parent_name"),
            "cmdline": process_info.get("cmdline"),
        },
        "detection_features": detection_features,
        "explanation": explanation,   # None if not computed for this alert
        "action_taken": action_taken,
        "schema_version": "1.1",
    }


def _severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.95:
        return "CRITICAL"
    if confidence >= 0.90:
        return "HIGH"
    if confidence >= 0.70:
        return "MEDIUM"
    return "LOW"
