r"""
alert_store.py  -  NeuraShield alert history (persistent) + readable descriptions
---------------------------------------------------------------------------------
Layers keep their alerts in memory or in their own folders, and the platform's
active-alert list expires. The operator needs to open the console and SEE what
happened while it was closed, and a notification needs to say WHAT was detected
and WHEN. This module provides:

  * AlertStore    - append-only JSON-lines history (data/console/alert_history.jsonl),
                    de-duplicated by id, with acknowledge;
  * describe_*()  - turn each layer's raw alert into one uniform record with a
                    plain-language title, detail line, severity, source and time;
  * AlertCollector- polls the layers (through a fetch callback), stores new alerts
                    and calls on_new() for ones that arrived after the console started.

Privacy: records carry only what the layers already expose (type, scores, intake
channel, counts, times). The content layer deliberately does not keep sender or
subject, and neither does this.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

_SEV_RANK = {"info": 0, "low": 0, "medium": 1, "high": 2, "critical": 3}

LAYER_NAMES = {
    "zta": "Zero-Trust", "network": "Network Guardian",
    "ransomware": "Ransomware Killer", "content": "Content Threat",
}

_INTAKE_LABELS = {
    "gmail_api": "Gmail inbox", "gmail": "Gmail inbox", "imap": "mailbox (IMAP)",
    "folder": "watched folder", "folder_watcher": "watched folder", "eml": "uploaded email file",
    "notification": "desktop notification", "notifications": "desktop notification",
    "call": "live call", "call_monitor": "live call", "manual": "manual analysis",
    "upload": "manual upload", "dashboard": "manual analysis",
}


# --------------------------------------------------------------------- time
def to_epoch(value) -> float:
    """Epoch seconds from an epoch number or an ISO-8601 string (naive = local)."""
    if value is None or value == "":
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            return d.timestamp()               # local time
        return d.astimezone(timezone.utc).timestamp()
    except ValueError:
        return time.time()


def clock(ts: float) -> str:
    """'14:32' for today, '19 Oct 14:32' for earlier days."""
    d = datetime.fromtimestamp(ts)
    if d.date() == datetime.now().date():
        return d.strftime("%H:%M")
    return d.strftime("%d %b %H:%M")


# ------------------------------------------------------------------- store
class AlertStore:
    def __init__(self, path, max_records=2000):
        self.path = str(path)
        self.max = int(max_records)
        self._lock = threading.RLock()
        self._recs = {}            # id -> record (insertion ordered)
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and r.get("id"):
                        self._recs[r["id"]] = r          # later lines win (ack updates)
        except OSError:
            pass

    def _append(self, rec):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _compact(self):
        keep = sorted(self._recs.values(), key=lambda r: r.get("ts", 0))[-self.max:]
        self._recs = {r["id"]: r for r in keep}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    def has(self, rid) -> bool:
        return rid in self._recs

    def add(self, rec) -> bool:
        """Store a new record; False if its id is already known."""
        with self._lock:
            if rec["id"] in self._recs:
                return False
            rec.setdefault("ack", False)
            self._recs[rec["id"]] = rec
            self._append(rec)
            if len(self._recs) > self.max * 1.25:
                self._compact()
            return True

    def clear(self):
        """Forget every record (memory + file)."""
        with self._lock:
            self._recs = {}
            try:
                os.remove(self.path)
            except OSError:
                pass

    def ack(self, rid, by="operator") -> bool:
        with self._lock:
            r = self._recs.get(rid)
            if not r:
                return False
            r["ack"] = True
            r["ack_at"] = time.time()
            self._append(r)
            return True

    def recent(self, n=20, layer=None, min_severity=None, unacked_only=False):
        with self._lock:
            rows = sorted(self._recs.values(), key=lambda r: r.get("ts", 0), reverse=True)
        out = []
        floor = _SEV_RANK.get((min_severity or "").lower(), 0) if min_severity else 0
        for r in rows:
            if layer and r.get("layer") != layer:
                continue
            if _SEV_RANK.get(r.get("severity", "low"), 0) < floor:
                continue
            if unacked_only and r.get("ack"):
                continue
            out.append(r)
            if len(out) >= n:
                break
        return out

    def counts(self):
        with self._lock:
            rows = list(self._recs.values())
        open_ = [r for r in rows if not r.get("ack")]
        return {"total": len(rows), "unacknowledged": len(open_),
                "unacknowledged_high": sum(1 for r in open_ if _SEV_RANK.get(r.get("severity"), 0) >= 2)}


# ------------------------------------------------------------- describers
def _pct(x):
    try:
        return f"{float(x) * 100:.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _record(rid, ts, layer, severity, title, detail, source="", score=None, action=""):
    return {"id": rid, "ts": float(ts), "layer": layer, "layer_name": LAYER_NAMES.get(layer, layer),
            "severity": severity if severity in _SEV_RANK else "medium",
            "title": title, "detail": detail, "source": source,
            "score": score, "action": action, "ack": False}


def describe_content(a):
    """Content-layer alert file -> record, or None if it is not a threat verdict."""
    if not isinstance(a, dict) or "final_decision" not in a:
        return None
    decision = a.get("final_decision", "")
    if decision == "LOW_RISK":
        return None
    mods = a.get("modality_scores") or {}
    kinds, scores = [], []
    for key, label, bad, kind in (("text", "text", "phishing", "Phishing email/message"),
                                  ("video", "video", "fake", "Deepfake video"),
                                  ("audio", "audio", "fake", "Cloned / synthetic voice")):
        m = mods.get(key) or {}
        pred = m.get("prediction")
        if pred in ("unknown", "error", "no_speech", None):
            continue
        scores.append(f"{label} {_pct(m.get('risk_score'))}")
        if pred == bad:
            kinds.append(kind)
    title = " + ".join(kinds) if kinds else "Suspicious content"
    raw_src = next((str(a[k]) for k in ("intake", "source", "origin", "channel", "input") if a.get(k)), "")
    src = _INTAKE_LABELS.get(raw_src.lower(), raw_src.replace("_", " ")) if raw_src else ""
    risk = a.get("final_risk_score")
    high = decision == "HIGH_RISK" or any(
        (mods.get(k) or {}).get("risk_score", 0) >= 0.8 and (mods.get(k) or {}).get("prediction") in ("phishing", "fake")
        for k in mods)
    parts = [", ".join(scores)] if scores else []
    parts.append(f"overall risk {_pct(risk)}")
    if src:
        parts.append(f"via {src}")
    advice = []
    if "Phishing email/message" in kinds:
        advice.append("do not click links or open attachments")
    if "Deepfake video" in kinds:
        advice.append("treat this video as untrusted")
    if "Cloned / synthetic voice" in kinds:
        advice.append("verify the caller by another channel")
    rec = ("; ".join(advice) or "review before acting").capitalize()
    return _record("content:" + str(a.get("alert_id") or a.get("timestamp")), to_epoch(a.get("timestamp")),
                   "content", "high" if high else "medium", title, " · ".join(parts), src, risk, rec)


def describe_ransomware(a):
    if not isinstance(a, dict) or not a.get("alert_id"):
        return None
    th = a.get("threat") or {}
    files = (a.get("affected_files") or {})
    proc = a.get("process") or {}
    act = a.get("action_taken") or {}
    bits = []
    if files.get("count"):
        bits.append(f"{files['count']} file(s) affected")
    if proc.get("name"):
        bits.append(f"process {proc['name']}" + (f" (pid {proc['pid']})" if proc.get("pid") else ""))
    bits.append(f"confidence {_pct(th.get('confidence'))}")
    did = []
    if act.get("process_killed"):
        did.append("process terminated")
    if act.get("files_restored"):
        did.append(f"{act['files_restored']} file(s) restored")
    return _record("ransom:" + str(a["alert_id"]), to_epoch(a.get("timestamp")), "ransomware",
                   str(th.get("severity") or "high").lower(),
                   "Ransomware activity" + (" contained" if did else " detected"),
                   " · ".join(bits), "file monitor", th.get("confidence"), ", ".join(did))


def describe_network(a):
    if not isinstance(a, dict) or str(a.get("status", "")).upper() not in ("ALERT", "WARNING"):
        return None
    sev = str(a.get("severity") or ("critical" if a.get("status") == "ALERT" else "medium")).lower()
    title = str(a.get("category") or a.get("threat_type") or "Network anomaly")
    msg = str(a.get("message") or "")
    rid = f"net:{a.get('timestamp')}:{a.get('threat_type')}:{msg[:60]}"
    return _record(rid, to_epoch(a.get("timestamp")), "network", sev, title, msg[:220], "network traffic",
                   a.get("confidence"))


def describe_platform(src, a):
    """Alerts the Zero-Trust platform raised itself (layers 1/2/4 are described from
    their own, richer, sources)."""
    ln = a.get("layer")
    if str(ln) in ("1", "2", "4"):
        return None
    sev = str(a.get("severity") or "medium").lower()
    if _SEV_RANK.get(sev, 1) < 1:
        return None
    msg = str(a.get("message") or src)
    return _record(f"plat:{src}:{a.get('timestamp')}", to_epoch(a.get("timestamp")), "zta", sev,
                   "Zero-Trust: " + msg[:70], msg[:220], "session monitor")


def notification_text(rec):
    """(title, body) for a desktop toast: what happened, how bad, when."""
    title = f"{rec['layer_name']} - {rec['severity'].upper()}: {rec['title']}"
    body = f"{clock(rec['ts'])}  {rec['detail']}"
    if rec.get("action"):
        body += f"  [{rec['action']}]"
    return title[:120], body[:230]


# --------------------------------------------------------------- collector
class AlertCollector:
    """Polls the layers and keeps the store current.

    fetch(url) -> parsed JSON or None.  on_new(record) is called for alerts that
    appeared after the console started (older ones are only stored, so launching
    the app never replays old alerts as notifications)."""

    SOURCES = (
        ("platform", "http://127.0.0.1:8000/api/platform/status"),
        ("content", "http://127.0.0.1:8001/api/alerts?limit=50"),
        ("ransomware", "http://127.0.0.1:5050/api/alerts?show_acknowledged=true"),
        ("network", "http://127.0.0.1:5000/api/alerts"),
    )

    def __init__(self, store, fetch, on_new=None, interval=5.0):
        self.store, self.fetch, self.on_new, self.interval = store, fetch, on_new, interval
        self.started_at = time.time()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.started_at = time.time()       # alerts older than this are history, not news
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="alert-collector")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _ingest(self, rec):
        if rec and self.store.add(rec):
            # an alert from before this console session is history, not news
            if self.on_new and rec["ts"] >= self.started_at - 2:
                try:
                    self.on_new(rec)
                except Exception:
                    pass

    def poll(self):
        for name, url in self.SOURCES:
            data = self.fetch(url)
            if data is None:
                continue
            if name == "platform":
                active = (data.get("active_alerts") or {}) if isinstance(data, dict) else {}
                for src, a in active.items():
                    self._ingest(describe_platform(src, a))
            elif name == "content":
                items = data.get("alerts", []) if isinstance(data, dict) else data
                for a in items or []:
                    self._ingest(describe_content(a))
            elif name == "ransomware":
                for a in (data if isinstance(data, list) else data.get("alerts", [])) or []:
                    self._ingest(describe_ransomware(a))
            elif name == "network":
                for a in (data if isinstance(data, list) else data.get("alerts", [])) or []:
                    self._ingest(describe_network(a))
