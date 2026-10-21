"""
Read-only: score the newest N emails in your mailbox with the REAL text model and
show the score distribution, so you can pick NEURASHIELD_EMAIL_PHISH_THRESHOLD
from your own data instead of guessing. Nothing is saved, no alert is sent, no
read/unread flag is changed (folder opened read-only, BODY.PEEK).

    python tools/check_mailbox_scores.py          # newest 50
    python tools/check_mailbox_scores.py 100
"""
import os, sys, email
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run(conn, n, predict, decode, extract_body, folder):
    conn.select(folder, readonly=True)
    st, data = conn.uid("SEARCH", None, "ALL")
    uids = [int(x) for x in (data[0] or b"").split()][-n:]
    rows = []
    for uid in uids:
        st, md = conn.uid("FETCH", str(uid), "(BODY.PEEK[])")
        if st != "OK" or not md or not md[0] or len(md[0]) < 2:
            continue
        msg = email.message_from_bytes(md[0][1])
        text = f"{decode(msg.get('Subject'))}\n\n{extract_body(msg)}".strip()
        if text:
            rows.append((predict(text)["risk_score"], decode(msg.get("Subject"))[:50]))
    return rows


def report(rows):
    if not rows:
        print("No emails scored."); return
    sc = sorted(r[0] for r in rows)
    print(f"\nScored {len(sc)} emails (newest first below, top 10 highest):")
    for s, subj in sorted(rows, reverse=True)[:10]:
        print(f"  {s:.3f}  {subj!r}")
    print("\nHow many would be flagged at each cut-off:")
    for t in (0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99):
        k = sum(1 for s in sc if s >= t)
        print(f"  threshold {t:.2f}: {k:3d} flagged ({100 * k / len(sc):.0f}%)")
    print("\nIf these are mostly NORMAL mails, set a threshold above where the normal "
          "ones stop, e.g.\n  set NEURASHIELD_EMAIL_PHISH_THRESHOLD=0.90   (then restart)")


if __name__ == "__main__":
    from src.watchers import mail_watcher as mw
    from src.fusion.multimodal_fusion_v5 import predict_text
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    c = mw.connect()
    try:
        report(run(c, n, predict_text, mw._decode, mw._extract_body, mw.FOLDER))
    finally:
        c.logout()
