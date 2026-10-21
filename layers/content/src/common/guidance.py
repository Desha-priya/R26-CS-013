"""
src/common/guidance.py
Plain-language "what to do / what not to do" guidance shown to the
user the moment a threat is detected, plus a short "see more"
explanation. This is what turns a raw risk score into something a
non-technical person can actually act on.

Keep this content honest and specific — generic "be careful online"
advice doesn't help anyone. Each category's dos/don'ts are the
concrete actions relevant to THAT specific threat.
"""

GUIDANCE = {
    "phishing": {
        "title": "Suspicious email detected",
        "summary": ("This email shows signs of phishing — content designed to "
                   "trick you into giving up credentials, money, or sensitive "
                   "information."),
        "dos": [
            "Verify the sender through a separate channel (call them, don't reply to the email)",
            "Hover over links before clicking to check where they actually go",
            "Report the email to your IT/security team if this is a work account",
        ],
        "donts": [
            "Don't click any links or download attachments in this email",
            "Don't enter passwords or personal details on a page it links to",
            "Don't reply with sensitive information, even if it looks urgent",
        ],
    },
    "deepfake_video": {
        "title": "Deepfake video detected on this call",
        "summary": ("The video on this call shows signs of AI manipulation — "
                   "the face may not belong to who it appears to be."),
        "dos": [
            "Verify the person's identity through a different channel (call their known number)",
            "Ask a question only the real person could answer, unscripted",
            "End the call if you're asked for money, credentials, or sensitive info",
        ],
        "donts": [
            "Don't share personal, financial, or company information based on this call alone",
            "Don't act on urgent requests (transfers, approvals) without independent verification",
            "Don't assume video is proof of identity — it no longer reliably is",
        ],
    },
    "cloned_voice": {
        "title": "Synthetic or cloned voice detected on this call",
        "summary": ("The voice on this call shows signs of being AI-generated or "
                   "cloned rather than a real live speaker."),
        "dos": [
            "Verify by calling the person back on a number you already know is theirs",
            "Ask something a script or clone wouldn't know",
            "Pause any requested action (payment, password reset, approval) until verified",
        ],
        "donts": [
            "Don't authorize payments or transfers based on a voice instruction alone",
            "Don't share one-time codes, passwords, or account details over this call",
            "Don't assume a familiar voice means a familiar person — voices can be cloned from short samples",
        ],
    },
    "coordinated": {
        "title": "Multiple signs of a coordinated attack detected",
        "summary": ("More than one independent check flagged this content as a "
                   "threat at the same time — this is a stronger signal than any "
                   "single check alone, and often indicates a deliberate, "
                   "multi-channel social-engineering attempt."),
        "dos": [
            "Treat this with extra caution — independent agreement across checks is significant",
            "Verify through a completely separate, trusted channel before acting on anything",
            "Report this to your security team as a likely coordinated attempt",
        ],
        "donts": [
            "Don't proceed with any request from this email/call until independently verified",
            "Don't dismiss this as a false alarm without checking — multiple signals agreeing is rare by chance",
        ],
    },
}

_DEFAULT_GUIDANCE = {
    "title": "Content flagged for review",
    "summary": "This content was flagged as potentially risky.",
    "dos": ["Review the detection details before acting on this content"],
    "donts": ["Don't treat this as automatically safe just because it's unclear"],
}


def _active_categories(alert: dict) -> list:
    """Which threat categories actually apply to this fused alert,
    based on which modalities independently called it a threat."""
    mods = alert.get("modality_scores", {}) or {}
    cats = []
    text = mods.get("text", {})
    video = mods.get("video", {})
    audio = mods.get("audio", {})
    if text.get("prediction") == "phishing":
        cats.append("phishing")
    if video.get("prediction") == "fake":
        cats.append("deepfake_video")
    if audio.get("prediction") == "fake":
        cats.append("cloned_voice")
    return cats


def get_guidance(alert: dict) -> dict:
    """
    Build the guidance payload for a fused alert. If more than one
    modality independently flagged a threat, this leads with the
    'coordinated' framing (the strongest, most specific signal) and
    still lists the individual categories underneath so the person
    sees exactly what was detected, not just a generic warning.
    """
    cats = _active_categories(alert)

    if not cats:
        return {"categories": [], **_DEFAULT_GUIDANCE}

    if len(cats) >= 2:
        primary = GUIDANCE["coordinated"]
        return {
            "categories": cats,
            "title": primary["title"],
            "summary": primary["summary"] + " Detected: " +
                      ", ".join(GUIDANCE[c]["title"].lower() for c in cats) + ".",
            "dos": primary["dos"] + [d for c in cats for d in GUIDANCE[c]["dos"]],
            "donts": primary["donts"] + [d for c in cats for d in GUIDANCE[c]["donts"]],
        }

    single = GUIDANCE.get(cats[0], _DEFAULT_GUIDANCE)
    return {"categories": cats, **single}
