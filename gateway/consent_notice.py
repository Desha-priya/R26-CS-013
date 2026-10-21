"""
consent_notice.py  —  the single, authoritative NeuraShield consent & privacy
notice. Served to the setup page and the console Consent view so there is ONE
source of truth. Written to the structure security/privacy standards expect
(GDPR Arts. 5/7/9/13, ISO/IEC 27001 A.5.34, Sri Lanka PDPA No. 9 of 2022).
"""

NOTICE_VERSION = "1.0"

CONSENT_HTML = """
<h3>1. Who is responsible</h3>
<p>NeuraShield runs entirely on this computer. The administrator of this machine is the
data controller. There is no cloud service, no external processor and no third party:
all collection, analysis and storage happen locally on this device.</p>

<h3>2. What this software is</h3>
<p>NeuraShield is an on-premises security platform of four AI detection layers — Zero-Trust
Authentication, Network Guardian, Ransomware Killer and Content Threat Detection. It processes
personal data for one purpose only: to protect this computer and its user from cyber threats.</p>

<h3>3. Information the system collects</h3>
<ul>
  <li><strong>Behavioural biometrics (special category).</strong> Keystroke <em>timing</em> — the rhythm and
  dwell/flight intervals of typing, not the characters typed. Used by Zero-Trust to confirm the person
  at the keyboard is the enrolled user.</li>
  <li><strong>Facial image (special category).</strong> A webcam frame captured only at the moment a liveness
  check is required, used to confirm a live human is present. Raw frames are not retained after the check.</li>
  <li><strong>File-system activity.</strong> Names, paths, sizes and change rates of files in the folders you
  choose, plus short-lived content entropy, used to detect and roll back ransomware.</li>
  <li><strong>Network flow metadata.</strong> Source/destination addresses, ports, protocol, packet and byte
  counts and timing. Packet <em>contents</em> are never inspected.</li>
  <li><strong>Communication content you opt in to.</strong> Email text, and call audio/video frames, screened for
  phishing, deepfakes and spoofed audio. Each of these sources is a separate, explicit opt-in inside the layer.</li>
  <li><strong>Security events and audit records.</strong> Alerts, decisions, scores and administrator actions.</li>
</ul>

<h3>4. Legal basis</h3>
<p>Processing relies on your explicit consent (GDPR Art. 6(1)(a) and, for the biometric categories, Art. 9(2)(a);
the equivalent provisions of Sri Lanka's Personal Data Protection Act No. 9 of 2022). Consent is given freely here,
is recorded, and may be withdrawn at any time without affecting the lawfulness of processing before withdrawal.</p>

<h3>5. How information is used</h3>
<p>Collected signals are used solely to compute a security decision — a risk score, a threat verdict, or an
automated response such as a session step-up, a process termination or a file rollback. The system does not
profile you for any purpose other than security, does not advertise, and does not make its decisions available
to anyone outside this computer.</p>

<h3>6. Retention</h3>
<ul>
  <li><strong>Keystroke timing:</strong> held transiently in memory for the active scoring window and discarded;
  an enrolled behavioural profile (derived features, not raw keystrokes) is kept until you reset it.</li>
  <li><strong>Webcam frames:</strong> discarded immediately after the liveness check; not stored.</li>
  <li><strong>Ransomware backups:</strong> versioned copies of your own files, kept under a storage cap and
  rotated; removable by you at any time.</li>
  <li><strong>Network and content records / alerts:</strong> kept in local logs that rotate by size, then are overwritten.</li>
  <li><strong>Audit log:</strong> retained as a tamper-evident chain for accountability; you may export and clear it.</li>
</ul>

<h3>7. Information security</h3>
<p>Secrets (the platform key and signing secret) are protected by the operating system's data-protection facility
(Windows DPAPI). Services listen only on this computer's loopback interface and require an API key. The audit trail
is hash-chained so any tampering is detectable. No data is transmitted off the device.</p>

<h3>8. Your rights</h3>
<ul>
  <li><strong>Consent &amp; withdrawal.</strong> Nothing is monitored until you accept. You may withdraw consent at any
  time from the console; all monitoring stops immediately.</li>
  <li><strong>Choice of layers.</strong> Zero-Trust is on by default once you accept; you may turn it, or any layer, off.
  Mailbox and call monitoring are separate opt-ins.</li>
  <li><strong>Access &amp; portability.</strong> You can view and export the audit record of what the system did.</li>
  <li><strong>Erasure.</strong> You can reset the behavioural profile and delete local logs and backups.</li>
  <li><strong>Objection &amp; human oversight.</strong> Automated responses are explained (per-decision scores), and you
  remain in control through this console.</li>
</ul>

<h3>9. No automated disclosure</h3>
<p>NeuraShield contacts no external service and shares no personal data with any third party. Your information does
not leave this computer.</p>

<p style="margin-top:14px"><strong>By accepting, you confirm that you have read and understood this notice and consent to the
processing described above. You may withdraw at any time.</strong></p>
"""
