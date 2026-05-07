/* ══════════════════════════════════════════════════════════════
   NeuraShield — Ransomware Killer  |  Frontend Logic
   ══════════════════════════════════════════════════════════════ */

const API = "http://127.0.0.1:5050/api";
let pollTimer   = null;
let radarAnim   = null;
let entropyData = [];
let lastStatus  = "";

/* ── Entropy mini-chart state ────────────────────────────────── */
const entropyCtx = document.getElementById("entropyChart").getContext("2d");
let chartInstance = null;

/* ══════════════════════════════════════════════════════════════
   RADAR CANVAS
   ══════════════════════════════════════════════════════════════ */
const radarCanvas = document.getElementById("radarCanvas");
const rCtx        = radarCanvas.getContext("2d");
let radarAngle    = 0;
let radarMode     = "idle"; // idle | scanning | detected | clean

function drawRadar() {
  const W = radarCanvas.width, H = radarCanvas.height;
  const cx = W / 2, cy = H / 2, R = W / 2 - 10;

  rCtx.clearRect(0, 0, W, H);

  const colors = {
    idle:     { ring: "rgba(0,212,255,0.12)", sweep: "rgba(0,212,255,0.3)",  glow: "#00d4ff" },
    scanning: { ring: "rgba(0,212,255,0.20)", sweep: "rgba(0,212,255,0.55)", glow: "#00d4ff" },
    detected: { ring: "rgba(239,68,68,0.20)", sweep: "rgba(239,68,68,0.6)",  glow: "#ef4444" },
    clean:    { ring: "rgba(16,185,129,0.15)",sweep: "rgba(16,185,129,0.4)", glow: "#10b981" },
  };
  const c = colors[radarMode] || colors.idle;

  // Rings
  [0.25, 0.5, 0.75, 1].forEach(f => {
    rCtx.beginPath();
    rCtx.arc(cx, cy, R * f, 0, Math.PI * 2);
    rCtx.strokeStyle = c.ring;
    rCtx.lineWidth   = 1;
    rCtx.stroke();
  });

  // Cross-hairs
  rCtx.strokeStyle = c.ring;
  rCtx.lineWidth   = 0.5;
  rCtx.beginPath(); rCtx.moveTo(cx, cy - R); rCtx.lineTo(cx, cy + R); rCtx.stroke();
  rCtx.beginPath(); rCtx.moveTo(cx - R, cy); rCtx.lineTo(cx + R, cy); rCtx.stroke();

  // Sweep gradient
  const grad = rCtx.createConicalGradient
    ? rCtx.createConicalGradient(radarAngle, cx, cy)
    : null;

  if (!grad) {
    // Fallback arc sweep
    rCtx.save();
    rCtx.translate(cx, cy);
    rCtx.rotate(radarAngle);
    const sg = rCtx.createLinearGradient(0, 0, R, 0);
    sg.addColorStop(0, c.sweep);
    sg.addColorStop(1, "transparent");
    rCtx.beginPath();
    rCtx.moveTo(0, 0);
    rCtx.arc(0, 0, R, -0.5, 0.5);
    rCtx.closePath();
    rCtx.fillStyle = sg;
    rCtx.globalAlpha = 0.5;
    rCtx.fill();
    rCtx.restore();
  }

  // Sweep line
  rCtx.save();
  rCtx.translate(cx, cy);
  rCtx.rotate(radarAngle);
  rCtx.beginPath();
  rCtx.moveTo(0, 0);
  rCtx.lineTo(R, 0);
  rCtx.strokeStyle = c.glow;
  rCtx.lineWidth   = 2;
  rCtx.shadowColor = c.glow;
  rCtx.shadowBlur  = 10;
  rCtx.stroke();
  rCtx.restore();

  // Center dot
  rCtx.beginPath();
  rCtx.arc(cx, cy, 4, 0, Math.PI * 2);
  rCtx.fillStyle = c.glow;
  rCtx.shadowColor = c.glow;
  rCtx.shadowBlur  = 12;
  rCtx.fill();
  rCtx.shadowBlur  = 0;

  // Blips when detected
  if (radarMode === "detected") {
    const blips = [
      { x: cx + R * 0.35, y: cy - R * 0.45 },
      { x: cx - R * 0.55, y: cy + R * 0.25 },
      { x: cx + R * 0.60, y: cy + R * 0.55 },
    ];
    blips.forEach(b => {
      rCtx.beginPath();
      rCtx.arc(b.x, b.y, 4, 0, Math.PI * 2);
      rCtx.fillStyle = "#ef4444";
      rCtx.shadowColor = "#ef4444";
      rCtx.shadowBlur = 10;
      rCtx.fill();
      rCtx.shadowBlur = 0;
    });
  }

  const speed = radarMode === "scanning" ? 0.04 : 0.018;
  radarAngle = (radarAngle + speed) % (Math.PI * 2);
  radarAnim  = requestAnimationFrame(drawRadar);
}

drawRadar();

/* ══════════════════════════════════════════════════════════════
   ENTROPY BAR CHART
   ══════════════════════════════════════════════════════════════ */
function buildEntropyChart(labels, values) {
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }

  // Manual canvas bar chart (no external lib needed)
  const canvas = document.getElementById("entropyChart");
  const ctx    = canvas.getContext("2d");
  const W      = canvas.offsetWidth || 400;
  const H      = 160;
  canvas.width  = W;
  canvas.height = H;

  const PAD_L = 30, PAD_R = 12, PAD_T = 12, PAD_B = 30;
  const chartW = W - PAD_L - PAD_R;
  const chartH = H - PAD_T - PAD_B;
  const n      = labels.length;
  const barW   = Math.max(8, chartW / n - 4);
  const maxVal = 8; // Shannon entropy max bits/byte

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  [0, 2, 4, 6, 8].forEach(v => {
    const y = PAD_T + chartH - (v / maxVal) * chartH;
    ctx.beginPath();
    ctx.moveTo(PAD_L, y);
    ctx.lineTo(PAD_L + chartW, y);
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "rgba(126,170,204,0.5)";
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.fillText(v, 2, y + 3);
  });

  // Threshold line at 7.0
  const threshY = PAD_T + chartH - (7 / maxVal) * chartH;
  ctx.beginPath();
  ctx.moveTo(PAD_L, threshY);
  ctx.lineTo(PAD_L + chartW, threshY);
  ctx.strokeStyle = "rgba(245,158,11,0.5)";
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(245,158,11,0.7)";
  ctx.font = "9px JetBrains Mono, monospace";
  ctx.fillText("7.0 threshold", PAD_L + chartW - 70, threshY - 4);

  // Bars
  labels.forEach((lbl, i) => {
    const val  = values[i];
    const bH   = (val / maxVal) * chartH;
    const x    = PAD_L + i * (chartW / n) + (chartW / n - barW) / 2;
    const y    = PAD_T + chartH - bH;
    const high = val >= 7.0;

    const grad = ctx.createLinearGradient(x, y, x, y + bH);
    if (high) {
      grad.addColorStop(0, "rgba(239,68,68,0.9)");
      grad.addColorStop(1, "rgba(239,68,68,0.3)");
    } else {
      grad.addColorStop(0, "rgba(0,212,255,0.85)");
      grad.addColorStop(1, "rgba(0,212,255,0.2)");
    }

    ctx.fillStyle = grad;
    ctx.shadowColor = high ? "#ef4444" : "#00d4ff";
    ctx.shadowBlur  = 6;
    ctx.fillRect(x, y, barW, bH);
    ctx.shadowBlur  = 0;

    // X label
    ctx.fillStyle = "rgba(126,170,204,0.6)";
    ctx.font      = "8px JetBrains Mono, monospace";
    const lblShort = lbl.replace("document_", "").replace(".txt", "");
    ctx.fillText(lblShort, x + barW / 2 - 6, H - 6);
  });
}

/* ══════════════════════════════════════════════════════════════
   THREAT METER
   ══════════════════════════════════════════════════════════════ */
function updateThreatMeter(score) {
  const pct = Math.min(1, Math.max(0, score)) * 100;
  document.getElementById("threatBarFill").style.width  = pct + "%";
  document.getElementById("threatMarker").style.left    = pct + "%";
  document.getElementById("threatValue").textContent    = (score || 0).toFixed(2);
}

/* ══════════════════════════════════════════════════════════════
   AUDIT LOG
   ══════════════════════════════════════════════════════════════ */
let renderedLogCount = 0;

function appendAuditEntries(entries) {
  const box = document.getElementById("auditLog");
  const newEntries = entries.slice(renderedLogCount);
  newEntries.forEach(msg => {
    const div = document.createElement("div");
    div.className = "audit-entry";
    if (msg.includes("RANSOMWARE") || msg.includes("🚨") || msg.includes("[ENC]")) div.classList.add("danger");
    else if (msg.includes("ROLL") || msg.includes("Restored") || msg.includes("backed")) div.classList.add("success");
    else if (msg.includes("snapshot") || msg.includes("Snapshot") || msg.includes("Alert")) div.classList.add("warn");
    div.textContent = msg;
    box.appendChild(div);
  });
  renderedLogCount = entries.length;
  box.scrollTop = box.scrollHeight;
}

function clearAuditLog() {
  document.getElementById("auditLog").innerHTML =
    '<div class="audit-entry init">[SYSTEM] Log cleared.</div>';
  renderedLogCount = 0;
}

/* ══════════════════════════════════════════════════════════════
   ROLLBACK LIST
   ══════════════════════════════════════════════════════════════ */
function renderRollback(affected, restored) {
  const box = document.getElementById("rollbackList");
  if (!affected.length) {
    box.innerHTML = '<div class="empty-msg">No rollback data yet</div>';
    return;
  }
  const restoredSet = new Set(restored);
  box.innerHTML = affected.map(fp => {
    const fname = fp.split(/[/\\]/).pop();
    const ok    = restoredSet.has(fp);
    return `<div class="rollback-item ${ok ? "" : "encrypted"}">
      <span>${ok ? "✔" : "✖"}</span>
      <span>${fname}</span>
      <span style="margin-left:auto;font-size:9px">${ok ? "RESTORED" : "ENCRYPTED"}</span>
    </div>`;
  }).join("");
}

/* ══════════════════════════════════════════════════════════════
   LAYERS NOTIFICATION
   ══════════════════════════════════════════════════════════════ */
function setLayersActive(active) {
  ["layerNetwork","layerZero","layerContent"].forEach((id,i) => {
    const el  = document.getElementById(id);
    const sid = [id+"Status","layerZeroStatus","layerContentStatus"][i];
    const st  = document.getElementById([
      "layerNetworkStatus","layerZeroStatus","layerContentStatus"
    ][i]);
    if (active) {
      el.classList.add("active");
      st.textContent = "ALERTED";
    } else {
      el.classList.remove("active");
      st.textContent = "—";
    }
  });
  // fix: individual IDs
  const ids = ["layerNetworkStatus","layerZeroStatus","layerContentStatus"];
  const chips = ["layerNetwork","layerZero","layerContent"];
  chips.forEach((cid, i) => {
    const chip = document.getElementById(cid);
    const stat = document.getElementById(ids[i]);
    if (active) { chip.classList.add("active"); stat.textContent = "ALERTED"; }
    else        { chip.classList.remove("active"); stat.textContent = "—"; }
  });
}

/* ══════════════════════════════════════════════════════════════
   MODAL
   ══════════════════════════════════════════════════════════════ */
function showModal(data) {
  const body = document.getElementById("modalBody");
  const proc = data.alerts?.[0]?.process || {};
  body.innerHTML = `
    <div class="modal-row"><span class="modal-key">Process</span><span class="modal-value danger">${proc.name || "cryptolocker_sim.exe"}</span></div>
    <div class="modal-row"><span class="modal-key">PID</span><span class="modal-value">${proc.pid ?? 99999}</span></div>
    <div class="modal-row"><span class="modal-key">Anomaly Score</span><span class="modal-value danger">${(proc.anomaly_score ?? data.anomaly_score ?? 0).toFixed(4)}</span></div>
    <div class="modal-row"><span class="modal-key">Isolation Forest</span><span class="modal-value danger">${data.iso_score ?? "N/A"}</span></div>
    <div class="modal-row"><span class="modal-key">Random Forest P(ransom)</span><span class="modal-value danger">${data.rf_prob != null ? (data.rf_prob * 100).toFixed(1) + "%" : "N/A"}</span></div>
    <div class="modal-row"><span class="modal-key">Files Encrypted</span><span class="modal-value">${data.files_affected?.length ?? 0}</span></div>
    <div class="modal-row"><span class="modal-key">Files Restored</span><span class="modal-value" style="color:var(--green)">${data.files_restored?.length ?? 0}</span></div>
    <div class="modal-row"><span class="modal-key">Layers Notified</span><span class="modal-value">NetworkGuardian, ZeroTrustAuth, ContentThreat</span></div>
    <div class="modal-row"><span class="modal-key">Recommended Action</span><span class="modal-value danger">Suspend all sessions. Escalate to SOC.</span></div>
  `;
  document.getElementById("alertModal").classList.remove("hidden");
}
function closeModal() {
  document.getElementById("alertModal").classList.add("hidden");
}

/* ══════════════════════════════════════════════════════════════
   STATE → UI RENDERER
   ══════════════════════════════════════════════════════════════ */
let modalShown = false;

function applyState(data) {
  const status = data.status;

  // Connection badge
  const badge = document.getElementById("connectionBadge");
  badge.classList.add("online");
  badge.classList.remove("offline");
  document.getElementById("connectionDot").style.background = "var(--green)";
  document.getElementById("connectionLabel").textContent    = "ONLINE";

  // Alert count
  document.getElementById("alertCountBadge").textContent =
    (data.alerts?.length || 0) + " ALERTS";

  // Scores
  const fmt = v => v != null ? v.toFixed(4) : "—";
  const isoEl = document.getElementById("isoScore");
  isoEl.textContent = fmt(data.iso_score);
  isoEl.className   = "score-val" + (data.iso_score != null && data.iso_score < 0 ? " danger" : "");

  const rfEl = document.getElementById("rfProb");
  rfEl.textContent  = data.rf_prob != null ? (data.rf_prob * 100).toFixed(1) + "%" : "—";
  rfEl.className    = "score-val" + (data.rf_prob != null && data.rf_prob > 0.7 ? " danger" : "");

  document.getElementById("anomalyScore").textContent = fmt(data.anomaly_score);

  const verdictEl = document.getElementById("verdictScore");
  if (status === "detected") { verdictEl.textContent = "RANSOMWARE"; verdictEl.className = "score-val danger"; }
  else if (status === "clean") { verdictEl.textContent = "CLEAN"; verdictEl.className = "score-val safe"; }
  else if (status === "running") { verdictEl.textContent = "SCANNING…"; verdictEl.className = "score-val"; }
  else { verdictEl.textContent = "—"; verdictEl.className = "score-val"; }

  // File stats
  document.getElementById("filesAffected").textContent = data.files_affected?.length ?? 0;
  document.getElementById("filesRestored").textContent = data.files_restored?.length ?? 0;
  document.getElementById("demoStatus").textContent    = status.toUpperCase();

  // Threat meter
  const threat = data.anomaly_score != null ? Math.min(1, data.anomaly_score) : 0;
  updateThreatMeter(threat);

  // Radar mode
  if (status === "running")   { radarMode = "scanning"; }
  else if (status === "detected") { radarMode = "detected"; }
  else if (status === "clean")    { radarMode = "clean"; }
  else                            { radarMode = "idle"; }

  // Radar label
  const labelMap = {
    idle:     ["WAITING",  "Select RUN DEMO to begin"],
    running:  ["SCANNING", "Analyzing behavioral patterns…"],
    detected: ["THREAT",   "Ransomware confirmed — response active"],
    clean:    ["CLEAN",    "No ransomware detected"],
    error:    ["ERROR",    "Check agent logs"],
  };
  const [rl, rs] = labelMap[status] || labelMap.idle;
  document.getElementById("radarLabel").textContent = rl;
  document.getElementById("radarSub").textContent   = rs;

  // Body class for CSS animations
  document.body.className = "";
  if (status === "running")   document.body.classList.add("scanning");
  if (status === "detected")  document.body.classList.add("detected");
  if (status === "clean")     document.body.classList.add("clean");

  // Entropy chart
  if (data.entropy_log?.length) {
    const labels = data.entropy_log.map(e => e.file);
    const values = data.entropy_log.map(e => e.entropy);
    buildEntropyChart(labels, values);
    document.getElementById("entropyBadge").textContent = labels.length + " files";
  }

  // Rollback
  renderRollback(data.files_affected || [], data.files_restored || []);

  // Layers
  setLayersActive(status === "detected");

  // Alert banner
  const banner = document.getElementById("alertBanner");
  if (status === "detected") {
    banner.classList.remove("hidden");
    document.getElementById("alertBannerSub").textContent =
      `PID 99999 killed · ${data.files_restored?.length ?? 0} files restored`;
  } else {
    banner.classList.add("hidden");
  }

  // Audit log
  if (data.audit_log?.length) appendAuditEntries(data.audit_log);

  // Buttons
  const running = status === "running";
  document.getElementById("btnRun").disabled      = running;
  document.getElementById("btnSimulate").disabled = running;

  // Modal (once)
  if (status === "detected" && !modalShown) {
    modalShown = true;
    setTimeout(() => showModal(data), 600);
  }
  if (status !== "detected") modalShown = false;

  lastStatus = status;
}

/* ══════════════════════════════════════════════════════════════
   API CALLS
   ══════════════════════════════════════════════════════════════ */
async function fetchStatus() {
  try {
    const res  = await fetch(`${API}/status`);
    const data = await res.json();
    applyState(data);
  } catch (e) {
    // Mark offline
    const badge = document.getElementById("connectionBadge");
    badge.classList.remove("online");
    badge.classList.add("offline");
    document.getElementById("connectionLabel").textContent = "OFFLINE";
    document.getElementById("connectionDot").style.background = "var(--red)";
  }
}

async function runDemo() {
  try {
    renderedLogCount = 0;
    document.getElementById("auditLog").innerHTML = "";
    await fetch(`${API}/run`, { method: "POST" });
    addLocalLog("[SYSTEM] Demo started — running agent pipeline…", "warn");
  } catch (e) {
    addLocalLog("[ERROR] Cannot reach API server. Is it running?", "danger");
  }
}

async function simulateDetection() {
  // Patch state via a quick manual run that injects simulated data
  addLocalLog("[SIM] Injecting simulated ransomware detection event…", "warn");
  try {
    await fetch(`${API}/reset`, { method: "POST" });
    await fetch(`${API}/run`,   { method: "POST" });
  } catch {
    addLocalLog("[ERROR] API server not reachable.", "danger");
  }
}

async function resetState() {
  try {
    await fetch(`${API}/reset`, { method: "POST" });
    renderedLogCount = 0;
    document.getElementById("auditLog").innerHTML =
      '<div class="audit-entry init">[SYSTEM] State reset. Ready.</div>';
    entropyData = [];
    const canvas = document.getElementById("entropyChart");
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("entropyBadge").textContent = "0 files";
    radarMode = "idle";
    updateThreatMeter(0);
    modalShown = false;
    await fetchStatus();
  } catch (e) {
    addLocalLog("[ERROR] Reset failed — API not reachable.", "danger");
  }
}

function addLocalLog(msg, cls = "") {
  const box = document.getElementById("auditLog");
  const div = document.createElement("div");
  div.className = "audit-entry " + cls;
  const ts = new Date().toLocaleTimeString("en-GB", { hour12: false });
  div.textContent = `[${ts}] ${msg}`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

/* ══════════════════════════════════════════════════════════════
   BOOT
   ══════════════════════════════════════════════════════════════ */
window.addEventListener("DOMContentLoaded", () => {
  fetchStatus();
  pollTimer = setInterval(fetchStatus, 1500);

  // Auto-resize entropy canvas on window resize
  window.addEventListener("resize", () => {
    if (entropyData.length) buildEntropyChart(
      entropyData.map(e => e.file),
      entropyData.map(e => e.entropy)
    );
  });
});
