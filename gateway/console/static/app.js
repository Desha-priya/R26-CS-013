/* NeuraShield console controller. Same-origin only; the console attaches the
   platform API key server-side, so the browser never holds it. */

/* ---- theme (light/dark) ---- */
function applyTheme(t){
  document.documentElement.setAttribute("data-theme", t);
  const ic = document.getElementById("themeIcon");
  if (ic) ic.innerHTML = (t === "dark")
    ? '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/>'      // moon
    : '<circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/>';
}
(function initTheme(){
  let t = "light";
  try { t = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"; } catch(_){}
  try { t = localStorage.getItem("ns_theme") || t; } catch(_){}
  document.documentElement.setAttribute("data-theme", t);  // before paint
})();

let CSRF = "";
let CONSENT = false;
let LAST_LAYERS = {};
let degradedStreak = 0;        // debounce for posture flicker
let bootGraceUntil = Date.now() + 12000; // show "Starting" for first 12s

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "X-CSRF-Token": CSRF, ...(opts.headers || {}) },
    method: opts.method || "GET", body: opts.body, credentials: "same-origin",
  });
  if (r.status === 401) { location.href = "/login"; throw new Error("auth"); }
  return r.json();
}
function esc(s){ const d=document.createElement("div"); d.textContent=s==null?"":String(s); return d.innerHTML; }
function fmtTime(ts){ try{ const d=typeof ts==="number"?new Date(ts*1000):new Date(ts);
  return d.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"}); }catch{ return ""; } }

const LAYER_META = {
  zta:        { name:"Zero-Trust Authentication", short:"Zero-Trust", n:3 },
  network:    { name:"Network Guardian", short:"Network Guardian", n:1 },
  ransomware: { name:"Ransomware Killer", short:"Ransomware", n:2 },
  content:    { name:"Content Threat Detection", short:"Content Threat", n:4 },
};
const CONSEQUENCE = {
  zta: { name:"Zero-Trust Authentication", points:[
    "The computer will no longer re-verify that the logged-in person is still the legitimate user.",
    "Keystroke-behaviour monitoring and liveness checks stop.",
    "A stolen or unlocked session will not be detected by behaviour." ]},
  network: { name:"Network Guardian", points:[
    "Live network traffic will no longer be inspected.",
    "Command-and-control, exfiltration and suspicious-port activity will not be flagged." ]},
  ransomware: { name:"Ransomware Killer", points:[
    "Files will no longer be watched for ransomware behaviour.",
    "No automatic rollback of encrypted files and no clean backups are taken while this is off." ]},
  content: { name:"Content Threat Detection", points:[
    "Email, call and media content will no longer be screened.",
    "Phishing text, deepfake video and spoofed audio will not be detected." ]},
};

/* ---- navigation ---- */
function showView(name){
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  const el = document.getElementById("view-" + name);
  if (el) el.classList.add("active");
}
document.getElementById("nav").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-view]"); if (!b || b.disabled) return;
  document.querySelectorAll(".nav button").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  const v = b.dataset.view;
  if (b.dataset.embed) { openEmbed(b.dataset.embed); showView("embed"); return; }
  showView(v);
  if (v === "alerts") loadAlerts();
  if (v === "audit") loadAudit();
  if (v === "consent") loadConsent();
  if (v === "settings") loadSettings();
  if (v === "incidents") loadIncidents();
});

/* ---- incidents ---- */
async function loadIncidents(){
  const d = await api("/api/incidents");
  const c = d.counts || {};
  document.getElementById("incOpen").textContent = c.open || 0;
  document.getElementById("incAck").textContent = c.acknowledged || 0;
  document.getElementById("incResolved").textContent = c.resolved || 0;
  const box = document.getElementById("incidentList");
  const items = d.incidents || [];
  if (!items.length) { box.innerHTML = `<div class="empty">No incidents recorded. All clear.</div>`; return; }
  box.innerHTML = items.map(it => `<div class="inc-row" data-id="${esc(it.id)}">
    <div class="inc-sev ${esc(it.severity)}"></div>
    <div class="inc-layer">${esc(it.layer)}</div>
    <div class="inc-body">
      <div class="inc-title">${esc(it.title)}</div>
      <div class="inc-action">Action: ${esc(it.action||'—')}</div>
      ${it.note ? `<div class="inc-note">Note: ${esc(it.note)}</div>` : ``}
      <div class="inc-actions">
        ${it.status!=="acknowledged"&&it.status!=="resolved" ? `<button class="inc-btn" data-act="acknowledged">Acknowledge</button>`:``}
        ${it.status!=="resolved" ? `<button class="inc-btn" data-act="resolved">Resolve</button>`:``}
        <button class="inc-btn" data-act="note">Add note</button>
      </div>
    </div>
    <div class="inc-right">
      <span class="inc-time">${esc(it.ts?fmtTime(it.ts):'')}</span>
      <span class="inc-status ${esc(it.status)}">${esc(it.status)}</span>
    </div>
  </div>`).join("");
  box.querySelectorAll(".inc-btn").forEach(b => b.addEventListener("click", async e => {
    const row = e.target.closest(".inc-row"); const id = row.dataset.id; const act = b.dataset.act;
    let note;
    if (act === "note") { note = prompt("Add a note to this incident:"); if (note===null) return; }
    await api(`/api/incidents/${encodeURIComponent(id)}/status`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(act==="note" ? {status:"acknowledged", note} : {status:act}) });
    loadIncidents();
  }));
}

/* ---- notification settings ---- */
let NS = null;
function _setToggle(id, on){ const t=document.getElementById(id); if(t) t.classList.toggle("on", !!on); }
function fmtBytes(n){
  if (!n) return "0 KB"; const u = ["B","KB","MB","GB"]; let i = 0; n = Number(n);
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i ? n.toFixed(n < 10 ? 1 : 0) : n) + " " + u[i];
}
async function loadDataInfo(){
  const el = document.getElementById("dataInfo"); if (!el) return;
  let i; try { i = await api("/api/data/info"); } catch(_) { return; }
  el.innerHTML = `Stored on this computer only:<br>
    <span class="mono">${esc(i.data_dir)}</span> · ${fmtBytes(i.data_bytes)} (account, audit log, alerts, layer data, logs)<br>
    <span class="mono">${esc(i.config_file)}</span> (consent and layer choices)<br>
    <span class="mono">${esc(i.secrets_dir)}</span> (platform key, encrypted)` +
    (i.backups_exist ? `<br><span class="mono">${esc(i.backups_dir)}</span> · ${fmtBytes(i.backups_bytes)} (ransomware backup copies of your files)` : "");
}

let _dmMode = null;
function openDataModal(mode){
  _dmMode = mode;
  const del = mode === "delete";
  document.getElementById("dmTitle").textContent = del ? "Delete account and start fresh" : "Clear my data";
  document.getElementById("dmClearOpts").style.display = del ? "none" : "";
  document.getElementById("dmDeleteOpts").style.display = del ? "" : "none";
  document.getElementById("dmGo").textContent = del ? "Delete everything" : "Clear selected";
  ["dmPassword","dmConfirm"].forEach(id => { const e = document.getElementById(id); if (e) e.value = ""; });
  ["dmAudit","dmLayers","dmBackups","dmDelBackups"].forEach(id => { const e = document.getElementById(id); if (e) e.checked = false; });
  document.getElementById("dmAlerts").checked = true;
  const err = document.getElementById("dmErr"); err.classList.remove("show"); err.textContent = "";
  document.getElementById("dataModal").classList.add("show");
  setTimeout(() => document.getElementById("dmPassword").focus(), 50);
}
function closeDataModal(){ document.getElementById("dataModal").classList.remove("show"); }
async function submitDataModal(){
  const err = document.getElementById("dmErr"), go = document.getElementById("dmGo");
  const pw = document.getElementById("dmPassword").value;
  const fail = m => { err.textContent = m; err.classList.add("show"); go.disabled = false; };
  err.classList.remove("show");
  if (!pw) return fail("Enter your password to continue.");
  go.disabled = true;
  try {
    if (_dmMode === "delete") {
      if (document.getElementById("dmConfirm").value.trim() !== "DELETE") return fail("Type DELETE (capital letters) to confirm.");
      const r = await fetch("/api/account/delete", { method:"POST", credentials:"same-origin",
        headers:{ "Content-Type":"application/json", "X-CSRF-Token": CSRF },
        body: JSON.stringify({ password: pw, confirm: "DELETE", backups: document.getElementById("dmDelBackups").checked }) });
      if (!r.ok) { let m = "Could not delete the account."; try { m = (await r.json()).detail || m; } catch(_){} return fail(m); }
      closeDataModal(); waitForRestart();
    } else {
      const body = { password: pw, alerts: document.getElementById("dmAlerts").checked, audit: document.getElementById("dmAudit").checked,
                     layers: document.getElementById("dmLayers").checked, backups: document.getElementById("dmBackups").checked };
      if (!body.alerts && !body.audit && !body.layers && !body.backups) return fail("Select something to clear.");
      const r = await fetch("/api/data/clear", { method:"POST", credentials:"same-origin",
        headers:{ "Content-Type":"application/json", "X-CSRF-Token": CSRF }, body: JSON.stringify(body) });
      if (!r.ok) { let m = "Could not clear the data."; try { m = (await r.json()).detail || m; } catch(_){} return fail(m); }
      closeDataModal(); go.disabled = false;
      _knownAlertIds = null; pollHistory(true); loadDataInfo(); loadOverview();
      showToast({severity:"low", layer_name:"NeuraShield", title:"Data cleared", detail:"The selected data was removed.", ts: Date.now()/1000});
    }
  } catch(e) { fail("Something went wrong. Try again."); }
}
function waitForRestart(){
  document.getElementById("restartOverlay").hidden = false;
  let wentDown = false, tries = 0;
  const tick = async () => {
    tries++;
    let up = false;
    try { const r = await fetch("/setup", { cache:"no-store" }); up = r.ok; } catch(_) { up = false; }
    if (!up) wentDown = true;
    if ((up && wentDown) || tries > 40) { location.href = "/setup"; return; }
    setTimeout(tick, 2000);
  };
  setTimeout(tick, 2500);
}
document.getElementById("btnClearData").onclick = () => openDataModal("clear");
document.getElementById("btnDeleteAccount").onclick = () => openDataModal("delete");
document.getElementById("dmCancel").onclick = closeDataModal;
document.getElementById("dmGo").onclick = submitDataModal;
document.getElementById("dmPassword").addEventListener("keydown", e => { if (e.key === "Enter") submitDataModal(); });

async function loadAutostart(){
  let a; try { a = await api("/api/autostart"); } catch(_) { return; }
  const t = document.getElementById("as-enabled"), st = document.getElementById("asState");
  if (!t) return;
  if (!a.supported) { t.classList.add("locked"); if (st) st.textContent = "Only available on Windows."; return; }
  t.classList.toggle("on", !!a.enabled);
  if (st) st.textContent = a.enabled
    ? (a.scope === "machine" ? "On (set by the installer for all users). Protection resumes after every restart."
                             : "On. Protection resumes after every restart; nothing monitors until consent is given.")
    : "Off. NeuraShield will only run when you open it.";
}
(function wireAutostart(){
  const t = document.getElementById("as-enabled"); if (!t) return;
  t.setAttribute("role","button"); t.setAttribute("tabindex","0");
  const act = async () => {
    if (t.classList.contains("locked")) return;
    const want = !t.classList.contains("on");
    t.classList.add("locked");
    try { await api("/api/autostart", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({enabled: want})}); }
    catch(_) {}
    t.classList.remove("locked"); loadAutostart();
  };
  t.addEventListener("click", act);
  t.addEventListener("keydown", e => { if (e.key===" "||e.key==="Enter"){ e.preventDefault(); act(); } });
})();

async function loadSettings(){
  loadAutostart();
  loadDataInfo();
  NS = await api("/api/notifications");
  _setToggle("n-enabled", NS.enabled);
  _setToggle("n-stepup", NS.stepup);
  const sev=document.getElementById("n-sev"); if(sev) sev.value = NS.min_severity || "medium";
  const L = NS.layers || {};
  ["zta","network","ransomware","content"].forEach(k => _setToggle("n-"+k, L[k] !== false));
}
async function saveSettings(){
  const L = {};
  ["zta","network","ransomware","content"].forEach(k => L[k] = document.getElementById("n-"+k).classList.contains("on"));
  const body = {
    enabled: document.getElementById("n-enabled").classList.contains("on"),
    stepup: document.getElementById("n-stepup").classList.contains("on"),
    min_severity: document.getElementById("n-sev").value, layers: L,
  };
  await api("/api/notifications", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
}
function _wireSettingToggle(id){ const t=document.getElementById(id); if(!t) return;
  const act=()=>{ t.classList.toggle("on"); saveSettings(); };
  t.setAttribute("role","button"); t.setAttribute("tabindex","0");
  t.addEventListener("click", act);
  t.addEventListener("keydown", e=>{ if(e.key===" "||e.key==="Enter"){ e.preventDefault(); act(); } });
}
["n-enabled","n-stepup","n-zta","n-network","n-ransomware","n-content"].forEach(_wireSettingToggle);
{
  const sev=document.getElementById("n-sev"); if(sev) sev.addEventListener("change", saveSettings);
  const test=document.getElementById("notifTest");
  if(test) test.addEventListener("click", async ()=>{ try{ await api("/api/notifications/test",{method:"POST"});
    test.textContent="Sent ✓"; setTimeout(()=>test.textContent="Send test",1500);}catch(_){ } });
}

/* ---- embedded layer dashboard ---- */
let _embedCurrent = null;
async function openEmbed(layerId){
  _embedCurrent = layerId;
  const m = LAYER_META[layerId] || { name: layerId };
  document.getElementById("embedTitle").textContent = m.name;
  document.getElementById("embedSub").textContent = "Layer " + (m.n||"") + " · part of the NeuraShield platform";
  const box = document.getElementById("layerDetail");
  box.innerHTML = `<div class="panel"><div class="panel-body"><div class="empty">Loading ${esc(m.name)}…</div></div></div>`;
  await renderLayerDetail(layerId);
}

async function renderLayerDetail(layerId){
  const box = document.getElementById("layerDetail");
  let d;
  try { d = await api("/api/layer/" + layerId + "/detail"); }
  catch(_) { return; }
  if (_embedCurrent !== layerId) return;  // user navigated away
  if (!d.running) {
    box.innerHTML = `<div class="panel"><div class="panel-body">
      <div class="empty">This layer is not running yet. Enable it under <b>Layers</b>, then it reports here.</div>
    </div></div>`;
    return;
  }
  const stats = (d.stats||[]).map(s =>
    `<div class="stat-cell"><div class="stat-label">${esc(s.label)}</div>
       <div class="stat-value" style="font-size:22px">${esc(s.value)}</div></div>`).join("");
  const alerts = (d.alerts||[]).length
    ? d.alerts.map(a => `<div class="alert-row">
        <div class="sev-bar ${esc((a.sev||'info'))}"></div>
        <div class="alert-src">${esc(LAYER_META[layerId]?LAYER_META[layerId].short:layerId)}</div>
        <div class="alert-msg">${esc(a.msg||'')}</div>
        <div class="alert-time">${esc(a.ts?fmtTime(a.ts):'')}</div></div>`).join("")
    : `<div class="empty">No recent activity.</div>`;
  box.innerHTML = `
    <div class="stat-strip" style="grid-template-columns:repeat(${Math.max(1,(d.stats||[]).length)},1fr)">${stats}</div>
    <div class="panel"><div class="panel-head"><span class="panel-title">Recent activity</span>
      <a class="open-link" id="layerOpenExt" href="#" title="Open the layer's full dashboard in a tab">Full dashboard ↗</a></div>
      <div class="panel-body">${alerts}</div></div>`;
  const ext = document.getElementById("layerOpenExt");
  if (ext) ext.onclick = e => { e.preventDefault(); window.open("/embed/" + layerId, "_blank"); };
}

/* ---- modal ---- */
const modal = document.getElementById("modal");
let modalResolve = null;
function confirmModal(title, bodyHTML, confirmLabel) {
  document.getElementById("modalTitle").textContent = title;
  document.getElementById("modalBody").innerHTML = bodyHTML;
  document.getElementById("modalConfirm").textContent = confirmLabel || "Turn off";
  modal.classList.add("show");
  return new Promise(res => { modalResolve = res; });
}
document.getElementById("modalCancel").onclick = () => { modal.classList.remove("show"); modalResolve && modalResolve(false); };
document.getElementById("modalConfirm").onclick = () => { modal.classList.remove("show"); modalResolve && modalResolve(true); };

/* ---- layer rows ---- */
function layerRowHTML(id, L) {
  const healthy = L.running && L.healthy;
  const led = !L.running ? "off" : healthy ? "on" : "bad";
  const state = !L.running ? "Stopped" : healthy ? "Running" : "Unhealthy";
  const stateCls = !L.running ? "off" : healthy ? "on" : "bad";
  let right;
  if (L.is_platform) right = `<span class="pill-static">Always on</span>`;
  else {
    const on = L.enabled ? "on" : "";
    right = `<span class="toggle ${on}" data-layer="${id}" role="button" tabindex="0" title="Toggle ${esc(L.name)}"><span class="switch"></span></span>`;
  }
  const openable = (L.running && !L.is_platform);
  const open = openable
    ? `<a class="open-link" data-open="${id}" href="#">Open ↗</a>`
    : `<span style="width:52px"></span>`;
  return `<div class="layer-row">
    <span class="led ${led}"></span>
    <div><div class="layer-name">${esc(L.name)}</div>
      <div class="layer-meta">Layer ${L.layer_number ?? "—"} · :${L.port ?? "—"} · <span class="layer-id">${esc(id)}</span></div></div>
    <div style="display:flex;align-items:center;gap:12px"><span class="state-txt ${stateCls}">${state}</span>${open}</div>
    <div style="text-align:right">${right}</div>
  </div>`;
}

function wireRows(scope) {
  scope.querySelectorAll(".toggle:not(.locked)").forEach(t => {
    const act = async () => {
      const id = t.dataset.layer;
      const turningOff = t.classList.contains("on");
      if (turningOff) {
        const c = CONSEQUENCE[id] || { name: id, points: ["This protection layer will stop."] };
        const ok = await confirmModal(`Turn off ${c.name}?`,
          `While this layer is off:<ul>${c.points.map(p => `<li>${esc(p)}</li>`).join("")}</ul>
           <p style="margin-top:10px">You can turn it back on at any time.</p>`);
        if (!ok) return;
      }
      t.classList.add("locked");
      try { await api(`/api/layer/${id}/${turningOff ? "stop" : "start"}`, { method: "POST" }); setTimeout(loadOverview, 700); }
      catch (_) { t.classList.remove("locked"); }
    };
    t.addEventListener("click", act);
    t.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } });
  });
  scope.querySelectorAll("[data-open]").forEach(a => {
    a.addEventListener("click", e => { e.preventDefault();
      const id = a.dataset.open;
      const nb = document.querySelector(`.nav button[data-embed="${id}"]`);
      if (nb) nb.click(); });
  });
}

/* ---- threat pill ---- */
function setThreat(lvl) {
  const pill = document.getElementById("threatPill");
  pill.className = "threat-pill " + lvl;
  const txt = { low: "Secure", medium: "Elevated", high: "High", critical: "Critical" };
  document.getElementById("threatText").textContent = txt[lvl] || "—";
}

/* ---- risk meter ---- */
function renderRisk(risk) {
  const panel = document.getElementById("riskPanel");
  const bd = document.getElementById("riskBreakdown");
  if (!risk || !risk.available) {
    panel.classList.add("off");
    document.getElementById("riskPct").textContent = "—";
    document.getElementById("riskFill").style.width = "0%";
    document.getElementById("riskDecision").textContent = CONSENT ? "standby" : "no consent";
    document.getElementById("riskDecision").className = "risk-decision";
    document.getElementById("riskSub").textContent = CONSENT
      ? "Zero-Trust is off or still initialising" : "awaiting consent";
    bd.innerHTML = "";
    return;
  }
  panel.classList.remove("off");
  const pct = (risk.percent != null) ? risk.percent : Math.round((risk.score || 0) * 100);
  document.getElementById("riskPct").textContent = pct;
  document.getElementById("riskFill").style.width = Math.max(2, Math.min(100, pct)) + "%";
  const thr = (risk.threshold != null) ? Math.round(risk.threshold * 100) : 85;
  document.getElementById("riskThreshold").style.left = Math.min(98, thr) + "%";
  const dec = (risk.decision || "monitor").toLowerCase();
  const d = document.getElementById("riskDecision");
  d.textContent = dec.replace(/_/g, " ");
  d.className = "risk-decision " + (dec.includes("mon") ? "monitor" : dec.includes("re") ? "reauth" : "step-up");
  document.getElementById("riskSub").textContent =
    `Zero-Trust live session risk · step-up at ${thr}%`;
  const parts = [["Behaviour", risk.behavioral], ["Alerts", risk.alert], ["Context", risk.context]];
  bd.innerHTML = parts.map(([lab, v]) => {
    const p = v == null ? 0 : Math.round(v * 100);
    return `<div class="rb-row"><span class="rb-label">${lab}</span>
      <div class="rb-track"><div class="rb-fill" style="width:${p}%"></div></div>
      <span class="rb-val">${v == null ? "—" : p + "%"}</span></div>`;
  }).join("");
}

/* ---- protection status banner ---- */
const _HERO_GLYPH = {
  ok:      "m10.5 16.5 4 4 7-8",
  warn:    "M16 10v8M16 21.5v.5",
  crit:    "M16 10v8M16 21.5v.5",
  standby: "M13 11v10M19 11v10",
};
function renderHero(d, lvl, coreOk, active, total){
  const box = document.getElementById("heroBanner"); if (!box) return;
  const layers = d.layers || {};
  const bad = Object.keys(layers).filter(k => !layers[k].is_platform && layers[k].enabled && (!layers[k].running || !layers[k].healthy));
  let state, title, sub;
  if (!CONSENT) {
    state = "standby"; title = "Protection is in standby";
    sub = "Nothing is being monitored. Accept the consent notice to start protecting this computer.";
  } else if (!coreOk && Date.now() < bootGraceUntil) {
    state = "standby"; title = "Starting protection…"; sub = "The platform is coming up; this takes a moment.";
  } else if (!coreOk) {
    state = "crit"; title = "Platform core is not responding"; sub = "Protection may be impaired. Open Layers to check the core.";
  } else if (lvl === "high" || lvl === "critical") {
    state = "crit"; title = "Attention needed";
    sub = `${d.n_active_alerts || 0} active alert(s) at threat level ${lvl}. Review them in Alerts.`;
  } else if (bad.length) {
    state = "warn"; title = "Protected, with issues";
    sub = `${bad.length} enabled layer(s) are not running yet — they may still be starting.`;
  } else {
    state = "ok"; title = "You are protected";
    sub = total ? `All ${total} enabled layer(s) are running. Last checked ${new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"})}.`
                : "The platform is running. Enable a layer under Layers to start monitoring.";
  }
  box.className = "hero " + state;
  document.getElementById("heroTitle").textContent = title;
  document.getElementById("heroSub").textContent = sub;
  document.getElementById("heroGlyph").setAttribute("d", _HERO_GLYPH[state]);
  document.getElementById("heroLayers").textContent = CONSENT ? active + " / " + total : "—";
  document.getElementById("heroAlerts").textContent = CONSENT ? (d.n_active_alerts ?? 0) : "—";
}

/* ---- alerts, last 24 hours ---- */
function renderChart(rows){
  const box = document.getElementById("alertChart"); if (!box) return;
  const now = Date.now()/1000, H = 3600, buckets = Array.from({length:24}, () => ({low:0, med:0, high:0}));
  let any = false;
  (rows || []).forEach(r => {
    const age = now - r.ts; if (age < 0 || age >= 24*H) return;
    const i = 23 - Math.floor(age / H), sv = r.severity;
    buckets[i][(sv === "high" || sv === "critical") ? "high" : (sv === "medium" ? "med" : "low")]++; any = true;
  });
  if (!any) { box.innerHTML = `<div class="empty">No alerts in the last 24 hours.</div>`; return; }
  const max = Math.max(1, ...buckets.map(b => b.low + b.med + b.high));
  box.innerHTML = `<div class="bars">` + buckets.map((b, i) => {
    const hr = new Date((now - (23 - i) * H) * 1000).getHours();
    const seg = (n, c) => n ? `<i class="seg ${c}" style="height:${(n / max) * 100}%"></i>` : "";
    return `<div class="bar" title="${String(hr).padStart(2,"0")}:00 — ${b.high} high, ${b.med} medium, ${b.low} low">${seg(b.high,"high")}${seg(b.med,"med")}${seg(b.low,"low")}</div>`;
  }).join("") + `</div><div class="bar-axis"><span>24 h ago</span><span>12 h</span><span>now</span></div>`;
}

/* ---- layer dashboard nav ---- */
function renderLayerNav(layers) {
  const box = document.getElementById("layerNav");
  const order = ["zta", "network", "ransomware", "content"];
  box.innerHTML = order.filter(id => layers[id]).map(id => {
    const L = layers[id];
    const on = L.running;
    const dis = on ? "" : "disabled";
    const onc = on ? "on" : "";
    const label = (LAYER_META[id] && LAYER_META[id].short) || L.name;
    return `<button class="layer-nav ${onc}" data-view="embed" data-embed="${id}" ${dis}
       title="${on ? "Open "+esc(L.name) : esc(L.name)+" is off — enable it in Layers"}">
       <svg class="nav-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 9h16" opacity=".5"/></svg>
       <span class="layer-nav-label">${esc(label)}</span>
       <span class="nav-led ${onc}"></span></button>`;
  }).join("");
}

/* ---- overview ---- */
async function loadOverview() {
  let d;
  try { d = await api("/api/overview"); } catch(_) { return; }
  LAST_LAYERS = d.layers || {};
  CONSENT = !!(d.consent && d.consent.accepted);
  const layers = d.layers || {};
  const ids = Object.keys(layers);
  const monitoring = ids.filter(k => !layers[k].is_platform);
  const active = monitoring.filter(k => layers[k].running && layers[k].healthy).length;
  document.getElementById("ovLayers").textContent = active;
  document.getElementById("ovLayersTotal").textContent = monitoring.length;

  // posture with debounce (no flicker). The platform responding is the
  // authoritative "core is up" signal; the supervisor's process view can lag.
  const coreOk = d.platform_ok || (layers.core && layers.core.running && layers.core.healthy);
  const pEl = document.getElementById("ovPosture");
  const subEl = document.getElementById("ovPostureSub");
  if (!CONSENT) {
    pEl.textContent = "Standby"; pEl.className = "stat-value warn"; subEl.textContent = "awaiting consent";
    degradedStreak = 0;
  } else if (coreOk) {
    pEl.textContent = "Active"; pEl.className = "stat-value steel"; subEl.textContent = "platform online";
    degradedStreak = 0;
  } else if (Date.now() < bootGraceUntil) {
    pEl.textContent = "Starting"; pEl.className = "stat-value"; subEl.textContent = "platform is starting…";
  } else {
    degradedStreak++;
    if (degradedStreak >= 2) { pEl.textContent = "Degraded"; pEl.className = "stat-value warn"; subEl.textContent = "core not responding"; }
    // else keep whatever is showing until a second bad read confirms it
  }

  document.getElementById("ovAlerts").textContent = d.n_active_alerts ?? 0;
  const lvl = CONSENT ? (d.threat_level || "low") : "low";
  document.getElementById("ovThreat").textContent = "threat level " + lvl;
  setThreat(lvl);
  renderRisk(d.risk);
  renderHero(d, lvl, coreOk, active, monitoring.length);

  // attention banner when something needs the operator
  const att = document.getElementById("attentionBanner");
  if (CONSENT && (lvl === "high" || lvl === "critical")) {
    att.innerHTML = `<div class="attention high"><span class="att-ic">!</span>
      <span><strong>Threat level ${esc(lvl)}.</strong> ${d.n_active_alerts} active alert(s) need review.</span>
      <a href="#" id="attGo">View alerts →</a></div>`;
    const g=document.getElementById("attGo"); if(g) g.onclick=e=>{e.preventDefault();document.getElementById("nav-alerts").click();};
  } else if (CONSENT) {
    // unhealthy enabled layer?
    const bad = Object.keys(layers).filter(k=>!layers[k].is_platform && layers[k].enabled && (!layers[k].running||!layers[k].healthy));
    att.innerHTML = bad.length
      ? `<div class="attention medium"><span class="att-ic">!</span>
         <span><strong>${bad.length} enabled layer(s) are not running.</strong> They may still be starting.</span>
         <a href="#" id="attGo2">Open layers →</a></div>`
      : "";
    const g2=document.getElementById("attGo2"); if(g2) g2.onclick=e=>{e.preventDefault();document.getElementById("nav-layers").click();};
  } else att.innerHTML = "";

  const cb = document.getElementById("consentBanner");
  cb.innerHTML = CONSENT ? "" :
    `<div class="consent-status off" style="margin-bottom:16px">
       <strong>Protection is in standby.</strong>&nbsp;Accept the consent notice to start monitoring.
       <a href="#" id="goConsent" style="margin-left:auto">Review consent →</a></div>`;
  const gc = document.getElementById("goConsent");
  if (gc) gc.onclick = e => { e.preventDefault(); document.getElementById("nav-consent").click(); };

  const orderRows = ["zta","core","network","ransomware","content"].filter(k => layers[k]);
  const rows = orderRows.map(id => layerRowHTML(id, layers[id])).join("");
  document.getElementById("ovLayerList").innerHTML = rows;
  document.getElementById("layerList").innerHTML = rows;
  wireRows(document.getElementById("ovLayerList"));
  wireRows(document.getElementById("layerList"));
  renderLayerNav(layers);
  document.getElementById("ovClock").textContent = new Date().toLocaleTimeString();
}

/* ---- alerts: persistent history (survives restarts), with filters ---- */
function relTime(ts){
  const s = Math.max(0, Math.round(Date.now()/1000 - ts));
  if (s < 45) return "just now";
  if (s < 3600) return Math.round(s/60) + " min ago";
  if (s < 86400) return Math.round(s/3600) + " h ago";
  return new Date(ts*1000).toLocaleDateString([], {day:"2-digit", month:"short"}) + " " +
         new Date(ts*1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
}
function absTime(ts){ try { return new Date(ts*1000).toLocaleString(); } catch(_) { return ""; } }

function alertRowHTML(r){
  const sev = esc(r.severity || "medium");
  return `<div class="ar ${r.ack ? "acked" : ""}" data-id="${esc(r.id)}">
    <div class="sev-bar ${sev}"></div>
    <div class="ar-main">
      <div class="ar-top"><span class="chip layer">${esc(r.layer_name || r.layer)}</span>
        <span class="chip sev-${sev}">${sev}</span>
        <span class="ar-title">${esc(r.title)}</span></div>
      <div class="ar-detail">${esc(r.detail)}</div>
      ${r.action ? `<div class="ar-action">What to do: ${esc(r.action)}</div>` : ``}
    </div>
    <div class="ar-right"><span class="ar-time" title="${esc(absTime(r.ts))}">${esc(relTime(r.ts))}</span>
      ${r.ack ? `<span class="ar-done">acknowledged</span>` : `<button class="inc-btn" data-ack="${esc(r.id)}">Acknowledge</button>`}</div>
  </div>`;
}

function wireAckButtons(scope){
  scope.querySelectorAll("[data-ack]").forEach(b => b.addEventListener("click", async () => {
    b.disabled = true;
    try { await api("/api/history/" + encodeURIComponent(b.dataset.ack) + "/ack", {method:"POST"}); } catch(_) {}
    pollHistory(true); if (document.getElementById("view-alerts").classList.contains("active")) loadAlerts();
  }));
}

let _knownAlertIds = null;       // null until the first load: old alerts are history, not news
let _lastHistory = null;

function updateBadge(counts){
  const b = document.getElementById("alertBadge"); if (!b) return;
  const n = counts ? counts.unacknowledged : 0;
  b.textContent = n > 99 ? "99+" : n;
  b.hidden = !n;
  b.classList.toggle("hot", !!(counts && counts.unacknowledged_high));
}

function showToast(r){
  const box = document.getElementById("toastStack"); if (!box) return;
  const t = document.createElement("div");
  t.className = "toast sev-" + (r.severity || "medium");
  t.innerHTML = `<div class="toast-top"><span class="chip layer">${esc(r.layer_name || r.layer)}</span>
      <span class="toast-time">${esc(relTime(r.ts))}</span><span class="toast-x" role="button" aria-label="Dismiss">✕</span></div>
    <div class="toast-title">${esc(r.title)}</div><div class="toast-detail">${esc(r.detail)}</div>`;
  const kill = () => { t.classList.add("out"); setTimeout(() => t.remove(), 250); };
  t.querySelector(".toast-x").onclick = e => { e.stopPropagation(); kill(); };
  t.onclick = () => { document.getElementById("nav-alerts").click(); kill(); };
  box.appendChild(t);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(kill, 12000);
}

async function pollHistory(silent){
  let d;
  try { d = await api("/api/history?limit=300"); } catch(_) { return; }
  _lastHistory = d;
  updateBadge(d.counts);
  const rows = d.alerts || [];
  if (_knownAlertIds === null) { _knownAlertIds = new Set(rows.map(r => r.id)); }
  else if (!silent) {
    rows.filter(r => !_knownAlertIds.has(r.id)).reverse().forEach(r => { _knownAlertIds.add(r.id); if (!r.ack) showToast(r); });
  } else rows.forEach(r => _knownAlertIds.add(r.id));
  renderChart(rows);
  const box = document.getElementById("ovRecentAlerts");
  if (box) {
    const top = rows.slice(0, 8);
    box.innerHTML = top.length ? top.map(alertRowHTML).join("")
      : `<div class="empty">No alerts recorded. All clear.</div>`;
    wireAckButtons(box);
  }
}

async function loadAlerts() {
  const box = document.getElementById("alertList");
  const q = new URLSearchParams({ limit: "200" });
  const layer = document.getElementById("alLayer").value, sev = document.getElementById("alSev").value;
  if (layer) q.set("layer", layer);
  if (sev) q.set("severity", sev);
  if (document.getElementById("alUnacked").checked) q.set("unacked", "1");
  let d;
  try { d = await api("/api/history?" + q.toString()); }
  catch(_) { return; }
  const needle = (document.getElementById("alSearch").value || "").trim().toLowerCase();
  let rows = d.alerts || [];
  if (needle) rows = rows.filter(r => (r.title + " " + r.detail + " " + r.layer_name + " " + (r.action||"")).toLowerCase().includes(needle));
  document.getElementById("alCount").textContent = rows.length + " shown · " + ((d.counts||{}).total || 0) + " recorded";
  updateBadge(d.counts);
  if (!rows.length) { box.innerHTML = `<div class="empty">${needle || layer || sev ? "No alerts match these filters." : "No alerts — all clear."}</div>`; return; }
  box.innerHTML = rows.map(alertRowHTML).join("");
  wireAckButtons(box);
}
["alLayer","alSev","alUnacked"].forEach(id => document.getElementById(id).addEventListener("change", loadAlerts));
document.getElementById("alSearch").addEventListener("input", () => { clearTimeout(window._alT); window._alT = setTimeout(loadAlerts, 200); });
document.getElementById("ovAlertsAll").addEventListener("click", e => { e.preventDefault(); document.getElementById("nav-alerts").click(); });

/* ---- consent ---- */
async function loadConsent() {
  const c = await api("/api/consent");
  CONSENT = !!c.accepted;
  document.getElementById("consentDoc").innerHTML = window.CONSENT_HTML || "";
  const st = document.getElementById("consentState");
  st.innerHTML = CONSENT
    ? `<div class="consent-status on"><strong>Consent given.</strong>&nbsp;Monitoring is permitted. Accepted ${esc(fmtTime(c.accepted_at))}.</div>`
    : `<div class="consent-status off"><strong>Consent not given.</strong>&nbsp;Nothing is being monitored on this computer.</div>`;
  document.getElementById("btnAccept").style.display = CONSENT ? "none" : "";
  document.getElementById("btnWithdraw").style.display = CONSENT ? "" : "none";
}
document.getElementById("btnAccept").onclick = async () => { await api("/api/consent/accept", { method: "POST" }); bootGraceUntil = Date.now()+12000; await loadConsent(); loadOverview(); };
document.getElementById("btnWithdraw").onclick = async () => {
  const ok = await confirmModal("Withdraw consent?",
    `This stops <strong>all</strong> monitoring immediately:
     <ul><li>Zero-Trust, Network, Ransomware and Content layers all stop.</li>
     <li>The platform and this console stay running, but nothing is monitored.</li></ul>
     <p style="margin-top:10px">You can give consent again at any time.</p>`, "Withdraw");
  if (!ok) return;
  await api("/api/consent/withdraw", { method: "POST" }); await loadConsent(); loadOverview();
};

/* ---- audit ---- */
async function loadAudit() {
  const d = await api("/api/audit");
  const v = d.platform_audit; const banner = document.getElementById("verifyBanner");
  if (v && typeof v.valid !== "undefined")
    banner.innerHTML = v.valid
      ? `<div class="verify-banner ok">✓ Platform audit chain verified — ${esc(v.rows ?? "")} records intact.</div>`
      : `<div class="verify-banner bad">⚠ Platform audit chain broken at record ${esc(v.broken_at ?? "?")}.</div>`;
  else banner.innerHTML = "";
  const list = document.getElementById("auditList"); const rows = d.console_audit || [];
  if (!rows.length) { list.innerHTML = `<div class="empty">No console actions yet.</div>`; return; }
  list.innerHTML = rows.map(r => `<div class="audit-line">
    <span>${esc(fmtTime(r.ts))}</span><span class="audit-ev">${esc(r.event)}</span>
    <span>${esc(r.detail || "")}</span></div>`).join("");
}

/* ---- guided tour ---- */
const TOUR = [
  { el:"#nav-overview", title:"Overview", text:"Your home base. The overall security risk, live layer status and active alerts are all here." },
  { el:"#nav-layers", title:"Layers", text:"Turn each protection layer on or off. Zero-Trust is on by default; the platform core always runs." },
  { el:"#layerNav", title:"Layer dashboards", text:"Each enabled layer gets its own button here. Click one to open that layer's full dashboard inside this console." },
  { el:"#nav-alerts", title:"Alerts", text:"Every layer reports threats into this one timeline, so you watch one feed instead of four." },
  { el:"#nav-consent", title:"Consent", text:"Review the privacy notice and withdraw consent at any time. Nothing is monitored without it." },
  { el:"#nav-audit", title:"Audit log", text:"A tamper-evident record of every platform event and console action, with an integrity check." },
];
let tourIdx = 0;
function showTourStep(){
  const t = TOUR[tourIdx];
  const target = document.querySelector(t.el);
  const spot = document.getElementById("tourSpot");
  const card = document.getElementById("tourCard");
  document.getElementById("tourStep").textContent = `${tourIdx+1} / ${TOUR.length}`;
  document.getElementById("tourTitle").textContent = t.title;
  document.getElementById("tourText").textContent = t.text;
  document.getElementById("tourNext").textContent = (tourIdx === TOUR.length-1) ? "Done" : "Next";
  if (target){
    const r = target.getBoundingClientRect();
    spot.style.left = (r.left-6)+"px"; spot.style.top=(r.top-6)+"px";
    spot.style.width=(r.width+12)+"px"; spot.style.height=(r.height+12)+"px";
    let cx = r.right + 16, cy = r.top;
    if (cx + 300 > window.innerWidth) cx = Math.max(16, r.left - 316);
    if (cy + 180 > window.innerHeight) cy = window.innerHeight - 190;
    card.style.left = cx+"px"; card.style.top = Math.max(16,cy)+"px";
  }
}
function startTour(){ tourIdx=0; document.getElementById("tour").classList.add("show"); showTourStep(); }
function endTour(){ document.getElementById("tour").classList.remove("show");
  try{ localStorage.setItem("ns_tour_done","1"); }catch(_){}
  api("/api/tour-seen", { method:"POST" }).catch(()=>{}); }
document.getElementById("tourNext").onclick = () => { if (tourIdx>=TOUR.length-1) endTour(); else { tourIdx++; showTourStep(); } };
document.getElementById("tourSkip").onclick = endTour;
document.getElementById("tourBtn").onclick = startTour;
window.addEventListener("resize", () => { if (document.getElementById("tour").classList.contains("show")) showTourStep(); });

/* ---- theme toggle ---- */
document.getElementById("themeBtn").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  const next = cur === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("ns_theme", next); } catch(_){}
};

/* ---- quick actions ---- */
let paused = false;
document.getElementById("qaSelfCheck").onclick = async () => {
  const box = document.getElementById("selfcheckResult");
  box.innerHTML = `<div class="selfcheck"><div class="sc-row"><span></span><span class="muted">Running self-check…</span><span></span></div></div>`;
  try {
    const d = await api("/api/selfcheck");
    box.innerHTML = `<div class="selfcheck"><span class="sc-close" id="scClose">✕</span>` +
      d.checks.map(c => `<div class="sc-row"><span class="sc-dot ${c.ok?"ok":"bad"}"></span>
        <span>${esc(c.name)}</span><span class="sc-detail">${esc(c.detail)}</span></div>`).join("") + `</div>`;
    const x=document.getElementById("scClose"); if(x) x.onclick=()=>box.innerHTML="";
  } catch(_) { box.innerHTML=""; }
};
document.getElementById("qaExport").onclick = () => { window.location = "/api/audit/export"; };
document.getElementById("qaPause").onclick = async () => {
  if (!paused) {
    const ok = await confirmModal("Pause all monitoring?",
      `All layers stop watching until you resume. Consent stays on record; the console keeps running.
       <p style="margin-top:10px">Use this only when you deliberately want protection to stand down.</p>`, "Pause all");
    if (!ok) return;
    await api("/api/pause-all", { method:"POST" });
    paused = true; document.getElementById("qaPause").textContent = "Resume all";
    document.getElementById("qaPause").classList.remove("danger");
  } else {
    await api("/api/resume-all", { method:"POST" });
    paused = false; document.getElementById("qaPause").textContent = "Pause all";
    document.getElementById("qaPause").classList.add("danger");
  }
  setTimeout(loadOverview, 700);
};

/* ---- boot ---- */
(async function () {
  let me;
  try {
    me = await api("/api/me");
    CSRF = me.csrf;
    window.CONSENT_HTML = me.consent_html || "";
    document.getElementById("userName").textContent = me.username || "admin";
    applyTheme(document.documentElement.getAttribute("data-theme") || "light");
    document.getElementById("aboutVer").textContent =
      `${me.product || "NeuraShield"} ${me.version || ""} · Autonomous AI cyber-defence platform · R26-CS-013`;
  } catch (_) { return; }
  await loadOverview();
  pollHistory(true);        // last alerts are on screen at launch
  // First-run tour: the server tells us if this account has seen it (survives
  // cache clears and new browsers); localStorage is only a fast local hint.
  let seen = !!me.tour_seen;
  try { if (localStorage.getItem("ns_tour_done")==="1") seen = true; } catch(_){}
  if (!seen) {
    // wait until the nav has rendered so the spotlight lands correctly
    setTimeout(() => { if (document.querySelector("#nav-overview")) startTour(); }, 900);
  }
  setInterval(() => {
    const v = document.querySelector(".nav button.active")?.dataset.view;
    if (v === "overview" || v === "layers") loadOverview();
    if (v === "alerts") loadAlerts();
    if (v === "embed" && _embedCurrent) renderLayerDetail(_embedCurrent);
  }, 5000);
  setInterval(() => pollHistory(false), 5000);   // new alerts -> in-app notification + badge
})();


/* ---- platform key (for the browser extension) ---- */
document.addEventListener("click", async (e) => {
  if (e.target.id === "keyReveal") {
    const r = await api("/api/platform-key", { method: "POST" });
    document.getElementById("keyValue").textContent = r.key || "unavailable";
    document.getElementById("keyCopy").style.display = "";
    setTimeout(() => { const v = document.getElementById("keyValue"); if (v) v.textContent = "hidden"; }, 30000);
  } else if (e.target.id === "keyCopy") {
    try { await navigator.clipboard.writeText(document.getElementById("keyValue").textContent); e.target.textContent = "Copied"; } catch (_) {}
  }
});
