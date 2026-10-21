// ---------- Live pulse waveform ----------
// Idle: gentle steady heartbeat-style wave.
// Alert: sharp spike, red, faster.

const pulsePath = document.getElementById('pulse-path');
const pulseWrap = document.querySelector('.pulse-wrap');
let pulseOffset = 0;
let alertMode = false;

function idleWavePoint(x, t) {
  // calm sine wave with a small periodic "heartbeat" bump
  const base = Math.sin((x + t) * 0.05) * 3;
  const beatPos = ((x + t) % 200);
  const beat = beatPos > 95 && beatPos < 115 ? Math.sin((beatPos - 95) * (Math.PI / 20)) * 14 : 0;
  return 30 - base - beat;
}

function alertWavePoint(x, t) {
  const spikePos = ((x + t * 2.5) % 60);
  const spike = spikePos < 6 ? (spikePos < 3 ? spikePos * 12 : (6 - spikePos) * 12) : 0;
  const jitter = (Math.random() - 0.5) * 3;
  return 30 - spike - jitter;
}

function drawPulse() {
  let d = 'M0,30';
  for (let x = 0; x <= 600; x += 4) {
    const y = alertMode ? alertWavePoint(x, pulseOffset) : idleWavePoint(x, pulseOffset);
    d += ` L${x},${y.toFixed(1)}`;
  }
  pulsePath.setAttribute('d', d);
  pulseOffset += alertMode ? 3 : 1.2;
  requestAnimationFrame(drawPulse);
}
requestAnimationFrame(drawPulse);

// ---------- Safe helpers ----------
// Process names, file names and explanations come from the monitored machine
// and can be attacker-controlled, so EVERY value is escaped before it is put
// into the page.
function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function apiPost(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'NeuraShield' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

// ---------- Formatting helpers ----------

function timeAgo(unixSeconds) {
  const diff = Math.max(0, Date.now() / 1000 - unixSeconds);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function fmtDuration(totalSeconds) {
  if (totalSeconds == null) return '—';
  if (totalSeconds < 1) return `${Math.round(totalSeconds * 1000)}ms`;
  return `${totalSeconds.toFixed(2)}s`;
}

// ---------- Status polling ----------

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    const chip = document.getElementById('status-chip');
    const text = document.getElementById('status-text');
    const threat = !data.protected || data.state === 'incident';
    alertMode = threat;
    pulseWrap.classList.toggle('alert', alertMode);
    chip.classList.toggle('threat', alertMode);
    const labels = {
      armed: 'PROTECTED — MONITORING LIVE',
      baseline: `BACKING UP FILES${data.backup_percent == null ? '' : ' — ' + data.backup_percent + '%'}`,
      standby: 'STANDBY — NOT PROTECTING YET',
      offline: data.demo_mode ? 'PROTECTED — MONITORING LIVE (demo)' : 'SERVICE NOT RUNNING',
    };
    text.textContent = threat ? 'THREAT DETECTED' : (labels[data.state] || 'CHECKING…');

    document.getElementById('cpu-val').textContent = `${Number(data.cpu_percent).toFixed(1)}%`;
    document.getElementById('mem-val').textContent = `${Number(data.memory_mb).toFixed(0)} MB`;
    const uptimeMin = Math.floor(data.uptime_seconds / 60);
    document.getElementById('uptime-val').textContent = uptimeMin < 1
      ? `${Math.floor(data.uptime_seconds)}s`
      : `${uptimeMin}m`;

    const rail = document.getElementById('layer-list');
    rail.innerHTML = data.layers.map(l => `
      <div class="layer-row">
        <span class="indicator ${esc(l.status)}"></span>
        <div class="info">
          <span class="name">${esc(l.name)}</span>
          <span class="role">${esc(l.role)}</span>
        </div>
      </div>
    `).join('');
  } catch (e) {
    console.error('status poll failed', e);
  }
}

async function pollStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('stat-threats').textContent = data.total_threats_detected;
    document.getElementById('stat-files').textContent = data.total_files_restored;
    document.getElementById('stat-ttr').textContent = fmtDuration(data.avg_ttr_seconds);
  } catch (e) {
    console.error('stats poll failed', e);
  }
}

function renderIncident(a) {
  const threat = a.threat || {};
  const action = a.action_taken || {};
  const proc = a.process || {};
  const affected = a.affected_files || { paths: [], count: 0 };
  const sev = String(threat.severity || '');
  const sevClass = (sev === 'CRITICAL' || sev === 'HIGH') ? '' : 'medium';
  const paths = Array.isArray(affected.paths) ? affected.paths : [];
  const files = paths.slice(0, 3).map(esc).join(', ') +
    (affected.count > 3 ? ` +${esc(affected.count - 3)} more` : '');

  const actions = [];
  if (action.process_killed) {
    actions.push({ text: `Process killed (${action.kill_method || 'terminate'})`, warn: false });
  } else if (action.kill_skipped_reason) {
    actions.push({ text: `Kill skipped: ${action.kill_skipped_reason}`, warn: true });
  }
  if (action.files_restored) {
    actions.push({ text: `${action.files_restored} files restored in ${fmtDuration(action.restore_elapsed_seconds)}`, warn: false });
  }

  const ex = a.explanation;
  const explanationHtml = ex ? `
    <div class="explanation-box">
      <div class="explanation-label">
        <span class="explanation-glyph">◈</span> Why
        <span class="method-badge">${ex.method === 'shap_tree_explainer' ? 'SHAP' : 'rule-based'}</span>
      </div>
      <div class="explanation-text">${esc(ex.summary)}</div>
    </div>` : '';

  const verifiedBadge = a.verified
    ? `<span class="verify-badge verify-ok">✓ Signature verified</span>`
    : `<span class="verify-badge verify-fail">⚠ Unsigned / tampered</span>`;
  const acknowledgedBadge = a.acknowledged ? `<span class="ack-badge">✓ Cleared</span>` : '';

  // Manual approval is offered only when the system deliberately held back an
  // automated response (confidence below the kill threshold).
  const approveButton = (action.kill_skipped_reason && !a.acknowledged)
    ? `<button class="approve-btn" data-action="approve" data-alert-id="${esc(a.alert_id)}">Approve rollback command</button>` : '';
  const ackButton = !a.acknowledged
    ? `<button class="ack-btn" data-action="ack" data-alert-id="${esc(a.alert_id)}">Acknowledge</button>`
    : `<button class="ack-btn ack-btn-undo" data-action="unack" data-alert-id="${esc(a.alert_id)}">Undo</button>`;

  return `
    <div class="incident ${sevClass} ${a.acknowledged ? 'incident-acknowledged' : ''}">
      <div class="incident-card">
        <div class="incident-top">
          <div class="incident-title">
            <span class="severity-badge ${sevClass}">${esc(sev)}</span>
            ${esc(String(threat.type || '').replace(/_/g, ' '))}
            ${acknowledgedBadge}
          </div>
          <div class="incident-time">${timeAgo(a.timestamp)}</div>
        </div>
        <div class="incident-detail">
          <b>Confidence:</b> ${(Number(threat.confidence) * 100).toFixed(1)}% &nbsp;·&nbsp;
          <b>Process:</b> ${esc(proc.name || 'unknown')} (PID ${esc(proc.pid ?? '—')}) &nbsp;·&nbsp;
          <b>Files:</b> ${files || 'none'}
        </div>
        <div class="incident-detail incident-detail-gap">${verifiedBadge}</div>
        ${explanationHtml}
        <div class="action-row">
          ${actions.map(x => `<span class="action-tag${x.warn ? ' action-tag-warn' : ''}">${esc(x.text)}</span>`).join('')}
        </div>
        <div class="incident-footer">${approveButton}${ackButton}</div>
      </div>
    </div>`;
}

// Alert data kept by id so a button click can look up its file list
// without ever putting file names into HTML attributes.
let alertsById = {};

async function pollAlerts() {
  try {
    const showCleared = document.getElementById('show-cleared-checkbox')?.checked ?? false;
    const res = await fetch(`/api/alerts?show_acknowledged=${showCleared}`);
    const alerts = await res.json();
    const timeline = document.getElementById('timeline');
    const emptyState = document.getElementById('empty-state');

    alertsById = {};
    alerts.forEach(a => { alertsById[a.alert_id] = a; });

    if (alerts.length === 0) {
      if (emptyState) emptyState.style.display = 'block';
      timeline.querySelectorAll('.incident').forEach(el => el.remove());
      return;
    }
    if (emptyState) emptyState.style.display = 'none';
    timeline.innerHTML = alerts.map(renderIncident).join('');
  } catch (e) {
    console.error('alerts poll failed', e);
  }
}

async function approveRollback(alertId, buttonEl) {
  const alert = alertsById[alertId];
  if (!alert) return;
  buttonEl.disabled = true;
  buttonEl.textContent = 'Sending signed command...';
  try {
    const res = await apiPost('/api/approve_rollback', { files: alert.affected_files.paths });
    const data = await res.json();
    if (data.status === 'sent') {
      buttonEl.textContent = '✓ Command sent (verifying + executing)';
    } else {
      buttonEl.textContent = 'Failed to send command';
      buttonEl.disabled = false;
    }
  } catch (e) {
    console.error('approve_rollback failed', e);
    buttonEl.textContent = 'Failed to send command';
    buttonEl.disabled = false;
  }
}

async function ackAction(path, buttonEl) {
  buttonEl.disabled = true;
  try {
    await apiPost(path);
    await pollAlerts();
  } catch (e) {
    console.error('acknowledge failed', e);
    buttonEl.disabled = false;
  }
}

async function clearAllAlerts(buttonEl) {
  const originalText = buttonEl.textContent;
  buttonEl.disabled = true;
  buttonEl.textContent = 'Clearing…';
  try {
    const res = await apiPost('/api/alerts/clear_all');
    const data = await res.json();
    buttonEl.textContent = `Cleared ${data.count}`;
    await pollAlerts();
    reportsKey = '';
    pollReports();
    setTimeout(() => { buttonEl.textContent = originalText; buttonEl.disabled = false; }, 1500);
  } catch (e) {
    console.error('clear_all failed', e);
    buttonEl.textContent = originalText;
    buttonEl.disabled = false;
  }
}

// ---------- Protection panel ----------

function fmtClock(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleTimeString([], { hour12: false });
}

async function pollProtection() {
  try {
    const res = await fetch('/api/protection');
    const d = await res.json();
    const svc = d.service;
    const stateEl = document.getElementById('protect-state');
    stateEl.className = `protect-state ${d.state}`;
    const names = { armed: 'ARMED', baseline: 'BASELINE BACKUP', standby: 'STANDBY', incident: 'INCIDENT', offline: 'SERVICE OFFLINE' };
    stateEl.textContent = names[d.state] || d.state.toUpperCase();

    const msgs = {
      standby: 'Layer 2 is running but is NOT protecting anything: no file is being read, watched or copied. It needs your agreement first.',
      baseline: 'Making the first backup of your files. Protection is partial until this reaches 100%; later only changed files are copied.',
      armed: 'All watched files have a clean backup. Changes are watched live and only changed files are copied.',
      incident: 'A ransomware-like attack was detected. Layer 2 is responding: stopping the process and restoring files.',
      offline: 'The Layer 2 service is not running (or is still starting). Start it with:  python -m main --dashboard',
    };
    document.getElementById('protect-msg').textContent = msgs[d.state] || '';

    const bar = document.getElementById('backup-progress');
    bar.hidden = !(d.state === 'baseline' && d.percent != null);
    document.getElementById('backup-bar').style.width = `${d.percent || 0}%`;

    const b = svc ? svc.backup : null;
    document.getElementById('ps-files').textContent = b ? b.files_protected : '—';
    document.getElementById('ps-events').textContent = svc ? svc.monitor.events_last_60s : '—';
    document.getElementById('ps-incidents').textContent = svc ? svc.incidents : '—';
    document.getElementById('ps-backups').textContent = !b ? '—' : (b.paused ? 'paused' : (b.phase || '—'));

    // Consent area: shown until consent covers the folders shown here.
    const needConsent = !(d.consent && d.consent.covers_current_folders);
    document.getElementById('consent-area').hidden = !needConsent;
    document.getElementById('revoke-btn').hidden = needConsent;
    if (needConsent && !d.scope) {
      document.getElementById('scope-list').innerHTML = '';
      document.getElementById('scope-note').textContent = 'Counting files…';
    }
    if (needConsent && d.scope) {
      document.getElementById('scope-list').innerHTML = d.scope.folders.map(f =>
        `<li class="${f.exists ? '' : 'miss'}"><span class="mono">${esc(f.path)}</span>` +
        `<span>${f.exists ? esc(f.files) + ' files · ' + esc(f.mb) + ' MB' : 'folder not found'}</span></li>`).join('');
      document.getElementById('scope-note').textContent =
        `About ${d.scope.files}${d.scope.approximate ? '+' : ''} files (${d.scope.mb} MB). ` +
        `Backups are limited to ${d.max_backup_mb} MB in total; the oldest versions are removed first.`;
    }

    // Backup policy line: when the last copy was made, and how changed files are handled.
    const bl = document.getElementById('backup-line');
    if (b && ['armed', 'incident', 'baseline'].includes(d.state)) {
      bl.hidden = false;
      const last = b.last_backup_at ? timeAgo(b.last_backup_at) : 'none yet';
      bl.innerHTML =
        `Last backup <b>${esc(last)}</b> · changed files are copied after <b>${esc(b.settle_seconds ?? '—')} s</b> of quiet, ` +
        `at most once per <b>${esc(b.min_interval_seconds ?? '—')} s</b> each · waiting <b>${esc(b.pending ?? 0)}</b> · ` +
        `changes backed up <b>${esc(b.changes_backed_up ?? 0)}</b> · encrypted copies blocked <b>${esc(b.guard_blocks ?? 0)}</b>`;
    } else {
      bl.hidden = true;
    }

    // Detection meter: live value of each rule next to the level that raises an incident.
    const m = svc && svc.monitor ? svc.monitor.meter : null;
    const meterBox = document.getElementById('meter');
    meterBox.hidden = !(m && (d.state === 'armed' || d.state === 'incident' || d.state === 'baseline'));
    if (!meterBox.hidden) {
      const row = (label, v, need) => {
        const pct = Math.min(100, Math.round((Number(v) / Number(need)) * 100)) || 0;
        return `<div class="meter-row"><span>${esc(label)}</span><span class="mono">${esc(v)} / ${esc(need)}</span>` +
               `<div class="meter-track"><div class="meter-fill${pct >= 100 ? ' hot' : ''}" data-pct="${pct}"></div></div></div>`;
      };
      document.getElementById('meter-rows').innerHTML =
        row('File changes per second (5 s)', m.file_rate_5s, m.rate_threshold) +
        row('Average entropy (5 s)', m.avg_entropy_5s, m.entropy_threshold) +
        row('Encrypted-looking writes (5 s)', m.high_entropy_5s, m.high_entropy_5s_needed) +
        row('Encrypted-looking writes (60 s)', m.high_entropy_60s, m.high_entropy_60s_needed) +
        row('Files renamed with a new extension (10 s)', m.renamed_with_new_ext_10s, m.renamed_needed) +
        (m.ransom_notes_60s ? `<div class="meter-row"><span>Ransom-note-like file seen</span><span class="mono hot">${esc(m.ransom_notes_60s)}</span></div>` : '');
      // CSP forbids inline style attributes, so widths are set through the DOM.
      document.querySelectorAll('#meter-rows .meter-fill').forEach(el => { el.style.width = el.dataset.pct + '%'; });
      const mon = svc.monitor;
      document.getElementById('meter-health').textContent =
        mon.observer_alive ? `· file monitor running · ${mon.raw_events_seen} raw events` +
        (mon.observer_restarts ? ` · restarted ${mon.observer_restarts}x` : '') +
        (mon.handler_errors ? ` · ${mon.handler_errors} errors` : '')
        : '· FILE MONITOR NOT RUNNING';
    }

    const act = document.getElementById('activity');
    const rows = svc ? svc.monitor.recent : [];
    act.innerHTML = rows.length ? rows.map(r => `
      <div class="activity-row">
        <span>${esc(fmtClock(r.t))}</span><span>${esc(r.type)}</span>
        <span>${esc(r.name)}</span>
        <span class="${r.entropy != null && r.entropy >= 7.2 ? 'hot' : ''}">${r.entropy == null ? '' : 'H ' + esc(r.entropy)}</span>
      </div>`).join('') :
      `<div class="activity-empty">${svc ? 'No file activity yet.' : 'Waiting for the service…'}</div>`;
  } catch (e) {
    console.error('protection poll failed', e);
  }
}

document.getElementById('consent-check').addEventListener('change', ev => {
  document.getElementById('consent-btn').disabled = !ev.currentTarget.checked;
});
document.getElementById('consent-btn').addEventListener('click', async ev => {
  const btn = ev.currentTarget;
  btn.disabled = true;
  btn.textContent = 'Starting…';
  try { await apiPost('/api/protection/consent', { confirm: true }); } catch (e) { console.error(e); }
  btn.textContent = 'Start protecting';
  pollProtection();
});
document.getElementById('revoke-btn').addEventListener('click', async ev => {
  if (!window.confirm('Stop protecting? Watching and new backups stop. Existing backups are kept.')) return;
  try { await apiPost('/api/protection/revoke'); } catch (e) { console.error(e); }
  pollProtection();
});

// ---------- Incident reports ----------

const openReports = new Set();
const reportBodies = {};     // id -> rendered full report html
let reportsKey = '';

function restoredCell(f) {
  if (f.restored === true) return '<td class="ok">✓ restored</td>';
  if (f.restored === false) return `<td class="no">✗ ${esc(f.note || 'not restored')}</td>`;
  return '<td class="wait">waiting</td>';
}

function renderReportBody(r) {
  const files = r.files || [];
  const shown = files.slice(0, 200);
  const fileRows = shown.map(f =>
    `<tr><td class="mono">${esc(f.path)}</td><td>${esc(f.damage)}</td>${restoredCell(f)}</tr>`).join('');
  const more = files.length > shown.length ? `<div>+ ${esc(files.length - shown.length)} more file(s)</div>` : '';
  const left = (r.leftovers || []).map(l =>
    `<tr><td class="mono">${esc(l.path)}</td><td>${esc(l.kind)}</td></tr>`).join('');
  const unprot = (r.unprotected || []).slice(0, 50).map(p => `<tr><td class="mono">${esc(p)}</td></tr>`).join('');
  const tl = (r.timeline || []).map(e =>
    `<li><span class="t">${esc(fmtClock(e.t))}</span>${esc(e.text)}</li>`).join('');
  const proc = r.process || {};
  return `
    <h4>Attacker</h4>
    <div>${esc(proc.name || 'unidentified')} ${proc.pid != null ? '(PID ' + esc(proc.pid) + ')' : ''} —
      ${r.process_killed ? 'stopped' : 'not stopped: ' + esc(r.kill_note || 'could not be identified')}</div>
    <h4>Files damaged by the attack (${esc(files.length)})</h4>
    ${files.length ? `<table class="rtable"><tr><th>File</th><th>What happened</th><th>Result</th></tr>${fileRows}</table>${more}` : '<div>None.</div>'}
    ${left ? `<h4>Left behind by the attack</h4><table class="rtable"><tr><th>File</th><th>What it is</th></tr>${left}</table>
      <div>These are left in place for investigation; they were not deleted.</div>` : ''}
    ${unprot ? `<h4>Changed files with no backup (could not be restored)</h4><table class="rtable">${unprot}</table>` : ''}
    <h4>Timeline</h4><ul class="tl">${tl}</ul>`;
}

function renderReport(r) {
  const c = r.counts || {};
  const sevClass = (r.severity === 'CRITICAL' || r.severity === 'HIGH') ? '' : 'medium';
  const needsRestore = r.status !== 'responding' && ((c.failed || 0) + (c.waiting || 0)) > 0;
  const open = openReports.has(r.id) ? ' open' : '';
  const body = reportBodies[r.id] || '<div>Loading…</div>';
  return `
    <div class="report ${esc(r.status)} ${r.acknowledged ? 'report-ack' : ''}">
      <div class="report-top">
        <div class="report-headline"><span class="severity-badge ${sevClass}">${esc(r.severity)}</span> ${esc(r.headline)}</div>
        <div class="report-time">${esc(timeAgo(r.detected_at))}</div>
      </div>
      <div class="report-text">${esc(r.text)}</div>
      <div class="report-counts">
        <div class="rc bad"><span class="n">${esc(c.affected ?? 0)}</span><span class="l">files damaged</span></div>
        <div class="rc good"><span class="n">${esc(c.restored ?? 0)}</span><span class="l">restored</span></div>
        <div class="rc ${(c.failed || c.waiting) ? 'warn' : ''}"><span class="n">${esc((c.failed || 0) + (c.waiting || 0))}</span><span class="l">not restored</span></div>
        <div class="rc"><span class="n">${esc(c.leftovers ?? 0)}</span><span class="l">left behind</span></div>
      </div>
      <details data-id="${esc(r.id)}"${open}><summary>Full report</summary><div class="report-body">${body}</div></details>
      <div class="report-footer">
        ${needsRestore ? `<button class="restore-btn" data-action="restore-incident" data-id="${esc(r.id)}">Restore again</button>` : ''}
        ${r.acknowledged
          ? `<button class="ack-btn ack-btn-undo" data-action="report-unack" data-id="${esc(r.id)}">Undo</button>`
          : `<button class="ack-btn" data-action="report-ack" data-id="${esc(r.id)}">Acknowledge</button>`}
      </div>
    </div>`;
}

async function loadReportBody(id) {
  try {
    const res = await fetch(`/api/incidents/${encodeURIComponent(id)}`);
    if (!res.ok) return;
    reportBodies[id] = renderReportBody(await res.json());
  } catch (e) { console.error('report detail failed', e); }
}

async function pollReports() {
  try {
    const showCleared = document.getElementById('show-cleared-checkbox')?.checked ?? false;
    const res = await fetch(`/api/incidents?show_acknowledged=${showCleared}`);
    const rows = await res.json();
    await Promise.all([...openReports].map(loadReportBody));
    const key = JSON.stringify(rows) + '|' + [...openReports].join(',') +
      '|' + Object.entries(reportBodies).map(([k, v]) => k + v.length).join(',');
    if (key === reportsKey) return;
    reportsKey = key;
    const box = document.getElementById('reports');
    const empty = document.getElementById('reports-empty');
    box.querySelectorAll('.report').forEach(el => el.remove());
    empty.style.display = rows.length ? 'none' : 'block';
    box.insertAdjacentHTML('beforeend', rows.map(renderReport).join(''));
  } catch (e) {
    console.error('reports poll failed', e);
  }
}

document.getElementById('reports').addEventListener('toggle', ev => {
  const d = ev.target;
  if (!(d instanceof HTMLDetailsElement) || !d.dataset.id) return;
  if (d.open) { openReports.add(d.dataset.id); loadReportBody(d.dataset.id).then(() => { reportsKey = ''; pollReports(); }); }
  else openReports.delete(d.dataset.id);
}, true);

document.getElementById('reports').addEventListener('click', async ev => {
  const btn = ev.target.closest('button[data-action]');
  if (!btn) return;
  const id = btn.dataset.id;
  btn.disabled = true;
  try {
    if (btn.dataset.action === 'report-ack') await apiPost(`/api/alerts/${encodeURIComponent(id)}/acknowledge`);
    else if (btn.dataset.action === 'report-unack') await apiPost(`/api/alerts/${encodeURIComponent(id)}/unacknowledge`);
    else if (btn.dataset.action === 'restore-incident') {
      btn.textContent = 'Restoring…';
      const res = await apiPost(`/api/incidents/${encodeURIComponent(id)}/restore`);
      const data = await res.json();
      btn.textContent = data.status === 'ok' ? `Restored ${data.restored}, failed ${data.failed}` : (data.reason || 'Failed');
      await new Promise(r => setTimeout(r, 1500));
    }
  } catch (e) { console.error(e); }
  reportsKey = '';
  pollReports();
});

// ---------- Disk scan ----------

let scanTimer = null;

function renderScan(scan) {
  const box = document.getElementById('scan-result');
  const btn = document.getElementById('scan-btn');
  if (!scan) { box.textContent = ''; return; }
  if (scan.running) {
    btn.disabled = true; btn.textContent = 'Scanning…';
    box.textContent = 'Scanning the protected folders…';
    return;
  }
  btn.disabled = false; btn.textContent = 'Scan now';
  const r = scan.result;
  if (!r) { box.textContent = ''; return; }
  if (r.error) { box.innerHTML = `<div class="scan-verdict bad">${esc(r.error)}</div>`; return; }
  const cls = r.verdict === 'clean' ? 'clean' : (r.verdict === 'damage_found' ? 'bad' : 'warn');
  const msg = {
    clean: 'No sign of ransomware found. Every protected file matches its backup and no ransomware leftovers were seen.',
    damage_found: `Damage found: ${r.findings.filter(f => ['encrypted', 'format destroyed', 'missing', 'emptied'].includes(f.kind)).length} protected file(s) differ from their clean backup. ${r.restorable} can be restored.`,
    leftovers_found: 'Files that look like ransomware leftovers were found (no protected file is damaged).',
  }[r.verdict];
  const rows = r.findings.slice(0, 100).map(f => `
    <tr><td>${esc(f.kind)}</td><td class="mono">${esc(f.path)}</td><td>${esc(f.detail)}</td>
    <td class="${f.leftover ? '' : (f.restorable ? 'ok' : 'no')}">${f.leftover ? 'leftover (review, then delete)' : (f.restorable ? 'restorable' : 'no backup')}</td></tr>`).join('');
  const restoreBtn = r.restorable > 0
    ? `<button class="restore-btn" id="scan-restore-btn">Restore ${esc(r.restorable)} damaged file(s)</button>` : '';
  box.innerHTML = `
    <div class="scan-verdict ${cls}">${esc(msg)}
      <span class="mono"> (${esc(r.files_scanned)} files checked in ${esc(r.seconds)} s${r.truncated ? ', stopped early' : ''}, ${esc(fmtClock(r.finished_at))})</span></div>
    ${rows ? `<table class="rtable"><tr><th>Finding</th><th>File</th><th>Details</th><th></th></tr>${rows}</table>` : ''}
    ${r.findings.length > 100 ? `<div>+ ${esc(r.findings.length - 100)} more</div>` : ''}
    ${restoreBtn}`;
}

async function pollScan() {
  try {
    const res = await fetch('/api/scan');
    if (!res.ok) { renderScan(null); return false; }
    const scan = await res.json();
    renderScan(scan);
    return !!scan.running;
  } catch (e) { console.error('scan poll failed', e); return false; }
}

function watchScan() {
  clearInterval(scanTimer);
  scanTimer = setInterval(async () => { if (!(await pollScan())) clearInterval(scanTimer); }, 1000);
}

document.getElementById('scan-btn').addEventListener('click', async () => {
  const box = document.getElementById('scan-result');
  try {
    const res = await apiPost('/api/scan');
    const data = await res.json();
    if (!res.ok && data.status !== 'running') { box.innerHTML = `<div class="scan-verdict warn">${esc(data.reason || 'Could not start the scan')}</div>`; return; }
  } catch (e) { console.error(e); return; }
  await pollScan();
  watchScan();
});

document.getElementById('scan-result').addEventListener('click', async ev => {
  const btn = ev.target.closest('#scan-restore-btn');
  if (!btn) return;
  btn.disabled = true; btn.textContent = 'Restoring…';
  try {
    const res = await apiPost('/api/scan/restore');
    const data = await res.json();
    btn.textContent = `Restored ${data.restored}, failed ${data.failed}. Run the scan again to check.`;
  } catch (e) { console.error(e); btn.textContent = 'Failed'; }
});

// ---------- Event wiring (no inline handlers: the page's CSP forbids them) ----------

document.getElementById('timeline').addEventListener('click', ev => {
  const btn = ev.target.closest('button[data-action]');
  if (!btn) return;
  const id = btn.dataset.alertId;
  if (btn.dataset.action === 'approve') approveRollback(id, btn);
  else if (btn.dataset.action === 'ack') ackAction(`/api/alerts/${encodeURIComponent(id)}/acknowledge`, btn);
  else if (btn.dataset.action === 'unack') ackAction(`/api/alerts/${encodeURIComponent(id)}/unacknowledge`, btn);
});
document.getElementById('clear-all-btn').addEventListener('click', ev => clearAllAlerts(ev.currentTarget));
document.getElementById('show-cleared-checkbox').addEventListener('change', () => { reportsKey = ''; pollAlerts(); pollReports(); });

function pollAll() {
  pollStatus();
  pollStats();
  pollAlerts();
  pollReports();
  pollProtection();
}

pollAll();
pollScan();
setInterval(pollAll, 1500);
