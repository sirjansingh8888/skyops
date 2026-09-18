/* SkyOps console: replay / scenario / live map (MapLibre + deck.gl), mission briefs and UTM geofences,
   drone perception, land use, fleet, metrics, assistant. */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) { let d = ''; try { d = (await r.json()).detail || ''; } catch (e) { /* ignore */ } throw new Error(d || `${path} -> HTTP ${r.status}`); }
  return r.json();
};
const postJSON = (path, body, method = 'POST') => api(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const fmt = (n, d = 0) => (n === null || n === undefined || Number.isNaN(n) ? '–' : Number(n).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const SEV_RGB = { critical: [239, 68, 68], alert: [251, 146, 60], warning: [245, 158, 11], marginal: [100, 116, 139], info: [100, 116, 139] };
const LABEL_RGB = { pedestrian: [80, 220, 100], people: [80, 220, 100], bicycle: [250, 200, 60], car: [60, 160, 255], van: [60, 200, 255],
  truck: [255, 120, 60], tricycle: [200, 120, 255], 'awning-tricycle': [200, 120, 255], bus: [255, 80, 160], motor: [250, 230, 80] };
const STATUS_RGB = { approved: [34, 197, 94], 'approved-with-caution': [245, 158, 11], forced: [251, 146, 60], rejected: [239, 68, 68] };

const state = {
  t: 0, n: 20, gen: 0, playing: false, timer: null, snapshots: {}, tracks: null, summary: null, selected: null,
  conflictSet: new Set(), anomalySet: new Set(), intruderSet: new Set(), latest: null,
  live: { on: false, timer: null },
  mission: { site: null, result: null, picking: false },
  drone: { frames: [], idx: 0, result: null, playing: false, timer: null },
  landuse: { tiles: [], tile: null, mode: 'overlay', result: null },
  fleet: { subset: 'FD001', data: null, unit: null },
  metrics: null, assistant: { available: false, model: '' },
};

// ------------------------------------------------------------------ map
const STYLE_URL = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
const FALLBACK_STYLE = { version: 8, sources: {}, layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#0b0f17' } }] };
let map, overlay;

function planeIcon() {
  const c = document.createElement('canvas'); c.width = 64; c.height = 64;
  const g = c.getContext('2d'); g.fillStyle = '#fff';
  g.beginPath(); g.moveTo(32, 4); g.lineTo(44, 40); g.lineTo(32, 34); g.lineTo(20, 40); g.closePath(); g.fill();
  return c.toDataURL();
}
const ICON_URL = planeIcon();

async function loadStyle() {
  try {
    const r = await fetch(STYLE_URL, { signal: AbortSignal.timeout(7000) });
    if (!r.ok) throw new Error('style');
    return await r.json();
  } catch (e) { console.warn('basemap unavailable, using plain background', e); return FALLBACK_STYLE; }
}

async function initMap() {
  const style = await loadStyle();
  map = new maplibregl.Map({ container: 'map', style, center: [79.5, 20.5], zoom: 4.2, attributionControl: { compact: true } });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-left');
  overlay = new deck.MapboxOverlay({ interleaved: false, layers: [], getTooltip: tooltip, onClick: onMapClick });
  map.addControl(overlay);
  map.on('zoom', () => render());
}

function tooltip({ object, layer }) {
  if (!object) return null;
  const style = { background: '#0f1626', color: '#e5e7eb', fontSize: '12px', border: '1px solid #233047', borderRadius: '6px', padding: '6px 8px' };
  if (layer.id === 'aircraft') {
    const s = object;
    return { html: `<b>${esc(s.callsign)}</b> · ${esc(s.origin_country)}${s.simulated ? ' (scenario)' : ''}<br>${fmt(s.altitude_ft)} ft · ${fmt(s.speed_kt)} kt · ${fmt(s.vs_fpm)} fpm · hdg ${fmt(s.true_track)}°${s.squawk ? ` · squawk ${esc(s.squawk)}` : ''}${s.on_ground ? '<br>on ground' : ''}`, style };
  }
  if (layer.id === 'detections') return { text: `${object.label} ${fmt(object.conf, 2)} · ${fmt(object.range_m)} m from drone` };
  if (layer.id === 'geofences') return { html: `<b>${esc(object.name)}</b><br>${esc(object.status)} · r ${object.radius_km} km · ceiling ${object.ceiling_m} m<br>brief: ${object.brief.verdict} (${object.brief.score})`, style };
  if (layer.id === 'airport-labels' || layer.id.startsWith('airport-zones')) return { text: `${object.name} (${object.icao})` };
  return null;
}

function onMapClick(info) {
  if (state.mission.picking && info.coordinate) {
    $('#m-lon').value = info.coordinate[0].toFixed(4); $('#m-lat').value = info.coordinate[1].toFixed(4);
    state.mission.picking = false; $('#m-pick').classList.remove('active');
    setMissionSite(); return;
  }
  if (info.object && info.layer.id === 'aircraft') { selectAircraft(info.object.icao24); return; }
  if (!info.object) { state.selected = null; renderAircraftCard(); render(); }
}

function altColor(s) {
  if (s.on_ground) return [100, 116, 139, 220];
  const a = s.altitude_m;
  if (a < 1500) return [245, 158, 11, 255];
  if (a < 6000) { const t = (a - 1500) / 4500; return [Math.round(245 + (56 - 245) * t), Math.round(158 + (189 - 158) * t), Math.round(11 + (248 - 11) * t), 255]; }
  const t = Math.min(1, (a - 6000) / 6000); return [Math.round(56 + (186 - 56) * t), Math.round(189 + (230 - 189) * t), Math.round(248 + (253 - 248) * t), 255];
}
function aircraftColor(s) {
  if (s.icao24 === state.selected) return [255, 255, 255, 255];
  if (state.conflictSet.has(s.icao24) || state.intruderSet.has(s.icao24) || s.squawk === '7700' || s.squawk === '7600' || s.squawk === '7500') return [239, 68, 68, 255];
  if (state.anomalySet.has(s.icao24)) return [245, 158, 11, 255];
  if (s.simulated) return [192, 132, 252, 255];
  return altColor(s);
}

function currentSnap() { return state.latest; }

function buildLayers() {
  const snap = currentSnap();
  const layers = [];
  const airports = (state.summary && state.summary.airports) || [];
  layers.push(new deck.ScatterplotLayer({ id: 'airport-zones-12', data: airports, getPosition: (d) => [d.lon, d.lat], getRadius: 12000, radiusUnits: 'meters',
    stroked: true, filled: true, getFillColor: [167, 139, 250, 10], getLineColor: [167, 139, 250, 80], lineWidthMinPixels: 1, pickable: true }));
  layers.push(new deck.ScatterplotLayer({ id: 'airport-zones-5', data: airports, getPosition: (d) => [d.lon, d.lat], getRadius: 5000, radiusUnits: 'meters',
    stroked: true, filled: true, getFillColor: [239, 68, 68, 20], getLineColor: [239, 68, 68, 110], lineWidthMinPixels: 1 }));
  layers.push(new deck.TextLayer({ id: 'airport-labels', data: airports, getPosition: (d) => [d.lon, d.lat], getText: (d) => d.iata, getSize: 11,
    getColor: [196, 181, 253, 220], getPixelOffset: [0, -14], fontFamily: 'Inter, Segoe UI, sans-serif', pickable: true }));

  if (snap) {
    const fences = (snap.missions || []).filter((m) => m.status !== 'rejected');
    layers.push(new deck.ScatterplotLayer({ id: 'geofences', data: fences, getPosition: (m) => [m.lon, m.lat], getRadius: (m) => m.radius_km * 1000, radiusUnits: 'meters',
      stroked: true, filled: true, getFillColor: (m) => [...(STATUS_RGB[m.status] || [34, 197, 94]), 28], getLineColor: (m) => [...(STATUS_RGB[m.status] || [34, 197, 94]), 220],
      lineWidthMinPixels: 2, radiusMinPixels: 5, pickable: true }));
    layers.push(new deck.TextLayer({ id: 'geofence-labels', data: fences, getPosition: (m) => [m.lon, m.lat], getText: (m) => m.name, getSize: 11, getColor: [187, 247, 208, 235],
      getPixelOffset: [0, -16], fontFamily: 'Inter, Segoe UI, sans-serif' }));
    const tRel = snap.t_rel;
    if (state.tracks) {
      const trails = snap.states.map((s) => ({ path: (state.tracks[s.icao24] || []).filter((p) => p[3] <= tRel).slice(-8).map((p) => [p[0], p[1]]) })).filter((d) => d.path.length > 1);
      layers.push(new deck.PathLayer({ id: 'trails', data: trails, getPath: (d) => d.path, getColor: [148, 163, 184, 80], widthMinPixels: 1, widthMaxPixels: 2 }));
    }
    const hot = (s) => state.conflictSet.has(s.icao24) || state.intruderSet.has(s.icao24);
    const preds = snap.states.filter((s) => snap.predicted[s.icao24] && !s.on_ground).map((s) => ({
      path: [[s.longitude, s.latitude], ...snap.predicted[s.icao24].map((p) => [p[0], p[1]])], color: hot(s) ? [239, 68, 68, 170] : [56, 189, 248, 110] }));
    layers.push(new deck.PathLayer({ id: 'predicted', data: preds, getPath: (d) => d.path, getColor: (d) => d.color, widthMinPixels: 1.2, updateTriggers: { getColor: [snap.snapshot_time] } }));
    const real = snap.conflicts.filter((c) => c.severity !== 'marginal');
    layers.push(new deck.LineLayer({ id: 'conflict-lines', data: real, getSourcePosition: (c) => [c.a.lon, c.a.lat], getTargetPosition: (c) => [c.b.lon, c.b.lat],
      getColor: (c) => [...SEV_RGB[c.severity], 230], getWidth: 2, widthUnits: 'pixels' }));
    layers.push(new deck.ScatterplotLayer({ id: 'conflict-rings', data: real.flatMap((c) => [{ p: [c.a.lon, c.a.lat], s: c.severity }, { p: [c.b.lon, c.b.lat], s: c.severity }]),
      getPosition: (d) => d.p, getRadius: 9, radiusUnits: 'pixels', stroked: true, filled: false, getLineColor: (d) => [...SEV_RGB[d.s], 200], lineWidthMinPixels: 1.5 }));
    layers.push(new deck.ScatterplotLayer({ id: 'anomaly-rings', data: snap.anomalies.filter((a) => a.severity !== 'info'), getPosition: (a) => [a.lon, a.lat], getRadius: 13, radiusUnits: 'pixels',
      stroked: true, filled: false, getLineColor: (a) => [...SEV_RGB[a.severity], 220], lineWidthMinPixels: 2 }));
    layers.push(new deck.ScatterplotLayer({ id: 'utm-rings', data: snap.utm_alerts || [], getPosition: (a) => [a.lon, a.lat], getRadius: 16, radiusUnits: 'pixels',
      stroked: true, filled: false, getLineColor: (a) => [...SEV_RGB[a.severity], 240], lineWidthMinPixels: 2.5 }));
    layers.push(new deck.IconLayer({ id: 'aircraft', data: snap.states, iconAtlas: ICON_URL, iconMapping: { plane: { x: 0, y: 0, width: 64, height: 64, mask: true } },
      getIcon: () => 'plane', getPosition: (s) => [s.longitude, s.latitude], getSize: (s) => (s.icao24 === state.selected ? 30 : s.simulated ? 26 : 20), sizeUnits: 'pixels',
      getAngle: (s) => 360 - (s.true_track || 0), getColor: aircraftColor, pickable: true, transitions: { getPosition: 800, getAngle: 800 },
      updateTriggers: { getColor: [snap.snapshot_time, state.selected], getSize: [state.selected] } }));
    const zoom = map ? map.getZoom() : 4;
    const labelled = snap.states.filter((s) => zoom >= 6.5 || s.simulated || s.icao24 === state.selected || hot(s) || state.anomalySet.has(s.icao24));
    layers.push(new deck.TextLayer({ id: 'callsigns', data: labelled, getPosition: (s) => [s.longitude, s.latitude], getText: (s) => s.callsign, getSize: 11,
      getColor: [229, 231, 235, 235], getPixelOffset: [0, 17], fontFamily: 'JetBrains Mono, Consolas, monospace', updateTriggers: { getText: [snap.snapshot_time] } }));
  }

  const site = state.mission.site;
  if (site) {
    layers.push(new deck.ScatterplotLayer({ id: 'mission-radius', data: [site], getPosition: (d) => [d.lon, d.lat], getRadius: site.radius_km * 1000, radiusUnits: 'meters',
      stroked: true, filled: false, getLineColor: [148, 163, 184, 140], lineWidthMinPixels: 1 }));
    layers.push(new deck.ScatterplotLayer({ id: 'mission-site', data: [site], getPosition: (d) => [d.lon, d.lat], getRadius: 6, radiusUnits: 'pixels', getFillColor: [34, 197, 94, 255],
      stroked: true, getLineColor: [255, 255, 255, 200], lineWidthMinPixels: 1 }));
  }
  const dr = state.drone.result;
  if (dr) {
    layers.push(new deck.PolygonLayer({ id: 'drone-footprint', data: [{ poly: dr.footprint }], getPolygon: (d) => d.poly, filled: true, stroked: true,
      getFillColor: [34, 197, 94, 30], getLineColor: [34, 197, 94, 200], lineWidthMinPixels: 1 }));
    layers.push(new deck.ScatterplotLayer({ id: 'drone-pos', data: [dr.origin], getPosition: (d) => [d.lon, d.lat], getRadius: 7, radiusUnits: 'pixels', getFillColor: [187, 247, 208, 255],
      stroked: true, getLineColor: [22, 101, 52, 255], lineWidthMinPixels: 2 }));
    const dets = dr.detections.filter((d) => d.lat !== null && d.lat !== undefined);
    layers.push(new deck.ScatterplotLayer({ id: 'detections', data: dets, getPosition: (d) => [d.lon, d.lat], getRadius: 5, radiusUnits: 'pixels',
      getFillColor: (d) => [...(LABEL_RGB[d.label] || [200, 200, 200]), 240], pickable: true }));
  }
  return layers;
}

function render() { if (overlay) overlay.setProps({ layers: buildLayers() }); }

// ------------------------------------------------------------------ replay / scenario / live
async function loadSnapshot(i) {
  const gen = state.gen;
  if (!state.snapshots[i]) { const s = await api(`/api/airspace/snapshot/${i}`); if (gen === state.gen) state.snapshots[i] = s; else return s; }
  return state.snapshots[i];
}

function applySnapshot(snap) {
  state.latest = snap; state.t = snap.t_idx;
  state.conflictSet = new Set(snap.conflicts.filter((c) => c.severity !== 'marginal').flatMap((c) => [c.a.icao24, c.b.icao24]));
  state.anomalySet = new Set(snap.anomalies.filter((a) => a.severity !== 'info').map((a) => a.icao24));
  state.intruderSet = new Set((snap.utm_alerts || []).map((a) => a.icao24));
  $('#slider').value = snap.t_idx;
  $('#time-label').textContent = state.live.on ? `LIVE ${new Date(snap.snapshot_time * 1000).toISOString().slice(11, 19)} UTC · ${snap.n_snapshots} snapshots`
    : `T+${snap.t_rel} s · snapshot ${snap.t_idx + 1}/${snap.n_snapshots}`;
  const cs = snap.conflict_summary, as = snap.anomaly_summary, ua = snap.utm_alerts || [];
  const nConf = cs.critical + cs.alert + cs.warning, nAnom = as.critical + as.alert + as.warning;
  $('#chip-aircraft').textContent = `${snap.states.length} aircraft · ${snap.states.filter((s) => !s.on_ground).length} airborne`;
  const cc = $('#chip-conflicts'); cc.textContent = `${nConf} conflicts (${cs.critical} critical) · ${cs.marginal} marginal`; cc.className = 'chip ' + (cs.critical ? 'hot' : nConf ? 'warm' : 'ok');
  const ca = $('#chip-anomalies'); ca.textContent = `${nAnom} anomalies · ${as.info} info`; ca.className = 'chip ' + (as.critical ? 'hot' : nAnom ? 'warm' : 'ok');
  const cu = $('#chip-utm'); cu.textContent = `${ua.length} geofence alerts · ${(snap.missions || []).filter((m) => m.status !== 'rejected').length} active missions`;
  cu.className = 'chip ' + (ua.some((a) => a.severity === 'critical') ? 'hot' : ua.length ? 'warm' : 'ok');
  renderLists(snap); renderMissions(snap.missions || []); renderAircraftCard(); render();
}

async function showSnapshot(i) { applySnapshot(await loadSnapshot(i)); }

function renderLists(snap) {
  const uu = $('#utm-alerts'); uu.innerHTML = ''; const ua = snap.utm_alerts || [];
  $('#utm-count').textContent = ua.length;
  for (const a of ua) {
    const li = document.createElement('li'); li.className = a.severity;
    li.innerHTML = `<b>${esc(a.callsign)}</b> → <b>${esc(a.mission)}</b> · ${a.type === 'INTRUSION' ? 'inside geofence now' : `entry in ${a.t_entry_s} s`}<br><span class="muted">${esc(a.detail)}</span>`;
    li.onclick = () => selectAircraft(a.icao24, true);
    uu.appendChild(li);
  }
  if (!ua.length) uu.innerHTML = `<li class="info">${(snap.missions || []).length ? 'All registered geofences are clear.' : 'No missions registered. Use the Mission tab to register one.'}</li>`;
  const ul = $('#conflicts'); ul.innerHTML = '';
  $('#conf-count').textContent = snap.conflicts.length;
  for (const c of snap.conflicts) {
    const li = document.createElement('li'); li.className = c.severity;
    li.innerHTML = `<b>${esc(c.a.callsign)}</b> ↔ <b>${esc(c.b.callsign)}</b> · ${c.severity}${c.terminal ? ' (terminal)' : ''}<br><span class="muted">${c.t_loss_s === 0 ? 'separation lost now' : `loss in ${c.t_loss_s} s`} · CPA ${c.cpa_nm} NM · Δalt ${c.v_sep_ft} ft · ${c.converging ? 'converging' : 'diverging'}</span>`;
    li.onclick = () => { state.selected = c.a.icao24; map.flyTo({ center: [(c.a.lon + c.b.lon) / 2, (c.a.lat + c.b.lat) / 2], zoom: 8.5 }); renderAircraftCard(); render(); };
    ul.appendChild(li);
  }
  if (!snap.conflicts.length) ul.innerHTML = '<li class="info">No conflicts at this snapshot.</li>';
  const ulA = $('#anomalies'); ulA.innerHTML = '';
  $('#anom-count').textContent = snap.anomalies.length;
  for (const a of snap.anomalies) {
    const li = document.createElement('li'); li.className = a.severity;
    li.innerHTML = `<b>${esc(a.callsign)}</b> · ${esc(a.type.toLowerCase().replace(/_/g, ' '))}<br><span class="muted">${esc(a.detail)}</span>`;
    li.onclick = () => selectAircraft(a.icao24, true);
    ulA.appendChild(li);
  }
}

function selectAircraft(icao, fly = false) {
  state.selected = icao;
  const s = currentSnap() && currentSnap().states.find((x) => x.icao24 === icao);
  if (fly && s) map.flyTo({ center: [s.longitude, s.latitude], zoom: Math.max(map.getZoom(), 7) });
  renderAircraftCard(); render();
}

function renderAircraftCard() {
  const el = $('#ac-card'); const snap = currentSnap();
  const s = snap && snap.states.find((x) => x.icao24 === state.selected);
  if (!s) { el.innerHTML = '<div class="muted">Click an aircraft on the map to inspect it.</div>'; return; }
  const pred = (snap.predicted[s.icao24] || []).map((p) => `+${p[3]}s → ${p[1].toFixed(3)}, ${p[0].toFixed(3)} @ ${fmt(p[2] / 0.3048)} ft`).join('<br>');
  const flags = [...snap.conflicts.filter((c) => c.a.icao24 === s.icao24 || c.b.icao24 === s.icao24).map((c) => `conflict with ${c.a.icao24 === s.icao24 ? c.b.callsign : c.a.callsign} (${c.severity})`),
    ...snap.anomalies.filter((a) => a.icao24 === s.icao24).map((a) => `${a.type.toLowerCase().replace(/_/g, ' ')}: ${a.detail}`),
    ...(snap.utm_alerts || []).filter((a) => a.icao24 === s.icao24).map((a) => a.detail)];
  el.innerHTML = `<div class="card-title">${esc(s.callsign)} <span class="muted">${esc(s.icao24)} · ${esc(s.origin_country)}</span>${s.simulated ? '<span class="badge sim">simulated</span>' : ''}</div>
    <div class="kv"><div><span>altitude</span>${fmt(s.altitude_ft)} ft</div><div><span>ground speed</span>${fmt(s.speed_kt)} kt</div><div><span>vertical</span>${fmt(s.vs_fpm)} fpm</div>
    <div><span>heading</span>${fmt(s.true_track)}°</div><div><span>squawk</span>${esc(s.squawk || '–')}</div><div><span>nearest airport</span>${fmt(s.airport_km)} km</div></div>
    ${flags.length ? `<ul class="reasons">${flags.map((f) => `<li class="warning">${esc(f)}</li>`).join('')}</ul>` : ''}
    <div class="hint">Dead-reckoning prediction<br>${pred || '–'}</div>
    <div class="row"><button class="link" id="ac-ask">Ask the assistant about ${esc(s.callsign)}</button><button class="link" id="ac-site">Assess a drone launch below it</button></div>`;
  $('#ac-ask').onclick = () => { switchTab('assistant'); $('#chat-input').value = `Tell me about ${s.callsign}: is it behaving normally and is it involved in any conflict?`; };
  $('#ac-site').onclick = () => { $('#m-lat').value = s.latitude.toFixed(4); $('#m-lon').value = s.longitude.toFixed(4); switchTab('mission'); setMissionSite(); assessMission(); };
}

function play(on) {
  if (state.live.on) on = false;
  state.playing = on; $('#btn-play').textContent = on ? '⏸ Pause' : '▶ Play';
  clearInterval(state.timer);
  if (on) state.timer = setInterval(() => showSnapshot((state.t + 1) % state.n), 1500);
}

async function resetReplay(keepT = true) {
  state.gen += 1; state.snapshots = {}; state.tracks = null;
  state.summary = await api('/api/airspace/summary');
  state.n = state.summary.n_snapshots; $('#slider').max = Math.max(0, state.n - 1);
  const notes = state.summary.scenario_notes || [];
  $('#scenario-card').classList.toggle('hidden', !notes.length);
  $('#scenario-notes').innerHTML = notes.map((n) => `<li class="warning">${esc(n)}</li>`).join('');
  await showSnapshot(state.live.on ? -1 : Math.min(keepT ? state.t : 0, state.n - 1));
  api('/api/airspace/tracks').then((d) => { state.tracks = d.tracks; render(); });
  if (!state.live.on) { const gen = state.gen; (async () => { for (let i = 0; i < state.n && gen === state.gen; i++) await loadSnapshot(i); })(); }
}

async function initScenarios() {
  const d = await api('/api/scenarios');
  $('#scenario').innerHTML = d.scenarios.map((s) => `<option value="${s.name}" title="${esc(s.description)}">${s.name === 'baseline' ? 'Recorded traffic' : 'Scenario: ' + s.name}</option>`).join('');
  $('#scenario').value = d.scenarios.some((s) => s.name === d.active) ? d.active : 'baseline';
}

async function setScenario(name) {
  if (state.live.on) await setLive(false, true);
  $('#scenario').value = name;
  await postJSON('/api/scenario', { name });
  await resetReplay(true);
}

async function setLive(on, silent = false) {
  const btn = $('#btn-live');
  clearInterval(state.live.timer);
  if (on) {
    btn.textContent = '● connecting…';
    try { await postJSON('/api/live/start', {}); } catch (e) { btn.textContent = '● LIVE'; alert('Live feed unavailable: ' + e.message + '\nThe recorded replay keeps working.'); return; }
    play(false); state.live.on = true; btn.classList.add('live'); btn.textContent = '■ LIVE'; $('#slider').disabled = true; $('#scenario').value = 'baseline';
    await resetReplay(false);
    state.live.timer = setInterval(async () => { state.gen += 1; state.snapshots = {}; try { applySnapshot(await api('/api/airspace/snapshot/-1')); api('/api/airspace/tracks').then((d) => { state.tracks = d.tracks; render(); }); } catch (e) { console.warn(e); } }, 10000);
  } else {
    state.live.on = false; btn.classList.remove('live'); btn.textContent = '● LIVE'; $('#slider').disabled = false;
    try { await postJSON('/api/live/stop', {}); } catch (e) { /* ignore */ }
    if (!silent) await resetReplay(false);
  }
}

// ------------------------------------------------------------------ mission + UTM desk
function setMissionSite() {
  const lat = parseFloat($('#m-lat').value), lon = parseFloat($('#m-lon').value), radius_km = parseFloat($('#m-radius').value) || 10;
  if (Number.isNaN(lat) || Number.isNaN(lon)) return;
  state.mission.site = { lat, lon, radius_km, alt_m: parseFloat($('#m-alt').value) || 100 };
  render();
}

async function assessMission() {
  setMissionSite();
  const site = state.mission.site; if (!site) return;
  const body = { lat: site.lat, lon: site.lon, alt_m: site.alt_m, radius_km: site.radius_km, t_idx: state.live.on ? -1 : state.t, use_gt: $('#d-gt').checked };
  if ($('#m-attach-frame').checked && state.drone.frames.length) body.frame = state.drone.frames[state.drone.idx].name;
  if ($('#m-attach-tile').checked && state.landuse.tile) body.tile = state.landuse.tile;
  const el = $('#mission-result'); el.innerHTML = '<div class="muted">Assessing…</div>';
  try {
    const r = await postJSON('/api/mission/brief', body);
    state.mission.result = r;
    const ac = r.traffic.aircraft.map((a) => `<tr><td>${esc(a.callsign)}</td><td>${fmt(a.altitude_ft)} ft</td><td>${fmt(a.speed_kt)} kt</td><td>${fmt(a.dist_km, 1)} km</td></tr>`).join('');
    el.innerHTML = `<div class="row"><span class="verdict ${r.verdict}">${r.verdict}</span><span class="muted">risk ${r.score}/100 · ${r.zone} zone · ${esc(r.airport.name)} ${r.airport.distance_km} km</span></div>
      <div class="score"><i style="width:${r.score}%"></i></div>
      <ul class="reasons">${r.reasons.map((x) => `<li class="${x.level}">${esc(x.text)}</li>`).join('')}</ul>
      ${r.perception ? `<div class="hint">Camera (${esc(r.perception.source)}): ${r.perception.summary.total} objects, landing ${r.perception.landing_zone.verdict} ${r.perception.landing_zone.score}</div>` : ''}
      ${r.landuse ? `<div class="hint">Land use: ${r.landuse.verdict}, suitability ${r.landuse.score}, hazard ${(r.landuse.hazard_fraction * 100).toFixed(0)}%</div>` : ''}
      ${ac ? `<div class="table-wrap"><table><thead><tr><th>Nearby</th><th>Alt</th><th>Speed</th><th>Dist</th></tr></thead><tbody>${ac}</tbody></table></div>` : ''}`;
    map.flyTo({ center: [site.lon, site.lat], zoom: Math.max(map.getZoom(), 8) });
    if (state.drone.result && $('#d-relocate').checked) loadDroneFrame();
  } catch (e) { el.innerHTML = `<div class="msg err">${esc(e.message)}</div>`; }
}

async function refreshAfterMissionChange() { state.gen += 1; state.snapshots = {}; await showSnapshot(state.live.on ? -1 : state.t); if (!state.live.on) { const gen = state.gen; (async () => { for (let i = 0; i < state.n && gen === state.gen; i++) await loadSnapshot(i); })(); } }

async function registerMission() {
  setMissionSite(); const site = state.mission.site; if (!site) return;
  try {
    const m = await postJSON('/api/utm/missions', { name: $('#u-name').value.trim(), lat: site.lat, lon: site.lon, radius_km: parseFloat($('#u-radius').value) || 1.5,
      ceiling_m: site.alt_m, t_idx: state.live.on ? -1 : state.t, force: $('#u-force').checked });
    $('#u-name').value = '';
    if (m.status === 'rejected') alert(`Registration rejected: brief is ${m.brief.verdict} (${m.brief.score}/100).\n${m.brief.reasons.map((r) => '• ' + r.text).join('\n')}\nTick "force" to override.`);
    await refreshAfterMissionChange();
  } catch (e) { alert(e.message); }
}

function renderMissions(missions) {
  $('#missions-count').textContent = missions.length;
  const ul = $('#missions'); ul.innerHTML = '';
  for (const m of missions) {
    const li = document.createElement('li'); li.className = m.status === 'rejected' ? 'critical' : m.status === 'approved' ? 'ok' : 'warning';
    li.innerHTML = `<b>${esc(m.name)}</b> · <span class="state ${m.status === 'approved' ? 'healthy' : m.status === 'rejected' ? 'ground' : 'watch'}">${esc(m.status)}</span> <button class="link del" data-id="${m.id}">remove</button><br>
      <span class="muted">${m.lat.toFixed(4)}, ${m.lon.toFixed(4)} · r ${m.radius_km} km · ceiling ${m.ceiling_m} m · brief ${m.brief.verdict} ${m.brief.score}/100 (${m.brief.zone} zone)</span>`;
    li.onclick = (ev) => { if (ev.target.classList.contains('del')) return; map.flyTo({ center: [m.lon, m.lat], zoom: 10 }); };
    ul.appendChild(li);
  }
  $$('#missions .del').forEach((b) => (b.onclick = async () => { await api(`/api/utm/missions/${b.dataset.id}`, { method: 'DELETE' }); await refreshAfterMissionChange(); }));
}

// ------------------------------------------------------------------ drone
async function initDrone() {
  try {
    const d = await api('/api/drone/frames');
    state.drone.frames = d.frames; state.drone.idx = Math.floor(d.frames.length / 2);
    $('#d-slider').max = Math.max(0, d.frames.length - 1); $('#d-slider').value = state.drone.idx;
    await loadDroneFrame();
  } catch (e) { $('#drone-info').innerHTML = `<div class="msg err">No drone frames on disk. Run scripts/download_data.py --only auair</div>`; }
}

async function loadDroneFrame() {
  const f = state.drone.frames[state.drone.idx]; if (!f) return;
  const gt = $('#d-gt').checked, tilt = $('#d-tilt').value, hfov = $('#d-hfov').value;
  const site = $('#d-relocate').checked && state.mission.site;
  const q = new URLSearchParams({ gt, tilt, hfov }); if (site) { q.set('origin_lat', site.lat); q.set('origin_lon', site.lon); }
  $('#d-index').textContent = `${state.drone.idx + 1}/${state.drone.frames.length}`;
  $('#drone-img').src = `/api/drone/frame/${f.name}/image?gt=${gt}`;
  try {
    const r = await api(`/api/drone/frame/${f.name}?${q}`);
    state.drone.result = r;
    const t = r.telemetry;
    $('#drone-source').textContent = `${r.source} · ${r.summary.total} objects (dataset has ${r.gt_count})`;
    $('#drone-info').innerHTML = `<div><span>altitude AGL</span>${fmt(t.alt_m, 1)} m</div><div><span>heading</span>${fmt(t.yaw_deg)}°</div><div><span>speed</span>${fmt(t.speed_ms, 2)} m/s</div>
      <div><span>pitch / roll</span>${fmt(t.pitch_deg, 1)}° / ${fmt(t.roll_deg, 1)}°</div><div><span>GPS</span>${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}</div><div><span>people / vehicles</span>${r.summary.people} / ${r.summary.vehicles}</div>`;
    const lz = r.landing_zone;
    $('#landing-card').innerHTML = `<div class="row"><span class="verdict ${lz.verdict}">${lz.verdict}</span><span class="muted">landing score ${lz.score}/100 · ${(lz.free_fraction * 100).toFixed(0)}% of the view clear · ${lz.people_present ? 'people present' : 'no people'}</span></div>
      <div class="score"><i style="width:${lz.score}%"></i></div>
      <div class="hint">Recommended touchdown cell: column ${lz.target.col + 1}, row ${lz.target.row + 1} (clearance ${lz.target.clearance_cells} cells). Objects are geo-located from the drone's GPS, altitude and heading with the camera model (tilt ${r.camera.tilt_deg}°, FOV ${r.camera.hfov_deg}°)${r.origin.relocated ? ', relocated to the mission site' : ''}.</div>
      ${r.detections.length ? `<div class="tools">${r.detections.slice(0, 12).map((d) => `<span>${esc(d.label)} ${fmt(d.conf, 2)}${d.range_m !== null ? ` · ${fmt(d.range_m)} m` : ''}</span>`).join('')}</div>` : ''}`;
    render();
  } catch (e) { $('#landing-card').innerHTML = `<div class="msg err">${esc(e.message)}</div>`; }
}

function playDrone(on) {
  state.drone.playing = on; $('#d-play').textContent = on ? '⏸ Pause' : '▶ Play flight';
  clearInterval(state.drone.timer);
  if (on) state.drone.timer = setInterval(() => { state.drone.idx = (state.drone.idx + 2) % state.drone.frames.length; $('#d-slider').value = state.drone.idx; loadDroneFrame(); }, $('#d-gt').checked ? 500 : 1500);
}

// ------------------------------------------------------------------ land use
async function initLanduse() {
  try {
    const d = await api('/api/landuse/tiles'); state.landuse.tiles = d.tiles;
    const sel = $('#lu-tile'); sel.innerHTML = d.tiles.map((t) => `<option value="${t}">${t}</option>`).join('');
    state.landuse.tile = d.tiles[0]; await loadTile();
  } catch (e) { $('#lu-assess').innerHTML = '<div class="msg err">No Dubai tiles on disk. Run scripts/download_data.py --only dubai</div>'; }
}

async function loadTile() {
  const t = state.landuse.tile; if (!t) return;
  $('#lu-img').src = `/api/landuse/tile/${t}/image?mode=${state.landuse.mode}`;
  const r = await api(`/api/landuse/tile/${t}`); state.landuse.result = r;
  $('#lu-model').textContent = `${r.model}${r.pixel_accuracy_vs_gt !== undefined ? ` · ${(r.pixel_accuracy_vs_gt * 100).toFixed(0)}% pixel accuracy vs truth` : ''}`;
  $('#lu-bars').innerHTML = Object.entries(r.fractions).map(([c, f]) => `<div><span>${c}</span><i style="width:${(f * 100).toFixed(1)}%;background:${r.legend[c]}"></i><span class="mono">${(f * 100).toFixed(1)}%</span></div>`).join('');
  $('#lu-legend').innerHTML = Object.entries(r.legend).map(([c, col]) => `<span><i style="background:${col}"></i>${c}</span>`).join('');
  const a = r.assessment;
  $('#lu-assess').innerHTML = `<div class="row"><span class="verdict ${a.verdict}">${a.verdict}</span><span class="muted">landing suitability ${a.score}/100 · hazards (buildings + water) ${(a.hazard_fraction * 100).toFixed(0)}%</span></div>
    <div class="score"><i style="width:${a.score}%"></i></div>
    <div class="hint">Best touchdown cell: row ${a.best_cell.row + 1}, column ${a.best_cell.col + 1} (suitability ${(a.best_cell.suitability * 100).toFixed(0)}%). Surface weights: unpaved land 1.0, vegetation 0.6, road 0.35, building/water 0.</div>`;
}

// ------------------------------------------------------------------ fleet
async function loadFleet() {
  const s = state.fleet.subset;
  $('#fleet-table tbody').innerHTML = '<tr><td colspan="6" class="muted">Loading fleet model…</td></tr>';
  const r = await api(`/api/fleet/${s}`); state.fleet.data = r;
  $('#fleet-model').textContent = `${r.model} · RMSE ${r.rmse_vs_truth} cycles vs truth`;
  $('#fleet-counts').innerHTML = Object.entries(r.counts).map(([k, v]) => `<span class="state ${k}">${k} ${v}</span>`).join('');
  $('#fleet-table tbody').innerHTML = r.engines.map((e) => `<tr class="clickable" data-unit="${e.unit}"><td>${e.engine_id}</td><td>${e.cycles_observed}</td><td><b>${fmt(e.rul_p50)}</b></td><td class="muted">${fmt(e.rul_p10)}–${fmt(e.rul_p90)}</td><td class="muted">${fmt(e.rul_true)}</td><td><span class="state ${e.state}">${e.state}</span></td></tr>`).join('');
  $$('#fleet-table tbody tr').forEach((tr) => { tr.onclick = () => loadEngine(parseInt(tr.dataset.unit)); });
  if (r.engines.length) loadEngine(r.engines[0].unit);
}

async function loadEngine(unit) {
  state.fleet.unit = unit;
  $$('#fleet-table tbody tr').forEach((tr) => tr.classList.toggle('selected', parseInt(tr.dataset.unit) === unit));
  const e = state.fleet.data.engines.find((x) => x.unit === unit);
  const h = await api(`/api/fleet/${state.fleet.subset}/${unit}`);
  const el = $('#engine-card');
  el.innerHTML = `<div class="card-title">${e.engine_id} <span class="state ${e.state}">${e.state}</span></div>
    <div class="muted small">${esc(e.action)}. Observed ${e.cycles_observed} cycles; predicted RUL ${e.rul_p50} (${e.rul_p10}–${e.rul_p90}), true ${e.rul_true}.</div>
    <div class="hint">Trending sensors: ${e.trending.map((t) => `${esc(t.name)} (${t.slope > 0 ? '+' : ''}${t.slope})`).join(', ')}</div>
    <div id="chart-rul"></div><div id="chart-sensors"></div>`;
  svgChart($('#chart-rul'), { title: 'Remaining useful life (cycles), 80% band', x: h.cycles, series: [{ y: h.rul.p50, color: '#38bdf8', name: 'RUL p50' }], band: { lo: h.rul.p10, hi: h.rul.p90, color: 'rgba(56,189,248,0.18)' } });
  const keys = ['s4', 's11', 's12', 's15'];
  svgChart($('#chart-sensors'), { title: 'Normalised sensors', x: h.cycles, series: keys.map((k, i) => ({ y: h.sensors[k], color: ['#f59e0b', '#a78bfa', '#22c55e', '#f472b6'][i], name: h.sensor_names[k] })) });
}

function svgChart(el, { title, x, series, band, w = 410, h = 190 }) {
  const pad = { l: 36, r: 8, t: 22, b: 20 };
  const all = series.flatMap((s) => s.y).concat(band ? band.lo.concat(band.hi) : []);
  const xmin = Math.min(...x), xmax = Math.max(...x), ymin = Math.min(...all), ymax = Math.max(...all);
  const sx = (v) => pad.l + ((v - xmin) / Math.max(1e-9, xmax - xmin)) * (w - pad.l - pad.r);
  const sy = (v) => h - pad.b - ((v - ymin) / Math.max(1e-9, ymax - ymin)) * (h - pad.t - pad.b);
  const path = (ys) => ys.map((v, i) => `${i ? 'L' : 'M'}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
  let svg = `<svg class="chart" viewBox="0 0 ${w} ${h}"><text x="${pad.l}" y="14">${esc(title)}</text>`;
  if (band) svg += `<path d="${path(band.hi)} ${band.lo.map((v, i) => `L${sx(x[band.lo.length - 1 - i]).toFixed(1)},${sy(band.lo[band.lo.length - 1 - i]).toFixed(1)}`).join(' ')} Z" fill="${band.color}"/>`;
  series.forEach((s) => { svg += `<path d="${path(s.y)}" fill="none" stroke="${s.color}" stroke-width="1.6"/>`; });
  svg += `<text x="4" y="${sy(ymax) + 4}">${ymax.toFixed(1)}</text><text x="4" y="${sy(ymin)}">${ymin.toFixed(1)}</text><text x="${sx(xmin)}" y="${h - 6}">cycle ${xmin}</text><text x="${sx(xmax) - 50}" y="${h - 6}">cycle ${xmax}</text>`;
  svg += series.map((s, i) => `<text x="${w - pad.r - 130}" y="${14 + i * 11}" style="fill:${s.color}">${esc(s.name).slice(0, 26)}</text>`).join('');
  el.innerHTML = svg + '</svg>';
}

// ------------------------------------------------------------------ metrics (model card)
async function loadMetrics() {
  const el = $('#metrics-body');
  try {
    const m = await api('/api/metrics'); state.metrics = m;
    const th = m.trajectory.horizons;
    const trajRows = Object.entries(th).map(([h, v]) => `<tr><td>+${h} s</td><td>${fmt(v.median_m)} m</td><td>${fmt(v.mean_m)} m</td><td>${fmt(v.p90_m)} m</td><td class="muted">${v.n}</td></tr>`).join('');
    const rp = m.replay;
    let fleet = `<div class="muted small">${esc(m.fleet.model)}${m.fleet.note ? ' · ' + esc(m.fleet.note) : ''}</div>`;
    if (m.fleet.test) {
      fleet += `<table><thead><tr><th>Subset</th><th>RMSE</th><th>MAE</th><th>NASA score</th><th>80% band coverage</th><th>Engines</th></tr></thead><tbody>${Object.entries(m.fleet.test).map(([k, v]) =>
        `<tr><td>${k}</td><td><b>${fmt(v.rmse, 1)}</b></td><td>${fmt(v.mae, 1)}</td><td>${fmt(v.nasa_score)}</td><td>${fmt(v.interval_coverage * 100)}% (±${fmt(v.interval_width / 2, 0)})</td><td class="muted">${v.n}</td></tr>`).join('')}</tbody></table>
        <div class="hint">Training-free k-NN baseline: RMSE ${m.fleet.baseline_knn_rmse_fd001} on FD001. Top features: ${m.fleet.top_features.map((f) => esc(f.feature)).join(', ')}. Trained in ${m.fleet.trained_seconds} s on CPU.</div>`;
    }
    const p = m.perception, lu = m.landuse;
    el.innerHTML = `<div class="card"><div class="card-title">Trajectory prediction <span class="muted">${esc(m.trajectory.predictor)}</span></div>
        <table><thead><tr><th>Horizon</th><th>Median</th><th>Mean</th><th>p90</th><th>n</th></tr></thead><tbody>${trajRows}</tbody></table><div class="hint">${esc(m.trajectory.note)}.</div></div>
      <div class="card"><div class="card-title">Replay statistics</div><div class="kv"><div><span>snapshots</span>${rp.snapshots}</div><div><span>distinct conflict pairs</span>${rp.distinct_conflict_pairs}</div><div><span>anomaly flags</span>${rp.anomaly_flags}</div>
        <div><span>critical</span>${rp.conflict_flags.critical}</div><div><span>alert / warning</span>${rp.conflict_flags.alert} / ${rp.conflict_flags.warning}</div><div><span>marginal (RVSM noise)</span>${rp.conflict_flags.marginal}</div></div></div>
      <div class="card"><div class="card-title">Fleet RUL</div>${fleet}</div>
      <div class="card"><div class="card-title">Drone perception</div><div class="muted small">${esc(p.model)}</div>${p.map50 !== undefined ? `<div class="kv"><div><span>mAP@50</span>${fmt(p.map50, 3)}</div><div><span>mAP@50-95</span>${fmt(p.map50_95, 3)}</div><div><span>epochs</span>${p.epochs}</div></div>` : '<div class="hint">Fine-tune on VisDrone to replace the placeholder (GPU run pending).</div>'}</div>
      <div class="card"><div class="card-title">Land use</div><div class="muted small">${esc(lu.model)}</div>${lu.best_miou !== undefined ? `<div class="kv"><div><span>mIoU</span>${fmt(lu.best_miou, 3)}</div><div><span>epochs</span>${lu.epochs}</div></div>` : `<div class="hint">Placeholder pixel accuracy ${lu.pixel_accuracy !== null ? fmt(lu.pixel_accuracy * 100) + '%' : '–'}; U-Net training pending (GPU).</div>`}</div>`;
    const h60 = th['60'], h120 = th['120'];
    if (h60) $('#eval-line').textContent = `${m.trajectory.predictor}: median error ${fmt(h60.median_m)} m @60 s · ${fmt(h120.median_m)} m @120 s (n=${h60.n})`;
  } catch (e) { el.innerHTML = `<div class="card"><div class="msg err">${esc(e.message)}</div></div>`; }
}

// ------------------------------------------------------------------ assistant
async function initAssistant() {
  try {
    const s = await api('/api/assistant/status'); state.assistant = s;
    $('#chip-assistant').textContent = s.available ? `assistant: ${s.model}` : 'assistant: offline (no API key)';
    $('#chip-assistant').className = 'chip ' + (s.available ? 'ok' : '');
    $('#assist-model').textContent = s.available ? s.model : 'offline: set ANTHROPIC_API_KEY in .env';
  } catch (e) { /* ignore */ }
}

function addMsg(kind, html) { const d = document.createElement('div'); d.className = `msg ${kind}`; d.innerHTML = html; $('#chat').appendChild(d); $('#chat').scrollTop = 1e9; return d; }

async function sendChat(text) {
  if (!text.trim()) return;
  addMsg('user', esc(text)); $('#chat-input').value = '';
  const wait = addMsg('bot', '<span class="muted">Consulting the tower…</span>');
  const ctx = { airspace: state.latest ? state.latest.airspace : 'baseline' };
  if (state.mission.site) ctx.mission_site = state.mission.site;
  if (state.drone.frames.length) ctx.current_drone_frame = state.drone.frames[state.drone.idx].name;
  if (state.landuse.tile) ctx.current_landuse_tile = state.landuse.tile;
  if (state.selected) ctx.selected_aircraft_icao24 = state.selected;
  try {
    const r = await postJSON('/api/assistant/chat', { message: text, t_idx: state.t, context: ctx });
    if (r.error) { wait.className = 'msg err'; wait.textContent = r.error; return; }
    wait.innerHTML = esc(r.answer || '(no answer)') + (r.tool_calls.length ? `<div class="tools">${r.tool_calls.map((t) => `<span>${esc(t.name)}</span>`).join('')}</div>` : '') + `<div class="hint">${r.model} · ${r.seconds}s</div>`;
  } catch (e) { wait.className = 'msg err'; wait.textContent = e.message; }
}

// ------------------------------------------------------------------ UI wiring
function switchTab(name) {
  $$('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
  if (name === 'fleet' && !state.fleet.data) loadFleet();
  if (name === 'metrics' && !state.metrics) loadMetrics();
}

function wire() {
  $$('.tabs button').forEach((b) => (b.onclick = () => switchTab(b.dataset.tab)));
  $('#btn-play').onclick = () => play(!state.playing);
  $('#btn-live').onclick = () => setLive(!state.live.on);
  $('#scenario').onchange = (e) => setScenario(e.target.value);
  $('#slider').oninput = (e) => { play(false); showSnapshot(parseInt(e.target.value)); };
  $('#m-pick').onclick = () => { state.mission.picking = !state.mission.picking; $('#m-pick').classList.toggle('active', state.mission.picking); };
  $('#m-assess').onclick = assessMission;
  $('#u-register').onclick = registerMission;
  $('#u-clear').onclick = async () => { await api('/api/utm/missions', { method: 'DELETE' }); await refreshAfterMissionChange(); };
  ['#m-lat', '#m-lon', '#m-radius', '#m-alt'].forEach((s) => ($(s).onchange = setMissionSite));
  $$('#tab-mission .presets .link').forEach((b) => (b.onclick = () => { const [la, lo] = b.dataset.site.split(','); $('#m-lat').value = la; $('#m-lon').value = lo; setMissionSite(); assessMission(); }));
  $('#d-slider').oninput = (e) => { state.drone.idx = parseInt(e.target.value); loadDroneFrame(); };
  $('#d-play').onclick = () => playDrone(!state.drone.playing);
  ['#d-gt', '#d-relocate'].forEach((s) => ($(s).onchange = loadDroneFrame));
  $('#d-tilt').oninput = (e) => { $('#d-tilt-v').textContent = e.target.value; };
  $('#d-hfov').oninput = (e) => { $('#d-hfov-v').textContent = e.target.value; };
  ['#d-tilt', '#d-hfov'].forEach((s) => ($(s).onchange = loadDroneFrame));
  $('#lu-tile').onchange = (e) => { state.landuse.tile = e.target.value; loadTile(); };
  $$('#tab-landuse .seg button').forEach((b) => (b.onclick = () => { $$('#tab-landuse .seg button').forEach((x) => x.classList.remove('active')); b.classList.add('active'); state.landuse.mode = b.dataset.mode; loadTile(); }));
  $('#fleet-subset').onchange = (e) => { state.fleet.subset = e.target.value; loadFleet(); };
  $('#chat-form').onsubmit = (e) => { e.preventDefault(); sendChat($('#chat-input').value); };
  $('#chat-reset').onclick = async () => { $('#chat').innerHTML = ''; await postJSON('/api/assistant/chat', { message: 'Conversation reset. Reply with one short line acknowledging.', reset: true, t_idx: state.t }).catch(() => {}); };
  $$('#tab-assistant .presets .link').forEach((b) => (b.onclick = () => { $('#chat-input').value = b.dataset.q; sendChat(b.dataset.q); }));
}

async function main() {
  wire();
  await initMap();
  await initScenarios();
  const live = await api('/api/live/status').catch(() => ({ active: false }));
  if (live.active) { state.live.on = true; $('#btn-live').classList.add('live'); $('#btn-live').textContent = '■ LIVE'; $('#slider').disabled = true;
    state.live.timer = setInterval(async () => { state.gen += 1; state.snapshots = {}; try { applySnapshot(await api('/api/airspace/snapshot/-1')); } catch (e) { console.warn(e); } }, 10000); }
  await resetReplay(false);
  api('/api/airspace/evaluate').then((ev) => { const h = ev.horizons; $('#eval-line').textContent = `${ev.predictor}: median error ${fmt(h['60'].median_m)} m @60 s · ${fmt(h['120'].median_m)} m @120 s · ${fmt(h['300'] ? h['300'].median_m : NaN)} m @300 s (n=${h['60'].n})`; }).catch(() => {});
  setMissionSite();
  initAssistant(); initDrone(); initLanduse();
  if (!state.live.on) setTimeout(() => play(true), 1200);
}

main().catch((e) => { console.error(e); alert('SkyOps failed to start: ' + e.message); });
