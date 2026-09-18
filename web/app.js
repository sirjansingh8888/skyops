/* SkyOps console: replay map (MapLibre + deck.gl), mission briefs, drone perception, land use, fleet, assistant. */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
  return r.json();
};
const fmt = (n, d = 0) => (n === null || n === undefined || Number.isNaN(n) ? '–' : Number(n).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const SEV_RGB = { critical: [239, 68, 68], alert: [251, 146, 60], warning: [245, 158, 11], marginal: [100, 116, 139], info: [100, 116, 139] };
const LABEL_RGB = { pedestrian: [80, 220, 100], people: [80, 220, 100], bicycle: [250, 200, 60], car: [60, 160, 255], van: [60, 200, 255],
  truck: [255, 120, 60], tricycle: [200, 120, 255], 'awning-tricycle': [200, 120, 255], bus: [255, 80, 160], motor: [250, 230, 80] };

const state = {
  t: 0, n: 20, playing: false, timer: null, snapshots: {}, tracks: null, summary: null, selected: null,
  conflictSet: new Set(), anomalySet: new Set(),
  mission: { site: null, result: null, picking: false },
  drone: { frames: [], idx: 0, result: null, playing: false, timer: null },
  landuse: { tiles: [], tile: null, mode: 'overlay', result: null },
  fleet: { subset: 'FD001', data: null, unit: null },
  assistant: { available: false, model: '' },
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
  if (layer.id === 'aircraft') {
    const s = object;
    return { html: `<b>${esc(s.callsign)}</b> · ${esc(s.origin_country)}<br>${fmt(s.altitude_ft)} ft · ${fmt(s.speed_kt)} kt · ${fmt(s.vs_fpm)} fpm · hdg ${fmt(s.true_track)}°${s.on_ground ? '<br>on ground' : ''}`,
      style: { background: '#0f1626', color: '#e5e7eb', fontSize: '12px', border: '1px solid #233047', borderRadius: '6px', padding: '6px 8px' } };
  }
  if (layer.id === 'detections') return { text: `${object.label} ${fmt(object.conf, 2)} · ${fmt(object.range_m)} m from drone` };
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
  if (state.conflictSet.has(s.icao24)) return [239, 68, 68, 255];
  if (state.anomalySet.has(s.icao24)) return [245, 158, 11, 255];
  return altColor(s);
}

function currentSnap() { return state.snapshots[state.t]; }

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
    const tRel = snap.t_rel;
    if (state.tracks) {
      const trails = snap.states.map((s) => ({ path: (state.tracks[s.icao24] || []).filter((p) => p[3] <= tRel).slice(-8).map((p) => [p[0], p[1]]) })).filter((d) => d.path.length > 1);
      layers.push(new deck.PathLayer({ id: 'trails', data: trails, getPath: (d) => d.path, getColor: [148, 163, 184, 80], widthMinPixels: 1, widthMaxPixels: 2 }));
    }
    const preds = snap.states.filter((s) => snap.predicted[s.icao24] && !s.on_ground).map((s) => ({
      path: [[s.longitude, s.latitude], ...snap.predicted[s.icao24].map((p) => [p[0], p[1]])],
      color: state.conflictSet.has(s.icao24) ? [239, 68, 68, 170] : [56, 189, 248, 110] }));
    layers.push(new deck.PathLayer({ id: 'predicted', data: preds, getPath: (d) => d.path, getColor: (d) => d.color, widthMinPixels: 1.2, updateTriggers: { getColor: [state.t] } }));
    layers.push(new deck.LineLayer({ id: 'conflict-lines', data: snap.conflicts.filter((c) => c.severity !== 'marginal'),
      getSourcePosition: (c) => [c.a.lon, c.a.lat], getTargetPosition: (c) => [c.b.lon, c.b.lat], getColor: (c) => [...SEV_RGB[c.severity], 230], getWidth: 2, widthUnits: 'pixels' }));
    layers.push(new deck.ScatterplotLayer({ id: 'conflict-rings', data: snap.conflicts.filter((c) => c.severity !== 'marginal').flatMap((c) => [{ p: [c.a.lon, c.a.lat], s: c.severity }, { p: [c.b.lon, c.b.lat], s: c.severity }]),
      getPosition: (d) => d.p, getRadius: 9, radiusUnits: 'pixels', stroked: true, filled: false, getLineColor: (d) => [...SEV_RGB[d.s], 200], lineWidthMinPixels: 1.5 }));
    layers.push(new deck.ScatterplotLayer({ id: 'anomaly-rings', data: snap.anomalies.filter((a) => a.severity !== 'info'), getPosition: (a) => [a.lon, a.lat], getRadius: 13, radiusUnits: 'pixels',
      stroked: true, filled: false, getLineColor: (a) => [...SEV_RGB[a.severity], 220], lineWidthMinPixels: 2 }));
    layers.push(new deck.IconLayer({ id: 'aircraft', data: snap.states, iconAtlas: ICON_URL, iconMapping: { plane: { x: 0, y: 0, width: 64, height: 64, mask: true } },
      getIcon: () => 'plane', getPosition: (s) => [s.longitude, s.latitude], getSize: (s) => (s.icao24 === state.selected ? 30 : 20), sizeUnits: 'pixels',
      getAngle: (s) => 360 - (s.true_track || 0), getColor: aircraftColor, pickable: true, transitions: { getPosition: 800, getAngle: 800 },
      updateTriggers: { getColor: [state.t, state.selected], getSize: [state.selected] } }));
    const zoom = map ? map.getZoom() : 4;
    const labelled = snap.states.filter((s) => zoom >= 6.5 || s.icao24 === state.selected || state.conflictSet.has(s.icao24) || state.anomalySet.has(s.icao24));
    layers.push(new deck.TextLayer({ id: 'callsigns', data: labelled, getPosition: (s) => [s.longitude, s.latitude], getText: (s) => s.callsign, getSize: 11,
      getColor: [229, 231, 235, 235], getPixelOffset: [0, 17], fontFamily: 'JetBrains Mono, Consolas, monospace', updateTriggers: { getText: [state.t] } }));
  }

  const site = state.mission.site;
  if (site) {
    layers.push(new deck.ScatterplotLayer({ id: 'mission-radius', data: [site], getPosition: (d) => [d.lon, d.lat], getRadius: site.radius_km * 1000, radiusUnits: 'meters',
      stroked: true, filled: true, getFillColor: [34, 197, 94, 18], getLineColor: [34, 197, 94, 160], lineWidthMinPixels: 1.5 }));
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

// ------------------------------------------------------------------ replay
async function loadSnapshot(i) {
  if (!state.snapshots[i]) state.snapshots[i] = await api(`/api/airspace/snapshot/${i}`);
  return state.snapshots[i];
}

async function showSnapshot(i) {
  state.t = i;
  const snap = await loadSnapshot(i);
  state.conflictSet = new Set(snap.conflicts.filter((c) => c.severity !== 'marginal').flatMap((c) => [c.a.icao24, c.b.icao24]));
  state.anomalySet = new Set(snap.anomalies.filter((a) => a.severity !== 'info').map((a) => a.icao24));
  $('#slider').value = i; $('#time-label').textContent = `T+${snap.t_rel} s · snapshot ${i + 1}/${snap.n_snapshots}`;
  const cs = snap.conflict_summary, as = snap.anomaly_summary;
  const nConf = cs.critical + cs.alert + cs.warning, nAnom = as.critical + as.alert + as.warning;
  $('#chip-aircraft').textContent = `${snap.states.length} aircraft · ${snap.states.filter((s) => !s.on_ground).length} airborne`;
  const cc = $('#chip-conflicts'); cc.textContent = `${nConf} conflicts (${cs.critical} critical) · ${cs.marginal} marginal`; cc.className = 'chip ' + (cs.critical ? 'hot' : nConf ? 'warm' : 'ok');
  const ca = $('#chip-anomalies'); ca.textContent = `${nAnom} anomalies · ${as.info} info`; ca.className = 'chip ' + (as.critical ? 'hot' : nAnom ? 'warm' : 'ok');
  renderLists(snap); renderAircraftCard(); render();
}

function renderLists(snap) {
  const ul = $('#conflicts'); ul.innerHTML = '';
  $('#conf-count').textContent = snap.conflicts.length;
  for (const c of snap.conflicts) {
    const li = document.createElement('li'); li.className = c.severity;
    li.innerHTML = `<b>${esc(c.a.callsign)}</b> ↔ <b>${esc(c.b.callsign)}</b> · ${c.severity}${c.terminal ? ' (terminal)' : ''}<br><span class="muted">${c.t_loss_s === 0 ? 'separation lost now' : `loss in ${c.t_loss_s} s`} · CPA ${c.cpa_nm} NM · Δalt ${c.v_sep_ft} ft · ${c.converging ? 'converging' : 'diverging'}</span>`;
    li.onclick = () => { state.selected = c.a.icao24; map.flyTo({ center: [(c.a.lon + c.b.lon) / 2, (c.a.lat + c.b.lat) / 2], zoom: 8.5 }); renderAircraftCard(); render(); };
    ul.appendChild(li);
  }
  if (!snap.conflicts.length) ul.innerHTML = '<li class="info">No conflicts at this snapshot.</li>';
  const ua = $('#anomalies'); ua.innerHTML = '';
  $('#anom-count').textContent = snap.anomalies.length;
  for (const a of snap.anomalies) {
    const li = document.createElement('li'); li.className = a.severity;
    li.innerHTML = `<b>${esc(a.callsign)}</b> · ${esc(a.type.toLowerCase().replace(/_/g, ' '))}<br><span class="muted">${esc(a.detail)}</span>`;
    li.onclick = () => selectAircraft(a.icao24, true);
    ua.appendChild(li);
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
    ...snap.anomalies.filter((a) => a.icao24 === s.icao24).map((a) => `${a.type.toLowerCase().replace(/_/g, ' ')}: ${a.detail}`)];
  el.innerHTML = `<div class="card-title">${esc(s.callsign)} <span class="muted">${esc(s.icao24)} · ${esc(s.origin_country)}</span></div>
    <div class="kv"><div><span>altitude</span>${fmt(s.altitude_ft)} ft</div><div><span>ground speed</span>${fmt(s.speed_kt)} kt</div><div><span>vertical</span>${fmt(s.vs_fpm)} fpm</div>
    <div><span>heading</span>${fmt(s.true_track)}°</div><div><span>squawk</span>${esc(s.squawk || '–')}</div><div><span>nearest airport</span>${fmt(s.airport_km)} km</div></div>
    ${flags.length ? `<ul class="reasons">${flags.map((f) => `<li class="warning">${esc(f)}</li>`).join('')}</ul>` : ''}
    <div class="hint">Dead-reckoning prediction<br>${pred || '–'}</div>
    <div class="row"><button class="link" id="ac-ask">Ask the assistant about ${esc(s.callsign)}</button><button class="link" id="ac-site">Assess a drone launch below it</button></div>`;
  $('#ac-ask').onclick = () => { switchTab('assistant'); $('#chat-input').value = `Tell me about ${s.callsign}: is it behaving normally and is it involved in any conflict?`; };
  $('#ac-site').onclick = () => { $('#m-lat').value = s.latitude.toFixed(4); $('#m-lon').value = s.longitude.toFixed(4); switchTab('mission'); setMissionSite(); assessMission(); };
}

function play(on) {
  state.playing = on; $('#btn-play').textContent = on ? '⏸ Pause' : '▶ Play';
  clearInterval(state.timer);
  if (on) state.timer = setInterval(() => showSnapshot((state.t + 1) % state.n), 1500);
}

// ------------------------------------------------------------------ mission
function setMissionSite() {
  const lat = parseFloat($('#m-lat').value), lon = parseFloat($('#m-lon').value), radius_km = parseFloat($('#m-radius').value) || 10;
  if (Number.isNaN(lat) || Number.isNaN(lon)) return;
  state.mission.site = { lat, lon, radius_km, alt_m: parseFloat($('#m-alt').value) || 100 };
  render();
}

async function assessMission() {
  setMissionSite();
  const site = state.mission.site; if (!site) return;
  const body = { lat: site.lat, lon: site.lon, alt_m: site.alt_m, radius_km: site.radius_km, t_idx: state.t, use_gt: $('#d-gt').checked };
  if ($('#m-attach-frame').checked && state.drone.frames.length) body.frame = state.drone.frames[state.drone.idx].name;
  if ($('#m-attach-tile').checked && state.landuse.tile) body.tile = state.landuse.tile;
  const el = $('#mission-result'); el.innerHTML = '<div class="muted">Assessing…</div>';
  try {
    const r = await api('/api/mission/brief', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
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
  svgChart($('#chart-rul'), { title: 'Remaining useful life (cycles)', x: h.cycles, series: [{ y: h.rul.p50, color: '#38bdf8', name: 'RUL p50' }], band: { lo: h.rul.p10, hi: h.rul.p90, color: 'rgba(56,189,248,0.18)' } });
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
  svg += series.map((s, i) => `<text x="${w - pad.r - 130}" y="${14 + i * 11}" fill="${s.color}" style="fill:${s.color}">${esc(s.name).slice(0, 26)}</text>`).join('');
  el.innerHTML = svg + '</svg>';
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
  const ctx = {};
  if (state.mission.site) ctx.mission_site = state.mission.site;
  if (state.drone.frames.length) ctx.current_drone_frame = state.drone.frames[state.drone.idx].name;
  if (state.landuse.tile) ctx.current_landuse_tile = state.landuse.tile;
  if (state.selected) ctx.selected_aircraft_icao24 = state.selected;
  try {
    const r = await api('/api/assistant/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text, t_idx: state.t, context: ctx }) });
    if (r.error) { wait.className = 'msg err'; wait.textContent = r.error; return; }
    wait.innerHTML = esc(r.answer || '(no answer)') + (r.tool_calls.length ? `<div class="tools">${r.tool_calls.map((t) => `<span>${esc(t.name)}</span>`).join('')}</div>` : '') + `<div class="hint">${r.model} · ${r.seconds}s</div>`;
  } catch (e) { wait.className = 'msg err'; wait.textContent = e.message; }
}

// ------------------------------------------------------------------ UI wiring
function switchTab(name) {
  $$('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
  if (name === 'fleet' && !state.fleet.data) loadFleet();
}

function wire() {
  $$('.tabs button').forEach((b) => (b.onclick = () => switchTab(b.dataset.tab)));
  $('#btn-play').onclick = () => play(!state.playing);
  $('#slider').oninput = (e) => { play(false); showSnapshot(parseInt(e.target.value)); };
  $('#m-pick').onclick = () => { state.mission.picking = !state.mission.picking; $('#m-pick').classList.toggle('active', state.mission.picking); };
  $('#m-assess').onclick = assessMission;
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
  $('#chat-reset').onclick = async () => { $('#chat').innerHTML = ''; await api('/api/assistant/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: 'Conversation reset. Reply with one short line acknowledging.', reset: true, t_idx: state.t }) }).catch(() => {}); };
  $$('#tab-assistant .presets .link').forEach((b) => (b.onclick = () => { $('#chat-input').value = b.dataset.q; sendChat(b.dataset.q); }));
}

async function main() {
  wire();
  await initMap();
  state.summary = await api('/api/airspace/summary');
  state.n = state.summary.n_snapshots; $('#slider').max = state.n - 1;
  await showSnapshot(0);
  api('/api/airspace/tracks').then((d) => { state.tracks = d.tracks; render(); });
  (async () => { for (let i = 1; i < state.n; i++) await loadSnapshot(i); })();
  api('/api/airspace/evaluate').then((ev) => { const h = ev.horizons; $('#eval-line').textContent = `${ev.predictor}: median error ${fmt(h['60'].median_m)} m @60 s · ${fmt(h['120'].median_m)} m @120 s · ${fmt(h['300'] ? h['300'].median_m : NaN)} m @300 s (n=${h['60'].n})`; }).catch(() => {});
  setMissionSite();
  initAssistant(); initDrone(); initLanduse();
  setTimeout(() => play(true), 1200);
}

main().catch((e) => { console.error(e); alert('SkyOps failed to start: ' + e.message); });
