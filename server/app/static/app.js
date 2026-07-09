// app.js — Sauron Tracking Dashboard

// ─────────────────────────────────────────────────────────────────────────────
// Map setup
// ─────────────────────────────────────────────────────────────────────────────

const MAP_DEFAULT_CENTER = [12.881, 77.551];
const MAP_DEFAULT_ZOOM   = 16;

const map = L.map('map').setView(MAP_DEFAULT_CENTER, MAP_DEFAULT_ZOOM);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20,
}).addTo(map);

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

let cameraMarkers  = {};
let fovPolygons    = {};
let objectMarkers  = {};
let rayLayers      = {};
let dotLayers      = {};
let mapFitted      = false;
let lastUpdateTime = null;
const _snapshotKeyCache = {};

// VLM deadman state (browser side)
let _vlmEnabled       = false;
let _vlmHeartbeatTimer = null;

const RAY_PALETTE = ['#4f7fe4', '#e4844f', '#5bc45b', '#c4a94f', '#9b72cf'];
const deviceColours = {};
let deviceColourIdx = 0;

function deviceColour(deviceId) {
    if (!deviceColours[deviceId]) {
        deviceColours[deviceId] = RAY_PALETTE[deviceColourIdx % RAY_PALETTE.length];
        deviceColourIdx++;
    }
    return deviceColours[deviceId];
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────────────────────────────────────

function switchTab(name) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => {
        const paneId = ['vlm-pane', 'log-pane'][i];
        const isActive = (name === 'vlm' && i === 0) || (name === 'log' && i === 1);
        b.classList.toggle('active', isActive);
        document.getElementById(paneId).classList.toggle('active', isActive);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function fmt(n, decimals = 6) {
    if (n == null || !Number.isFinite(n)) return '—';
    return n.toFixed(decimals);
}

function fmtDeg(n) {
    if (n == null || !Number.isFinite(n)) return '—';
    return n.toFixed(1) + '°';
}

function relTime(isoString) {
    if (!isoString) return '—';
    const dt  = new Date(isoString);
    const sec = Math.round((Date.now() - dt.getTime()) / 1000);
    if (sec < 2)  return 'just now';
    if (sec < 60) return `${sec}s ago`;
    if (sec < 3600) return `${Math.floor(sec/60)}m ago`;
    return `${Math.floor(sec/3600)}h ago`;
}

function fmtUTC(isoString) {
    if (!isoString) return '—';
    const d = new Date(isoString);
    return d.toISOString().replace('T', ' ').replace(/\.\d+Z$/, 'Z');
}

function fmtEpochRelTime(epochSec) {
    if (!epochSec) return '—';
    return relTime(new Date(epochSec * 1000).toISOString());
}

// ─────────────────────────────────────────────────────────────────────────────
// FOV cone rendering
// ─────────────────────────────────────────────────────────────────────────────

function buildFovPoints(lat, lon, bearingDeg, fovDeg, radiusM = 300) {
    const R = 6378137.0;
    const latRad = lat * Math.PI / 180;
    const pts = [[lat, lon]];
    const segments = 24;
    const start = bearingDeg - fovDeg / 2;
    const end   = bearingDeg + fovDeg / 2;
    for (let i = 0; i <= segments; i++) {
        const angle = (start + (i / segments) * (end - start)) * Math.PI / 180;
        const dy = radiusM * Math.cos(angle);
        const dx = radiusM * Math.sin(angle);
        pts.push([
            lat + (dy / R) * (180 / Math.PI),
            lon + (dx / (R * Math.cos(latRad))) * (180 / Math.PI),
        ]);
    }
    return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Live rays
// ─────────────────────────────────────────────────────────────────────────────

const _METERS_PER_DEG_LAT = 111320;

function _rayEndpoint(lat, lon, dx, dy, distanceM) {
    const refLatRad = lat * Math.PI / 180;
    const metersPerDegLon = _METERS_PER_DEG_LAT * Math.cos(refLatRad);
    return [
        lat + (dy * distanceM) / _METERS_PER_DEG_LAT,
        lon + (dx * distanceM) / metersPerDegLon,
    ];
}

async function updateRays() {
    let rays;
    try {
        const res = await fetch('/api/live_rays?within_seconds=5');
        rays = await res.json();
    } catch (e) {
        return;
    }

    const seenRays = new Set();
    const seenDots = new Set();

    rays.forEach(r => {
        if (!Number.isFinite(r.camera_lat) || !Number.isFinite(r.dx) || !Number.isFinite(r.tri_lat)) return;

        const rayKey = r.device_id;
        const dotKey = r.device_id;
        seenRays.add(rayKey);
        seenDots.add(dotKey);

        const colour = deviceColour(r.device_id);

        const refLatRad = r.camera_lat * Math.PI / 180;
        const mPerDegLon = _METERS_PER_DEG_LAT * Math.cos(refLatRad);
        const dLat = (r.tri_lat - r.camera_lat) * _METERS_PER_DEG_LAT;
        const dLon = (r.tri_lon - r.camera_lon) * mPerDegLon;
        const distToTri = Math.sqrt(dLat * dLat + dLon * dLon);
        const rayLen = Math.max(distToTri * 1.5, 150);
        const rayEnd = _rayEndpoint(r.camera_lat, r.camera_lon, r.dx, r.dy, rayLen);
        const latlngs = [[r.camera_lat, r.camera_lon], rayEnd];

        if (rayLayers[rayKey]) {
            rayLayers[rayKey].setLatLngs(latlngs);
        } else {
            rayLayers[rayKey] = L.polyline(latlngs, {
                color: colour, weight: 1.5, opacity: 0.55, dashArray: '6 4',
            }).addTo(map);
        }

        if (!dotLayers[dotKey]) {
            dotLayers[dotKey] = L.circleMarker([r.tri_lat, r.tri_lon], {
                radius: 5,
                color: 'rgba(255,255,255,0.6)',
                fillColor: '#4f7fe4',
                fillOpacity: 1,
                weight: 1.5,
            }).bindTooltip(
                `${r.device_id}<br>${fmt(r.tri_lat, 5)}, ${fmt(r.tri_lon, 5)}${r.tri_alt_m != null ? '<br>alt: ' + r.tri_alt_m.toFixed(1) + 'm' : ''}`,
                { sticky: true, className: 'ray-tooltip' }
            ).addTo(map);
        } else {
            dotLayers[dotKey].setLatLng([r.tri_lat, r.tri_lon]);
            dotLayers[dotKey].setTooltipContent(
                `${r.device_id}<br>${fmt(r.tri_lat, 5)}, ${fmt(r.tri_lon, 5)}${r.tri_alt_m != null ? '<br>alt: ' + r.tri_alt_m.toFixed(1) + 'm' : ''}`
            );
        }
    });

    Object.keys(rayLayers).forEach(k => {
        if (!seenRays.has(k)) { map.removeLayer(rayLayers[k]); delete rayLayers[k]; }
    });
    Object.keys(dotLayers).forEach(k => {
        if (!seenDots.has(k)) { map.removeLayer(dotLayers[k]); delete dotLayers[k]; }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshots
// ─────────────────────────────────────────────────────────────────────────────

async function refreshSnapshot(deviceId) {
    try {
        const res = await fetch(`/api/cameras/${encodeURIComponent(deviceId)}/snapshot`);
        if (!res.ok) return;
        const data = await res.json();
        if (!data.s3_key || data.s3_key === _snapshotKeyCache[deviceId]) return;
        _snapshotKeyCache[deviceId] = data.s3_key;
        const safeId = deviceId.replace(/[^a-zA-Z0-9_-]/g, '_');
        const wrap = document.getElementById(`snap-wrap-${safeId}`);
        const img  = document.getElementById(`snap-img-${safeId}`);
        const ts   = document.getElementById(`snap-ts-${safeId}`);
        if (!wrap || !img) return;
        img.src = data.url;
        wrap.classList.add('visible');
        if (ts && data.time) ts.textContent = relTime(data.time);
    } catch (_) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Cameras
// ─────────────────────────────────────────────────────────────────────────────

async function loadCameras() {
    let data;
    try {
        const res = await fetch('/api/cameras');
        data = await res.json();
    } catch (e) {
        console.error('Camera fetch failed:', e);
        return;
    }

    const valid = data.filter(c => Number.isFinite(c.lat) && Number.isFinite(c.lon));
    const container = document.getElementById('cameras-container');
    document.getElementById('camera-count').textContent = valid.length;

    if (valid.length === 0) {
        container.innerHTML = '<div class="empty-state">No cameras registered yet.</div>';
    }

    if (!mapFitted && valid.length > 0) {
        mapFitted = true;
        const lats = valid.map(c => c.lat);
        const lons = valid.map(c => c.lon);
        map.fitBounds([
            [Math.min(...lats) - 0.002, Math.min(...lons) - 0.002],
            [Math.max(...lats) + 0.002, Math.max(...lons) + 0.002],
        ]);
    }

    const seenIds = new Set();
    let html = '';

    data.forEach(cam => {
        seenIds.add(cam.device_id);
        const lat     = cam.lat;
        const lon     = cam.lon;
        const heading = Number.isFinite(cam.last_heading) ? cam.last_heading : 0;
        const pitch   = Number.isFinite(cam.last_pitch)   ? cam.last_pitch   : 0;
        const roll    = Number.isFinite(cam.last_roll)    ? cam.last_roll    : 0;
        const fov     = 90;
        const colour  = deviceColour(cam.device_id);

        const isValid  = Number.isFinite(lat) && Number.isFinite(lon);
        const lastSeen = cam.last_seen;
        const seenAgo  = relTime(lastSeen);
        const isStale  = lastSeen
            ? (Date.now() - new Date(lastSeen).getTime()) > 30_000
            : true;

        if (isValid) {
            if (!cameraMarkers[cam.device_id]) {
                const icon = L.divIcon({
                    className: '',
                    html: `<div style="width:10px;height:10px;border-radius:50%;background:${colour};border:2px solid rgba(255,255,255,0.5);"></div>`,
                    iconSize: [10, 10],
                    iconAnchor: [5, 5],
                });
                cameraMarkers[cam.device_id] = L.marker([lat, lon], { icon })
                    .addTo(map)
                    .bindPopup(`<b>${cam.device_id}</b>`);
            } else {
                cameraMarkers[cam.device_id].setLatLng([lat, lon]);
            }
            cameraMarkers[cam.device_id]
                .getPopup()
                .setContent(
                    `<b>${cam.device_id}</b><br>` +
                    `${fmt(lat, 6)}, ${fmt(lon, 6)}<br>` +
                    `H: ${fmtDeg(heading)}  P: ${fmtDeg(pitch)}  R: ${fmtDeg(roll)}`
                );

            if (fovPolygons[cam.device_id]) map.removeLayer(fovPolygons[cam.device_id]);
            fovPolygons[cam.device_id] = L.polygon(
                buildFovPoints(lat, lon, heading, fov),
                { color: colour, fillColor: colour, fillOpacity: 0.06, weight: 1, dashArray: '4 4' }
            ).addTo(map);
        }

        const dotClass = isStale ? 'dot stale' : 'dot';
        const colourStyle = `style="background:${colour};box-shadow:0 0 4px ${colour}"`;
        const safeId = cam.device_id.replace(/[^a-zA-Z0-9_-]/g, '_');
        html += `
        <div class="camera-card">
            <div class="camera-id">
                <span class="${dotClass}" ${isStale ? '' : colourStyle}></span>
                ${cam.device_id}
            </div>
            <div class="camera-grid">
                <div class="kv"><span class="k">Heading</span><span class="v">${fmtDeg(heading)}</span></div>
                <div class="kv"><span class="k">Pitch</span><span class="v">${fmtDeg(pitch)}</span></div>
                <div class="kv"><span class="k">Roll</span><span class="v">${fmtDeg(roll)}</span></div>
                <div class="kv"><span class="k">FOV</span><span class="v">${fov}°</span></div>
            </div>
            <div class="camera-coords">${fmt(lat, 6)}, ${fmt(lon, 6)}</div>
            <div class="camera-lastseen">Last seen: ${seenAgo}</div>
            <div class="snapshot-wrap" id="snap-wrap-${safeId}">
                <img class="snapshot-img" id="snap-img-${safeId}" alt="latest frame" />
                <div class="snapshot-time" id="snap-ts-${safeId}"></div>
            </div>
        </div>`;
    });

    if (data.length > 0) container.innerHTML = html;

    data.forEach(cam => { delete _snapshotKeyCache[cam.device_id]; });
    data.forEach(cam => refreshSnapshot(cam.device_id));

    Object.keys(cameraMarkers).forEach(id => {
        if (!seenIds.has(id)) {
            map.removeLayer(cameraMarkers[id]);
            delete cameraMarkers[id];
            if (fovPolygons[id]) { map.removeLayer(fovPolygons[id]); delete fovPolygons[id]; }
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats header
// ─────────────────────────────────────────────────────────────────────────────

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const s = await res.json();
        document.getElementById('stat-cameras').textContent = s.camera_count ?? '—';
        const trackVal = document.getElementById('stat-tracks');
        trackVal.textContent = s.active_tracks ?? '—';
        trackVal.className   = 'value ' + (s.active_tracks > 0 ? 'warn' : 'ok');
        document.getElementById('stat-updated').textContent = relTime(s.last_camera_seen);
    } catch (_) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Live tracks
// ─────────────────────────────────────────────────────────────────────────────

async function updateTracks() {
    let tracks;
    try {
        const res = await fetch('/api/live_tracks?within_seconds=15');
        tracks = await res.json();
    } catch (e) {
        console.error('Track fetch failed:', e);
        return;
    }

    lastUpdateTime = new Date();
    document.getElementById('footer-time').textContent =
        lastUpdateTime.toLocaleTimeString();

    const container  = document.getElementById('tracks-container');
    const trackCount = document.getElementById('track-count');
    trackCount.textContent = tracks.length;

    const seenIds = new Set();
    let html = '';

    tracks.forEach(t => {
        seenIds.add(t.object_id);
        const pos       = [t.lat, t.lon];
        const sources   = Array.isArray(t.source_cameras) ? t.source_cameras.join(', ') : '—';
        const trackTime = t.time ? relTime(t.time) : '—';
        const alt       = t.altitude_m != null ? t.altitude_m.toFixed(1) + 'm' : '—';

        if (!objectMarkers[t.object_id]) {
            const icon = L.divIcon({ className: 'object-marker', iconSize: [10, 10] });
            objectMarkers[t.object_id] = L.marker(pos, { icon })
                .addTo(map)
                .bindPopup(`<b>${t.object_id}</b>`);
        } else {
            objectMarkers[t.object_id].setLatLng(pos);
        }
        objectMarkers[t.object_id].getPopup().setContent(
            `<b>${t.object_id}</b><br>` +
            `${fmt(t.lat, 6)}, ${fmt(t.lon, 6)}<br>` +
            `Alt: ${alt}<br>` +
            `Sources: ${sources}`
        );

        html += `
        <div class="track-card" onclick="map.setView([${t.lat},${t.lon}], 18)">
            <div class="track-id">${t.object_id}</div>
            <div class="track-pos">${fmt(t.lat, 6)}, ${fmt(t.lon, 6)}</div>
            <div class="track-alt">Alt: ${alt}</div>
            <div class="track-detail-grid">
                <div class="kv"><span class="k">updated</span><span class="v">${trackTime}</span></div>
                <div class="kv"><span class="k">sources</span><span class="v">${sources}</span></div>
            </div>
        </div>`;
    });

    Object.keys(objectMarkers).forEach(id => {
        if (!seenIds.has(id)) {
            map.removeLayer(objectMarkers[id]);
            delete objectMarkers[id];
        }
    });

    container.innerHTML = html || '<div class="empty-state">No active tracks.</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// VLM — deadman toggle + heartbeat
// ─────────────────────────────────────────────────────────────────────────────

function onVLMToggle(checked) {
    _vlmEnabled = checked;

    fetch('/api/vlm/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: checked }),
    }).catch(e => console.error('VLM toggle failed:', e));

    if (checked) {
        // Immediately send first heartbeat, then every 30 s
        _scheduleVLMHeartbeats();
    } else {
        _stopVLMHeartbeats();
    }
}

function _scheduleVLMHeartbeats() {
    _stopVLMHeartbeats();
    _sendHeartbeat();
    _vlmHeartbeatTimer = setInterval(_sendHeartbeat, 30_000);
}

function _stopVLMHeartbeats() {
    if (_vlmHeartbeatTimer !== null) {
        clearInterval(_vlmHeartbeatTimer);
        _vlmHeartbeatTimer = null;
    }
}

async function _sendHeartbeat() {
    if (!_vlmEnabled) return;
    try {
        await fetch('/api/vlm/heartbeat', { method: 'POST' });
    } catch (_) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// VLM — render history cards
// ─────────────────────────────────────────────────────────────────────────────

function _renderVLMCard(entry, idx) {
    const isCurrent = idx === 0;
    const badgeClass = isCurrent ? 'vlm-card-badge' : 'vlm-card-badge past';
    const badgeText  = isCurrent ? 'Latest' : `#${idx + 1}`;
    const timeStr    = entry.run_ts
        ? fmtUTC(new Date(entry.run_ts * 1000).toISOString())
        : '—';

    let bodyHtml = '';

    if (entry.error) {
        bodyHtml = `<div class="vlm-card-error">${escHtml(entry.error.slice(0, 200))}</div>`;
    } else if (entry.result) {
        let parsed = null;
        try { parsed = JSON.parse(entry.result); } catch (_) {}

        if (parsed && typeof parsed === 'object') {
            let droneVal = '—';
            let droneClass = 'vlm-field-val';
            if (parsed.is_drone === true)  { droneVal = 'YES'; droneClass = 'vlm-field-val drone-yes'; }
            if (parsed.is_drone === false) { droneVal = 'NO';  droneClass = 'vlm-field-val drone-no'; }

            bodyHtml = `
            <div class="vlm-fields">
                <div class="vlm-field">
                    <span class="vlm-field-key">Drone?</span>
                    <span class="${droneClass}">${droneVal}</span>
                </div>
                <div class="vlm-field">
                    <span class="vlm-field-key">Type</span>
                    <span class="vlm-field-val">${escHtml(parsed.drone_type || '—')}</span>
                </div>
                <div class="vlm-field">
                    <span class="vlm-field-key">Action</span>
                    <span class="vlm-field-val">${escHtml(parsed.action || '—')}</span>
                </div>
            </div>`;
        } else {
            bodyHtml = `<div class="vlm-field-val">${escHtml(entry.result.slice(0, 200))}</div>`;
        }
    }

    const meta = [
        entry.device_id ? escHtml(entry.device_id) : null,
        entry.model ? escHtml(entry.model) : null,
        entry.duration_ms ? `${entry.duration_ms}ms` : null,
    ].filter(Boolean).join(' · ');

    return `
    <div class="vlm-card ${isCurrent ? 'is-current' : ''}">
        <div class="vlm-card-header">
            <span class="${badgeClass}">${badgeText}</span>
            <span class="vlm-card-time">${timeStr}</span>
        </div>
        ${bodyHtml}
        ${meta ? `<div class="vlm-card-meta">${meta}</div>` : ''}
    </div>`;
}

async function loadVLMStatus() {
    try {
        const res = await fetch('/api/vlm/status');
        const s = await res.json();

        // Sync toggle state with server (server is source of truth on expiry)
        const chk = document.getElementById('vlm-toggle');
        if (chk) {
            // Only update if server says it expired — don't fight user clicks
            if (!s.enabled && _vlmEnabled) {
                // server expired the deadman
                _vlmEnabled = false;
                chk.checked = false;
                _stopVLMHeartbeats();
            } else if (s.enabled && !chk.checked) {
                chk.checked = true;
            }
        }

        // Countdown badge
        const ttlEl = document.getElementById('vlm-ttl');
        if (ttlEl && s.enabled && s.enabled_until) {
            const secsLeft = Math.max(0, Math.round(s.enabled_until - Date.now() / 1000));
            ttlEl.textContent = `${secsLeft}s`;
        } else if (ttlEl) {
            ttlEl.textContent = '';
        }

        // Render history
        const history = Array.isArray(s.history) ? s.history : [];
        const historyEl = document.getElementById('vlm-history');
        const emptyEl   = document.getElementById('vlm-empty');

        if (history.length === 0) {
            if (historyEl) historyEl.innerHTML = '';
            if (emptyEl)   emptyEl.style.display = 'block';
        } else {
            if (emptyEl) emptyEl.style.display = 'none';
            if (historyEl) {
                historyEl.innerHTML = history.slice(0, 4).map((e, i) => _renderVLMCard(e, i)).join('');
            }
        }

    } catch (_) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Message log
// ─────────────────────────────────────────────────────────────────────────────

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function toggleMsgRow(el) {
    el.classList.toggle('expanded');
}

async function loadMessages() {
    if (document.querySelector('#log-container .msg-row.expanded')) return;

    let msgs;
    try {
        const res = await fetch('/api/messages?limit=10');
        msgs = await res.json();
    } catch (_) { return; }

    const container = document.getElementById('log-container');
    const emptyEl   = document.getElementById('log-empty');

    if (!msgs || msgs.length === 0) {
        if (emptyEl) emptyEl.style.display = 'block';
        return;
    }

    if (emptyEl) emptyEl.style.display = 'none';

    let html = '';
    msgs.forEach(m => {
        const ts      = fmtUTC(m.time);
        const device  = escHtml(m.device_id || '?');
        const topic   = escHtml(m.topic || '');
        const payload = m.payload
            ? escHtml(JSON.stringify(m.payload, null, 2))
            : '(no payload)';

        html += `
        <div class="msg-row" onclick="this.classList.toggle('expanded')">
            <div class="msg-row-header">
                <span class="msg-ts">${ts}</span>
                <span class="msg-device">${device}</span>
            </div>
            <div class="msg-topic">${topic}</div>
            <pre class="msg-payload">${payload}</pre>
        </div>`;
    });

    container.innerHTML = html;
}

// ─────────────────────────────────────────────────────────────────────────────
// Situation summary
// ─────────────────────────────────────────────────────────────────────────────

let _lastSummaryTs = 0;

async function refreshSummary(force = false) {
    const btn    = document.getElementById('summary-btn');
    const body   = document.getElementById('summary-body');
    const meta   = document.getElementById('summary-meta');

    if (btn) { btn.disabled = true; btn.textContent = '…'; }

    try {
        const url = force ? '/api/summary?force=true' : '/api/summary';
        const res = await fetch(url);
        const s   = await res.json();

        if (s.result) {
            body.innerHTML = escHtml(s.result);
            _lastSummaryTs = s.ts;
            const dur = s.duration_ms ? ` · ${s.duration_ms}ms` : '';
            meta.textContent = s.ts
                ? `${fmtUTC(new Date(s.ts * 1000).toISOString())}${dur}`
                : '';
        } else if (s.error) {
            body.innerHTML = `<span class="summary-error">${escHtml(s.error)}</span>`;
            meta.textContent = '';
        }
    } catch (e) {
        if (body) body.innerHTML = `<span class="summary-error">Request failed</span>`;
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Refresh'; }
    }
}

// Auto-refresh summary every 60 s (uses cache on server side, not wasteful)
setInterval(() => refreshSummary(false), 60_000);

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────

loadCameras();
loadStats();
updateTracks();
updateRays();
loadVLMStatus();
loadMessages();

setInterval(updateTracks,   2_000);
setInterval(updateRays,     1_000);
setInterval(loadCameras,   15_000);
setInterval(loadStats,     10_000);
setInterval(loadVLMStatus,  3_000);
setInterval(loadMessages,   2_000);
