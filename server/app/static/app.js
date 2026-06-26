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

let cameraMarkers  = {};   // device_id → L.Marker
let fovPolygons    = {};   // device_id → L.Polygon
let objectMarkers  = {};   // object_id → L.Marker
let rayLayers      = {};   // `${device_id}__${object_id}` → L.Polyline
let dotLayers      = {};   // `${tri_lat},${tri_lon}` → L.CircleMarker
let mapFitted      = false;
let lastUpdateTime = null;
let telemetryOn    = true; // mirrors server state
const _snapshotKeyCache = {};  // deviceId → last s3_key seen

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

function speedMs(velLat, velLon) {
    const degPerSec = Math.sqrt((velLat||0)**2 + (velLon||0)**2);
    return (degPerSec * 111320).toFixed(2);
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

        // Key by device_id: one ray + one dot per camera, stable across polls.
        // object_id in detection_rays is position-based and changes every frame,
        // so using it as a key would cause flicker every 500 ms poll.
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
        // Only reload image if the S3 key changed — avoids constant flicker
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

    // Refresh snapshots after DOM is updated
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
// Telemetry toggle
// ─────────────────────────────────────────────────────────────────────────────

async function onTelemetryBtn() {
    const next = !telemetryOn;
    try {
        await fetch('/api/telemetry/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: next }),
        });
        telemetryOn = next;
        _updateTelemetryBtn();
    } catch (e) {
        console.error('Telemetry toggle failed:', e);
    }
}

function _updateTelemetryBtn() {
    const btn = document.getElementById('btn-telemetry');
    if (!btn) return;
    if (telemetryOn) {
        btn.textContent = 'Pause IoT';
        btn.classList.remove('success');
        btn.classList.add('danger');
    } else {
        btn.textContent = 'Resume IoT';
        btn.classList.remove('danger');
        btn.classList.add('success', 'active');
    }
}

async function syncTelemetryStatus() {
    try {
        const res = await fetch('/api/telemetry/status');
        const data = await res.json();
        telemetryOn = data.enabled;
        _updateTelemetryBtn();
    } catch (_) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// VLM
// ─────────────────────────────────────────────────────────────────────────────

async function onVLMToggle(enabled) {
    try {
        await fetch('/api/vlm/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled }),
        });
    } catch (e) {
        console.error('VLM toggle failed:', e);
    }
}

async function loadVLMStatus() {
    try {
        const res = await fetch('/api/vlm/status');
        const s = await res.json();

        const chk = document.getElementById('vlm-toggle');
        if (chk) chk.checked = s.enabled;

        const fieldsEl   = document.getElementById('vlm-fields');
        const isDroneEl  = document.getElementById('vlm-is-drone');
        const typeEl     = document.getElementById('vlm-drone-type');
        const actionEl   = document.getElementById('vlm-action');
        const metaEl     = document.getElementById('vlm-meta');

        if (s.last_result) {
            let parsed = null;
            try { parsed = JSON.parse(s.last_result); } catch (_) {}

            if (parsed && typeof parsed === 'object') {
                if (fieldsEl) fieldsEl.style.display = 'flex';

                if (isDroneEl) {
                    if (parsed.is_drone === true) {
                        isDroneEl.textContent = 'YES';
                        isDroneEl.className = 'vlm-field-val drone-yes';
                    } else if (parsed.is_drone === false) {
                        isDroneEl.textContent = 'NO';
                        isDroneEl.className = 'vlm-field-val drone-no';
                    } else {
                        isDroneEl.textContent = 'uncertain';
                        isDroneEl.className = 'vlm-field-val';
                    }
                }
                if (typeEl)   typeEl.textContent   = parsed.drone_type || '—';
                if (actionEl) actionEl.textContent  = parsed.action     || '—';
            } else {
                // Fallback: raw text (shouldn't happen with new analyzer)
                if (fieldsEl)  fieldsEl.style.display = 'none';
                if (actionEl)  actionEl.textContent   = s.last_result;
            }

            if (metaEl) {
                const ago = s.last_run ? relTime(new Date(s.last_run * 1000).toISOString()) : '—';
                metaEl.textContent = `${s.model || ''} · ${ago}`;
            }

        } else if (s.last_error) {
            if (fieldsEl)  fieldsEl.style.display = 'none';
            if (metaEl)    metaEl.textContent = 'Error: ' + s.last_error.slice(0, 120);
        } else {
            if (fieldsEl)  fieldsEl.style.display = 'none';
            if (metaEl)    metaEl.textContent = s.enabled
                ? 'Waiting for snapshot from edge…'
                : '';
        }
    } catch (_) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────

syncTelemetryStatus();
loadCameras();
loadStats();
updateTracks();
updateRays();
loadVLMStatus();

setInterval(updateTracks,  2_000);
setInterval(updateRays,    1_000);
setInterval(loadCameras,  15_000);
setInterval(loadStats,    10_000);
setInterval(loadVLMStatus, 60_000);
