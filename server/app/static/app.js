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
let velocityLines  = {};   // object_id → L.Polyline
let mapFitted      = false;
let lastUpdateTime = null;

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

    // Auto-fit map on first load
    if (!mapFitted && valid.length > 0) {
        mapFitted = true;
        const lats = valid.map(c => c.lat);
        const lons = valid.map(c => c.lon);
        map.fitBounds([
            [Math.min(...lats) - 0.002, Math.min(...lons) - 0.002],
            [Math.max(...lats) + 0.002, Math.max(...lons) + 0.002],
        ]);
    }

    // Update map markers + sidebar cards
    const seenIds = new Set();
    let html = '';

    data.forEach(cam => {
        seenIds.add(cam.device_id);
        const lat     = cam.lat;
        const lon     = cam.lon;
        const heading = Number.isFinite(cam.last_heading) ? cam.last_heading : 0;
        const pitch   = Number.isFinite(cam.last_pitch)   ? cam.last_pitch   : 0;
        const roll    = Number.isFinite(cam.last_roll)    ? cam.last_roll    : 0;
        const fov     = 90;  // default; not stored in DB yet

        const isValid = Number.isFinite(lat) && Number.isFinite(lon);
        const lastSeen = cam.last_seen;
        const seenAgo  = relTime(lastSeen);
        const isStale  = lastSeen
            ? (Date.now() - new Date(lastSeen).getTime()) > 30_000
            : true;

        if (isValid) {
            // Map marker
            if (!cameraMarkers[cam.device_id]) {
                cameraMarkers[cam.device_id] = L.marker([lat, lon])
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

            // FOV cone — update on every refresh
            if (fovPolygons[cam.device_id]) map.removeLayer(fovPolygons[cam.device_id]);
            fovPolygons[cam.device_id] = L.polygon(
                buildFovPoints(lat, lon, heading, fov),
                { color: '#ffaa00', fillColor: '#ffaa00', fillOpacity: 0.08, weight: 1, dashArray: '4 4' }
            ).addTo(map);
        }

        // Sidebar card
        const dotClass = isStale ? 'dot stale' : 'dot';
        html += `
        <div class="camera-card">
            <div class="camera-id">
                <span class="${dotClass}"></span>
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
        </div>`;
    });

    if (data.length > 0) container.innerHTML = html;

    // Remove stale map layers
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
        const res = await fetch('/api/live_tracks?within_seconds=5');
        tracks = await res.json();
    } catch (e) {
        console.error('Track fetch failed:', e);
        return;
    }

    lastUpdateTime = new Date();
    document.getElementById('footer-time').textContent =
        lastUpdateTime.toLocaleTimeString();

    const container   = document.getElementById('tracks-container');
    const trackCount  = document.getElementById('track-count');
    trackCount.textContent = tracks.length;

    const seenIds = new Set();
    let html = '';

    tracks.forEach(t => {
        seenIds.add(t.object_id);
        const pos       = [t.lat, t.lon];
        const futurePos = [t.lat + (t.vel_lat||0) * 2, t.lon + (t.vel_lon||0) * 2];
        const spd       = parseFloat(speedMs(t.vel_lat, t.vel_lon));
        const spdClass  = spd > 3 ? 'fast' : '';
        const sources   = Array.isArray(t.source_cameras) ? t.source_cameras.join(', ') : '—';
        const trackTime = t.time ? relTime(t.time) : '—';

        // Map marker
        if (!objectMarkers[t.object_id]) {
            const icon = L.divIcon({ className: 'object-marker', iconSize: [10, 10] });
            objectMarkers[t.object_id] = L.marker(pos, { icon })
                .addTo(map)
                .bindPopup(`<b>${t.object_id}</b>`);
            velocityLines[t.object_id] = L.polyline([pos, futurePos], {
                color: '#00ffcc', weight: 1.5, opacity: 0.4,
            }).addTo(map);
        } else {
            objectMarkers[t.object_id].setLatLng(pos);
            velocityLines[t.object_id].setLatLngs([pos, futurePos]);
        }
        objectMarkers[t.object_id].getPopup().setContent(
            `<b>${t.object_id}</b><br>` +
            `${fmt(t.lat, 6)}, ${fmt(t.lon, 6)}<br>` +
            `Speed: ${spd} m/s<br>` +
            `Sources: ${sources}`
        );

        // Sidebar card
        html += `
        <div class="track-card" onclick="map.setView([${t.lat},${t.lon}], 18)">
            <div class="track-id">${t.object_id}</div>
            <div class="track-pos">${fmt(t.lat, 6)}, ${fmt(t.lon, 6)}</div>
            <span class="speed-tag ${spdClass}">${spd} m/s</span>
            <div class="track-detail-grid">
                <div class="kv"><span class="k">vel lat</span><span class="v">${fmt(t.vel_lat, 6)}</span></div>
                <div class="kv"><span class="k">vel lon</span><span class="v">${fmt(t.vel_lon, 6)}</span></div>
                <div class="kv"><span class="k">updated</span><span class="v">${trackTime}</span></div>
            </div>
            <div class="track-sources">Sources: ${sources}</div>
        </div>`;
    });

    // Prune stale map layers
    Object.keys(objectMarkers).forEach(id => {
        if (!seenIds.has(id)) {
            map.removeLayer(objectMarkers[id]);
            map.removeLayer(velocityLines[id]);
            delete objectMarkers[id];
            delete velocityLines[id];
        }
    });

    container.innerHTML = html || '<div class="empty-state">No active tracks.</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────

loadCameras();
loadStats();
updateTracks();

setInterval(updateTracks, 500);
setInterval(loadCameras, 5_000);
setInterval(loadStats,   5_000);
