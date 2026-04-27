// app.js — logic for the Sauron Tracking Dashboard

const MAP_CENTER = [12.881, 77.551]; // Bangalore — midpoint between cameras

const ZOOM_LEVEL = 16;

const map = L.map('map').setView(MAP_CENTER, ZOOM_LEVEL);

// Use CartoDB Dark Matter tiles
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// State
let cameras = {};
let objectMarkers = {}; // object_id -> L.Marker
let velocityVectors = {}; // object_id -> L.Polyline

// ---------------------------------------------------------
// Helper: Draw FOV Cone
// ---------------------------------------------------------
function drawFOVCone(lat, lon, bearing, fov, radiusMeters = 500) {
    const earthRadius = 6378137.0; // meters
    const latRad = lat * Math.PI / 180;
    
    const points = [[lat, lon]]; // origin
    
    const startAngle = bearing - (fov / 2);
    const endAngle = bearing + (fov / 2);
    const numSegments = 20;
    
    for (let i = 0; i <= numSegments; i++) {
        const angle = startAngle + (i / numSegments) * (endAngle - startAngle);
        const angleRad = angle * Math.PI / 180;
        
        // Calculate destination point
        const dy = radiusMeters * Math.cos(angleRad);
        const dx = radiusMeters * Math.sin(angleRad);
        
        const dLat = (dy / earthRadius) * (180 / Math.PI);
        const dLon = (dx / (earthRadius * Math.cos(latRad))) * (180 / Math.PI);
        
        points.push([lat + dLat, lon + dLon]);
    }
    
    L.polygon(points, {
        color: '#ffaa00',
        fillColor: '#ffaa00',
        fillOpacity: 0.1,
        weight: 1,
        dashArray: '5, 5'
    }).addTo(map);
}

// ---------------------------------------------------------
// Fetch Cameras Once
// ---------------------------------------------------------
async function loadCameras() {
    try {
        const res = await fetch('/api/cameras');
        const data = await res.json();
        const validCameras = data.filter(c => Number.isFinite(c.lat) && Number.isFinite(c.lon));
        
        if (validCameras.length > 0) {
            // Fit map to show all cameras with some padding
            const lats = validCameras.map(c => c.lat);
            const lons = validCameras.map(c => c.lon);
            const bounds = L.latLngBounds(
                [Math.min(...lats) - 0.002, Math.min(...lons) - 0.002],
                [Math.max(...lats) + 0.002, Math.max(...lons) + 0.002]
            );
            map.fitBounds(bounds);
        }

        validCameras.forEach(cam => {
            cameras[cam.device_id] = cam;
            const heading = Number.isFinite(cam.last_heading)
                ? cam.last_heading
                : (Number.isFinite(cam.bearing_deg) ? cam.bearing_deg : 0);
            const pitch = Number.isFinite(cam.last_pitch) ? cam.last_pitch : 0;
            const roll = Number.isFinite(cam.last_roll) ? cam.last_roll : 0;
            const fov = Number.isFinite(cam.fov_deg) ? cam.fov_deg : 60;
            
            // Marker
            L.marker([cam.lat, cam.lon]).addTo(map)
                .bindPopup(
                    `<b>${cam.device_id}</b><br>Heading: ${heading.toFixed(1)}&deg;<br>` +
                    `Pitch: ${pitch.toFixed(1)}&deg;<br>Roll: ${roll.toFixed(1)}&deg;`
                );
            
            // FOV Cone
            drawFOVCone(cam.lat, cam.lon, heading, fov, 250);
        });
    } catch (e) {
        console.error("Failed to load cameras:", e);
    }
}


// ---------------------------------------------------------
// Fetch Tracks Loop
// ---------------------------------------------------------
async function updateTracks() {
    try {
        const res = await fetch('/api/live_tracks');
        const tracks = await res.json();
        
        const container = document.getElementById('tracks-container');
        let html = '';
        
        const seenIds = new Set();

        tracks.forEach(t => {
            seenIds.add(t.object_id);
            const pos = [t.lat, t.lon];
            
            // Velocity projection (line 2 seconds into the future)
            const futurePos = [
                t.lat + (t.vel_lat * 2),
                t.lon + (t.vel_lon * 2)
            ];

            // 1. Update or create Map Marker
            if (!objectMarkers[t.object_id]) {
                const icon = L.divIcon({ className: 'object-marker', iconSize: [12, 12] });
                objectMarkers[t.object_id] = L.marker(pos, { icon }).addTo(map);
                
                velocityVectors[t.object_id] = L.polyline([pos, futurePos], {
                    color: '#00ffcc', weight: 2, opacity: 0.5
                }).addTo(map);
            } else {
                objectMarkers[t.object_id].setLatLng(pos);
                velocityVectors[t.object_id].setLatLngs([pos, futurePos]);
            }
            
            // 2. Update Sidebar
            const speedDegS = Math.sqrt(t.vel_lat**2 + t.vel_lon**2);
            const speedMS = (speedDegS * 111320).toFixed(2); // approx m/s
            
            html += `
                <div class="track-item">
                    <div class="track-id">${t.object_id}</div>
                    <div class="track-stats">
                        Sources: ${t.source_cameras.join(', ')}<br>
                        Speed: ${speedMS} m/s
                    </div>
                </div>
            `;
        });
        
        // Remove stale markers
        Object.keys(objectMarkers).forEach(id => {
            if (!seenIds.has(id)) {
                map.removeLayer(objectMarkers[id]);
                map.removeLayer(velocityVectors[id]);
                delete objectMarkers[id];
                delete velocityVectors[id];
            }
        });
        
        container.innerHTML = html || 'No active tracks.';
        
    } catch (e) {
        console.error("Failed to load tracks:", e);
    }
}

// Start
loadCameras();
setInterval(updateTracks, 500);
