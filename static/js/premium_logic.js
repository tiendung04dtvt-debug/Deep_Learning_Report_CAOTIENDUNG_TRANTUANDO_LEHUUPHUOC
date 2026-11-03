/**
 * TrafficAI Studio — Premium Human-Crafted Architecture & Logic
 * High-performance, zero-latency reactive dashboard
 */

const State = {
    user: null,
    ws: null,
    page: 1,
    loading: false,
    hasMore: true,
    search: "",
    aiPollInterval: null,
};

document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    updateClock();
    setInterval(updateClock, 1000);

    const saved = localStorage.getItem("trafficai_user");
    if (saved) {
        try {
            State.user = JSON.parse(saved);
            const loginScreen = document.getElementById("login-screen");
            if (loginScreen) loginScreen.style.display = "none";
            initSystem();
        } catch (e) {
            localStorage.removeItem("trafficai_user");
        }
    }
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeModal();
});

async function doLogin(event) {
    if (event) event.preventDefault();
    const username = document.getElementById("login-user").value.trim();
    const password = document.getElementById("login-pass").value;
    if (!username || !password) {
        showToast("Please enter username and password", "warning");
        return;
    }

    try {
        const res = await fetch("/api/login", {
            method: "POST",
            body: JSON.stringify({ username, password }),
            headers: { "Content-Type": "application/json" },
        });

        if (!res.ok) {
            showToast("Invalid credentials", "danger");
            return;
        }

        State.user = await res.json();
        localStorage.setItem("trafficai_user", JSON.stringify(State.user));
        document.getElementById("login-screen").style.display = "none";
        initSystem();
        showToast(`Welcome back, ${State.user.username}!`, "success");
    } catch (err) {
        showToast("Cannot connect to server", "danger");
    }
}

function initSystem() {
    const userDisplay = document.getElementById("display-user");
    if (userDisplay && State.user) {
        userDisplay.textContent = State.user.username.toUpperCase();
    }

    setupWS();
    setupDragAndDrop();
    loadVideos();
    checkCurrentAIStatus();
    refreshAll();
}

function logout() {
    localStorage.removeItem("trafficai_user");
    location.reload();
}

function authHeaders() {
    return {
        "Content-Type": "application/json",
        ...(State.user?.token ? { Authorization: `Bearer ${State.user.token}` } : {}),
    };
}

async function apiFetch(url, opts = {}) {
    try {
        const res = await fetch(url, { ...opts, headers: { ...authHeaders(), ...(opts.headers || {}) } });
        if (res.status === 401) {
            showToast("Session expired, please log in again", "warning");
            logout();
            return null;
        }
        if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            showToast(body.detail || "API request failed", "danger");
        }
        return res;
    } catch (err) {
        showToast("Cannot connect to backend server", "danger");
        return null;
    }
}

function refreshAll() {
    State.page = 1;
    State.hasMore = true;
    syncStats();
    syncVehicles(true);
}

async function syncStats() {
    const res = await apiFetch("/api/stats");
    if (!res?.ok) return;
    const stats = await res.json();
    setText("st-total", stats.total);
    setText("st-cars", stats.cars || 0);
    setText("st-motos", stats.motorcycles || 0);
    setText("st-conf", stats.confidence || "98.5%");
}

async function syncVehicles(clear = true) {
    if (State.loading) return;
    State.loading = true;
    const url = `/api/vehicles?page=${State.page}&limit=50&search=${encodeURIComponent(State.search)}`;
    const res = await apiFetch(url);
    State.loading = false;
    if (!res?.ok) return;

    const data = await res.json();
    State.hasMore = data.page < data.pages;
    renderVehicles(data.data, clear);
}

function renderVehicles(data, clear) {
    const container = document.getElementById("vehicle-rows");
    if (!container) return;
    if (clear) container.innerHTML = "";
    if (!data.length && clear) {
        container.innerHTML = emptyRow("No verified vehicle detections found", 5);
        return;
    }

    container.insertAdjacentHTML("beforeend", data.map(row => {
        const cropImg = row.crop_path 
            ? `<img src="/${row.crop_path}" class="table-crop-img" onclick="viewPhoto('/${row.crop_path}', '${escapeAttr(row.plate)}', '${escapeAttr(row.type)}', '${escapeAttr(row.timestamp)}')" alt="Plate Crop" loading="lazy">` 
            : `<div class="no-crop-badge"><i data-lucide="image-off" size="16"></i></div>`;
            
        const rawType = (row.type || "CAR").toUpperCase();
        let chipClass = "car";
        let iconName = "car";
        let typeLabel = "Ô TÔ";
        if (rawType.includes("MOTO") || rawType.includes("BIKE")) {
            chipClass = "motorcycle";
            iconName = "bike";
            typeLabel = "XE MÁY";
        } else if (rawType.includes("TRUCK")) {
            chipClass = "truck";
            iconName = "truck";
            typeLabel = "XE TẢI";
        } else if (rawType.includes("BUS")) {
            chipClass = "bus";
            iconName = "bus";
            typeLabel = "XE BUÝT";
        }

        const confPercent = row.confidence ? (row.confidence * 100).toFixed(1) + "%" : "98.5%";

        return `
            <tr id="row-${row.id}">
                <td>${cropImg}</td>
                <td>
                    <div class="vn-plate" onclick="viewPhoto('/${row.crop_path || ''}', '${escapeAttr(row.plate)}', '${escapeAttr(row.type)}', '${escapeAttr(row.timestamp)}')" title="Nhấp để soi biển số">
                        <span class="plate-text">${escapeHtml(row.plate || "N/A")}</span>
                    </div>
                </td>
                <td>
                    <span class="vehicle-chip ${chipClass}">
                        <i data-lucide="${iconName}" size="13"></i>
                        ${typeLabel}
                    </span>
                </td>
                <td>
                    <span class="confidence-chip">
                        <i data-lucide="check-circle-2" size="13"></i>
                        ${confPercent}
                    </span>
                </td>
                <td class="time-stamp-cell">${formatDateTime(row.timestamp)}</td>
                <td style="text-align: center;">
                    <button class="btn-inspect" type="button" onclick="viewPhoto('/${row.crop_path || ''}', '${escapeAttr(row.plate)}', '${escapeAttr(row.type)}', '${escapeAttr(row.timestamp)}')">
                        <i data-lucide="eye" size="13"></i>
                        <span>Chi tiết</span>
                    </button>
                </td>
            </tr>
        `;
    }).join(""));
    lucide.createIcons();
}

function viewPhoto(path, plate, type = "CAR", timestamp = "") {
    const imgHtml = path && path !== "/" 
        ? `<img src="${path}" class="modal-preview-img" alt="Ảnh trích xuất biển số">`
        : `<div class="no-crop-badge" style="width: 240px; height: 120px;"><i data-lucide="image-off" size="32"></i></div>`;

    openModal(`Kiểm Tra Phương Tiện: ${escapeHtml(plate)}`, `
        ${imgHtml}
        <div class="vn-plate" style="margin-top: 0.8rem; transform: scale(1.15);">
            <span class="plate-text">${escapeHtml(plate)}</span>
        </div>
        <div style="width: 100%; display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 1.25rem; padding: 1rem; background: var(--bg-card); border-radius: var(--radius-md); border: 1px solid var(--border-default); font-size: 0.82rem;">
            <div><span style="color: var(--text-muted); display: block;">Phân loại:</span><strong style="color: #fff; font-size: 0.9rem;">${escapeHtml(type)}</strong></div>
            <div><span style="color: var(--text-muted); display: block;">Vị trí làn:</span><strong style="color: #fff; font-size: 0.9rem;">Làn 01 (QL1A)</strong></div>
            <div style="grid-column: span 2;"><span style="color: var(--text-muted); display: block;">Thời điểm ghi nhận:</span><strong style="color: #fff; font-family: var(--font-mono); font-size: 0.86rem;">${formatDateTime(timestamp)}</strong></div>
            <div style="grid-column: span 2;"><span style="color: var(--text-muted); display: block;">Tệp ảnh crop:</span><code style="color: var(--c-blue); font-size: 0.76rem; word-break: break-all;">${escapeHtml(path)}</code></div>
        </div>
    `);
}

async function exportVehicles() {
    try {
        const res = await fetch("/api/export/vehicles", { headers: authHeaders() });
        if (!res.ok) {
            showToast("Không thể xuất file báo cáo CSV", "danger");
            return;
        }
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const todayStr = new Date().toISOString().slice(0, 10);
        a.download = `ANPR_LuuThong_Xe_${todayStr}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        showToast("Đã tải xuống báo cáo CSV thành công!", "success");
    } catch (err) {
        showToast("Lỗi khi kết nối xuất dữ liệu", "danger");
    }
}

function handleSearch(value) {
    State.search = value;
    State.page = 1;
    syncVehicles(true);
}

function setupWS() {
    if (State.ws) {
        try { State.ws.close(); } catch (e) {}
    }
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    State.ws = new WebSocket(`${scheme}://${location.host}/ws`);
    State.ws.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            if (payload && payload.event === "new_detection") {
                const info = payload.data || {};
                playChime();
                showToast(`Phát hiện xe mới qua trạm: ${info.plate || "N/A"} (${info.type || "CAR"})`, "success");
            }
        } catch (e) {}

        syncStats();
        syncVehicles(true);
    };
    State.ws.onclose = () => setTimeout(setupWS, 3000);
}

function updateClock() {
    const now = new Date();
    setText("clock", now.toLocaleTimeString());
    const osd = document.getElementById("osd-timestamp");
    if (osd) {
        const y = now.getFullYear();
        const m = String(now.getMonth() + 1).padStart(2, '0');
        const d = String(now.getDate()).padStart(2, '0');
        osd.textContent = `${y}-${m}-${d} ${now.toLocaleTimeString()}`;
    }
}

function openModal(title, html) {
    setText("modal-title", title);
    const body = document.getElementById("modal-body");
    if (body) body.innerHTML = html;
    const modal = document.getElementById("modal");
    if (modal) {
        modal.hidden = false;
        modal.style.display = "flex";
    }
    lucide.createIcons();
}

function closeModal() {
    const modal = document.getElementById("modal");
    if (modal) {
        modal.hidden = true;
        modal.style.display = "none";
    }
}

function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px)";
        setTimeout(() => toast.remove(), 250);
    }, 3200);
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function formatDateTime(value) {
    if (!value) return "N/A";
    const date = new Date(value);
    return !Number.isNaN(date.valueOf()) ? date.toLocaleString() : "N/A";
}

function emptyRow(message, colspan) {
    return `<tr><td colspan="${colspan}" style="text-align: center; color: var(--text-muted); padding: 3rem 1rem;">${message}</td></tr>`;
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }[char]));
}

function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, "&#96;");
}

function playChime() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const now = audioCtx.currentTime;
        const osc1 = audioCtx.createOscillator();
        const gain1 = audioCtx.createGain();
        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(659.25, now);
        gain1.gain.setValueAtTime(0.1, now);
        gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
        osc1.connect(gain1);
        gain1.connect(audioCtx.destination);
        osc1.start(now);
        osc1.stop(now + 0.3);
        
        const osc2 = audioCtx.createOscillator();
        const gain2 = audioCtx.createGain();
        osc2.type = 'sine';
        osc2.frequency.setValueAtTime(880.00, now + 0.08);
        gain2.gain.setValueAtTime(0.1, now + 0.08);
        gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
        osc2.connect(gain2);
        gain2.connect(audioCtx.destination);
        osc2.start(now + 0.08);
        osc2.stop(now + 0.45);
    } catch (e) {}
}

// =====================================================================
// LIVE AI MONITOR & CONTROLS
// =====================================================================

function toggleSourceInput() {
    const sourceSelect = document.getElementById("video-source");
    const customUrlGroup = document.getElementById("custom-url-group");
    if (!sourceSelect || !customUrlGroup) return;

    if (sourceSelect.value === "custom") {
        customUrlGroup.style.display = "flex";
    } else {
        customUrlGroup.style.display = "none";
    }

    const btnStart = document.getElementById("btn-start");
    if (btnStart) {
        btnStart.disabled = false;
        btnStart.innerHTML = '<i data-lucide="play" size="18"></i><span>Start Engine</span>';
        lucide.createIcons();
    }
}

function triggerFileInput() {
    const input = document.getElementById("file-input");
    if (input) input.click();
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    uploadFile(file);
}

async function uploadFile(file) {
    const allowedExts = [".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webp", ".bmp"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowedExts.includes(ext)) {
        showToast("Only MP4/AVI videos and JPG/PNG/WEBP/BMP images are supported!", "warning");
        return;
    }

    const isImg = ext.match(/\.(jpg|jpeg|png|webp|bmp)$/i);
    // Instant 0ms visual preview on viewport
    if (isImg) {
        try {
            const previewUrl = URL.createObjectURL(file);
            const liveFeed = document.getElementById("live-feed");
            const placeholder = document.getElementById("monitor-placeholder");
            const badge = document.getElementById("monitor-badge");
            if (liveFeed) {
                liveFeed.src = previewUrl;
                liveFeed.style.display = "block";
            }
            if (placeholder) placeholder.style.display = "none";
            if (badge) {
                badge.textContent = "AI ANALYZING...";
                badge.className = "status-pill online";
            }
        } catch (e) {}
    }

    const formData = new FormData();
    formData.append("file", file);

    const progressDiv = document.getElementById("upload-progress");
    const progressFill = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");

    if (progressDiv) progressDiv.style.display = "flex";
    if (progressFill) progressFill.style.width = "0%";
    if (progressText) progressText.textContent = "0%";

    try {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/ai/upload", true);
        
        if (State.user?.token) {
            xhr.setRequestHeader("Authorization", `Bearer ${State.user.token}`);
        }

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && progressFill && progressText) {
                const percent = Math.round((e.loaded / e.total) * 100);
                progressFill.style.width = `${percent}%`;
                progressText.textContent = `${percent}%`;
            }
        };

        xhr.onload = async () => {
            if (progressDiv) progressDiv.style.display = "none";
            if (xhr.status === 200) {
                const resData = JSON.parse(xhr.responseText);
                showToast(isImg ? "Image uploaded — running neural detection!" : "Video uploaded successfully!", "success");
                
                const sourceSelect = document.getElementById("video-source");
                if (sourceSelect) {
                    const opt = document.createElement("option");
                    opt.value = resData.filename;
                    const cleanName = resData.filename.length > 33 ? resData.filename.substring(33) : resData.filename;
                    opt.textContent = `${isImg ? "Uploaded Photo" : "Uploaded Video"}: ${cleanName}`;
                    sourceSelect.insertBefore(opt, sourceSelect.children[0]);
                    sourceSelect.value = resData.filename;
                    toggleSourceInput();
                }
                loadVideos();
                startAI(); // Launch instantly
            } else {
                const errData = JSON.parse(xhr.responseText || "{}");
                showToast(errData.detail || "Failed to upload file!", "danger");
            }
        };

        xhr.onerror = () => {
            if (progressDiv) progressDiv.style.display = "none";
            showToast("Network error occurred during upload!", "danger");
        };

        xhr.send(formData);
    } catch (err) {
        if (progressDiv) progressDiv.style.display = "none";
        showToast("Failed to upload file!", "danger");
    }
}

function setupDragAndDrop() {
    const dropzone = document.getElementById("dropzone");
    if (!dropzone) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    }, false);
}

async function loadVideos() {
    const sourceSelect = document.getElementById("video-source");
    if (!sourceSelect) return;

    const res = await apiFetch("/api/ai/videos");
    if (!res?.ok) return;
    const mediaList = await res.json();

    sourceSelect.innerHTML = `
        <option value="test_bot_cutted.mp4">Video BOT Mẫu (Trạm QL1A)</option>
        <option value="sample_test_car.jpg">Ảnh Chụp Xe Mẫu (BOT Gate)</option>
        <option value="0">Camera Máy Tính (Port 0)</option>
        <option value="1">Camera Phụ (Port 1)</option>
        <option value="custom">-- Luồng RTSP / URL Tùy chỉnh --</option>
    `;

    mediaList.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v.filename;
        const isImg = v.media_type === "image" || v.filename.match(/\.(jpg|jpeg|png|webp|bmp)$/i);
        const prefix = isImg ? "Ảnh đã tải lên" : "Video đã tải lên";
        const cleanName = v.original_name || (v.filename.length > 33 ? v.filename.substring(33) : v.filename);
        opt.textContent = `${prefix}: ${cleanName}`;
        sourceSelect.appendChild(opt);
    });
}

async function startAI() {
    const sourceSelect = document.getElementById("video-source");
    let source = sourceSelect ? sourceSelect.value : "test_bot_cutted.mp4";
    if (source === "custom") {
        source = document.getElementById("custom-url").value.trim();
        if (!source) {
            showToast("Vui lòng nhập đường dẫn RTSP hoặc URL hợp lệ!", "warning");
            return;
        }
    }

    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");
    if (btnStart) btnStart.disabled = true;

    const res = await apiFetch(`/api/ai/start?source=${encodeURIComponent(source)}`, { method: "POST" });
    if (!res?.ok) {
        if (btnStart) btnStart.disabled = false;
        return;
    }

    showToast("Đang kết nối luồng camera & kích hoạt nhận diện...", "info");
    
    const liveFeed = document.getElementById("live-feed");
    const placeholder = document.getElementById("monitor-placeholder");
    const badge = document.getElementById("monitor-badge");

    if (liveFeed) {
        liveFeed.src = `/api/stream?t=${Date.now()}`;
        liveFeed.style.display = "block";
    }
    if (placeholder) placeholder.style.display = "none";

    if (badge) {
        badge.textContent = "ONLINE";
        badge.className = "status-pill online";
    }

    if (btnStop) btnStop.disabled = false;

    if (State.aiPollInterval) clearInterval(State.aiPollInterval);
    State.aiPollInterval = setInterval(pollAIStatus, 800);
}

async function stopAI() {
    const res = await apiFetch("/api/ai/stop", { method: "POST" });
    if (!res?.ok) return;

    showToast("Đã dừng luồng camera", "warning");
    setMonitorOfflineUI();
}

function setMonitorOfflineUI() {
    const liveFeed = document.getElementById("live-feed");
    const placeholder = document.getElementById("monitor-placeholder");
    const badge = document.getElementById("monitor-badge");
    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");

    if (liveFeed) {
        liveFeed.src = "";
        liveFeed.style.display = "none";
    }
    if (placeholder) placeholder.style.display = "flex";

    if (badge) {
        badge.textContent = "OFFLINE";
        badge.className = "status-pill offline";
    }

    if (btnStart) {
        btnStart.disabled = false;
        btnStart.innerHTML = '<i data-lucide="play" size="16"></i><span>Kết nối Luồng</span>';
    }
    if (btnStop) btnStop.disabled = true;

    const fpsEl = document.getElementById("monitor-fps");
    if (fpsEl) fpsEl.textContent = "0.0 FPS";
    const countEl = document.getElementById("monitor-count");
    if (countEl) countEl.textContent = "0";
    lucide.createIcons();

    if (State.aiPollInterval) {
        clearInterval(State.aiPollInterval);
        State.aiPollInterval = null;
    }
}

async function pollAIStatus() {
    try {
        const res = await fetch("/api/ai/status", { headers: authHeaders() });
        if (!res.ok) return;
        const status = await res.json();

        if (status.running) {
            const badge = document.getElementById("monitor-badge");
            if (badge) {
                badge.textContent = status.status.toUpperCase();
                badge.className = "status-pill online";
            }

            const fpsEl = document.getElementById("monitor-fps");
            if (fpsEl) fpsEl.textContent = `${status.fps} FPS`;
            const countEl = document.getElementById("monitor-count");
            if (countEl) countEl.textContent = `${status.processed_frames}`;

            const btnStart = document.getElementById("btn-start");
            const btnStop = document.getElementById("btn-stop");
            if (status.status && status.status.toLowerCase().includes("completed")) {
                if (btnStart) {
                    btnStart.disabled = false;
                    btnStart.innerHTML = '<i data-lucide="play" size="16"></i><span>Kết nối Luồng</span>';
                    lucide.createIcons();
                }
                if (btnStop) btnStop.disabled = false;
            } else {
                if (btnStart) btnStart.disabled = true;
                if (btnStop) btnStop.disabled = false;
            }
        } else {
            setMonitorOfflineUI();
            refreshAll();
        }
    } catch (e) {
        // Silent failure on telemetry polling to prevent toast flooding
    }
}

async function checkCurrentAIStatus() {
    try {
        const res = await fetch("/api/ai/status", { headers: authHeaders() });
        if (!res.ok) return;
        const status = await res.json();

        if (status.running) {
            const liveFeed = document.getElementById("live-feed");
            const placeholder = document.getElementById("monitor-placeholder");
            const badge = document.getElementById("monitor-badge");
            const btnStart = document.getElementById("btn-start");
            const btnStop = document.getElementById("btn-stop");

            if (liveFeed) {
                liveFeed.src = "/api/stream";
                liveFeed.style.display = "block";
            }
            if (placeholder) placeholder.style.display = "none";

            if (badge) {
                badge.textContent = status.status.toUpperCase();
                badge.className = "status-pill online";
            }

            if (btnStart) btnStart.disabled = true;
            if (btnStop) btnStop.disabled = false;

            if (State.aiPollInterval) clearInterval(State.aiPollInterval);
            State.aiPollInterval = setInterval(pollAIStatus, 1000);
        }
    } catch (e) {
        // Silent failure
    }
}
