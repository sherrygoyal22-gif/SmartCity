(() => {
    "use strict";

    const $ = (selector) => document.querySelector(selector);

    /* =========================
       SIDEBAR
    ========================= */

    const sidebar = $("#sidebar");

    $("[data-menu]")?.addEventListener("click", () => {
        sidebar?.classList.toggle("open");
    });


    /* =========================
       DARK / LIGHT THEME
    ========================= */

    const theme = document.documentElement;
    const savedTheme = localStorage.getItem("smartcity-theme");

    if (savedTheme) {
        theme.dataset.theme = savedTheme;
    }

    $("[data-theme]")?.addEventListener("click", () => {
        const newTheme =
            theme.dataset.theme === "dark" ? "light" : "dark";

        theme.dataset.theme = newTheme;
        localStorage.setItem("smartcity-theme", newTheme);
    });


    /* =========================
       TOAST CLOSE
    ========================= */

    document.querySelectorAll(".toast button").forEach((button) => {
        button.addEventListener("click", () => {
            button.parentElement?.remove();
        });
    });


    /* =========================
       CONFIRMATION
    ========================= */

    document.querySelectorAll("[data-confirm]").forEach((form) => {
        form.addEventListener("submit", (event) => {
            const message = form.dataset.confirm;

            if (message && !confirm(message)) {
                event.preventDefault();
            }
        });
    });


    /* =========================
       DETECTION TREND CHART
    ========================= */

    const chart = $("#trendChart");

    if (chart) {
        const labels = window.trendData || [];
        const values = window.trendValues || [];

        if (!values.length) {
            chart.hidden = true;

            const empty = $("#chartEmpty");

            if (empty) {
                empty.style.display = "grid";
            }
        } else {
            const context = chart.getContext("2d");

            const width = chart.clientWidth;
            const height = 195;

            chart.width = width * window.devicePixelRatio;
            chart.height = height * window.devicePixelRatio;

            context.scale(
                window.devicePixelRatio,
                window.devicePixelRatio
            );

            const max = Math.max(...values, 1);
            const padding = 30;

            context.strokeStyle = "#dce5ec";
            context.lineWidth = 1;

            context.beginPath();
            context.moveTo(padding, 160);
            context.lineTo(width - 5, 160);
            context.stroke();

            context.strokeStyle = "#149b57";
            context.lineWidth = 3;

            context.beginPath();

            values.forEach((value, index) => {
                const x =
                    padding +
                    index *
                        ((width - padding - 10) /
                            Math.max(values.length - 1, 1));

                const y =
                    160 -
                    (value / max) * 130;

                if (index === 0) {
                    context.moveTo(x, y);
                } else {
                    context.lineTo(x, y);
                }
            });

            context.stroke();

            context.fillStyle = "#6e7c91";
            context.font = "11px Arial";

            labels.forEach((label, index) => {
                if (
                    index === 0 ||
                    index === labels.length - 1
                ) {
                    const x =
                        padding +
                        index *
                            ((width - padding - 10) /
                                Math.max(labels.length - 1, 1));

                    context.fillText(
                        label.slice(5),
                        x - 12,
                        184
                    );
                }
            });
        }
    }


    /* =========================
       IMAGE PREVIEW
    ========================= */

    const imageInput = $("#image");
    const imagePreview = $("#imagePreview");
    const dropZone = $(".drop-zone");

    if (imageInput) {

        const showPreview = (file) => {
            if (!file) {
                return;
            }

            if (!file.type.startsWith("image/")) {
                return;
            }

            if (imagePreview) {
                imagePreview.src =
                    URL.createObjectURL(file);

                imagePreview.hidden = false;
            }
        };

        imageInput.addEventListener("change", () => {
            showPreview(imageInput.files[0]);
        });

        if (dropZone) {

            ["dragenter", "dragover"].forEach((eventName) => {
                dropZone.addEventListener(
                    eventName,
                    (event) => {
                        event.preventDefault();
                        dropZone.classList.add("dragover");
                    }
                );
            });

            ["dragleave", "drop"].forEach((eventName) => {
                dropZone.addEventListener(
                    eventName,
                    (event) => {
                        event.preventDefault();
                        dropZone.classList.remove("dragover");
                    }
                );
            });

            dropZone.addEventListener("drop", (event) => {
                const files = event.dataTransfer.files;

                if (files.length) {
                    imageInput.files = files;
                    showPreview(files[0]);
                }
            });
        }
    }


    /* =========================
       CONFIDENCE SLIDER
    ========================= */

    $("#confidence")?.addEventListener("input", (event) => {
        const output = $("#confidenceOutput");

        if (output) {
            output.textContent =
                Math.round(event.target.value * 100) + "%";
        }
    });


    /* =========================
       UPLOAD FORM
    ========================= */

    $("#uploadForm")?.addEventListener("submit", () => {
        const button = document.querySelector(
            '#uploadForm button[type="submit"]'
        );

        if (button) {
            button.disabled = true;
            button.textContent = "Analyzing image…";
        }

        if (!document.getElementById("busyOverlay")) {
            const overlay = document.createElement("div");
            overlay.id = "busyOverlay";
            overlay.setAttribute("role", "status");
            overlay.innerHTML = '<div class="busy-box"><div class="busy-spinner"></div><b>Analysing your photo…</b><span>Please keep this screen open.</span></div>';
            document.body.appendChild(overlay);
        }
    });


    /* =====================================================
       AUTOMATIC GPS
       ===================================================== */

    const latitudeInput = $("#latitude");
    const longitudeInput = $("#longitude");
    const locationStatus = $("#locationStatus");

    function updateLocationStatus(message) {
        if (locationStatus) {
            locationStatus.textContent = message;
        }
    }

    function getAutomaticLocation() {

        if (!latitudeInput || !longitudeInput) {
            return;
        }

        if (!navigator.geolocation) {
            updateLocationStatus(
                "GPS is not supported by this browser."
            );
            return;
        }

        updateLocationStatus(
            "📍 Detecting current location..."
        );

        const useFix = (position, precise) => {
            latitudeInput.value = position.coords.latitude;
            longitudeInput.value = position.coords.longitude;
            updateLocationStatus(
                precise ? "📍 Location detected (GPS)." : "📍 Location detected."
            );
        };

        // 1) quick network/Wi-Fi fix so detection is never kept waiting for satellites,
        // 2) then a precise GPS fix that replaces it when it arrives.
        navigator.geolocation.getCurrentPosition(
            (position) => {
                useFix(position, false);
                navigator.geolocation.getCurrentPosition(
                    (precise) => useFix(precise, true),
                    () => {},
                    { enableHighAccuracy: true, timeout: 20000, maximumAge: 0 }
                );
            },
            (error) => {
                console.warn("GPS quick fix failed:", error.message);
                navigator.geolocation.getCurrentPosition(
                    (precise) => useFix(precise, true),
                    (err) => {
                        console.warn("GPS error:", err.message);
                        updateLocationStatus("⚠️ Location unavailable. Detection can continue.");
                    },
                    { enableHighAccuracy: true, timeout: 15000, maximumAge: 30000 }
                );
            },
            { enableHighAccuracy: false, timeout: 8000, maximumAge: 120000 }
        );
    }

    /*
       Automatically request GPS when the detection page loads.
       The user does not enter the location manually.
    */
    getAutomaticLocation();


    /* =====================================================
       CAMERA
       ===================================================== */

    let stream = null;

    const video = $("#cameraVideo");
    const startCamera = $("#startCamera");
    const stopCamera = $("#stopCamera");
    const captureCamera = $("#captureCamera");
    const cameraFallback = $("#cameraFallback");
    const cameraStatus = $("#cameraStatus");

    startCamera?.addEventListener(
        "click",
        async () => {

            if (!navigator.mediaDevices?.getUserMedia) {

                if (cameraStatus) {
                    cameraStatus.textContent =
                        "Camera is unavailable in this browser.";
                }

                return;
            }

            try {

                startCamera.disabled = true;
                if (cameraStatus) {
                    cameraStatus.textContent = "Starting camera… allow camera access if asked.";
                }

                const withTimeout = (promise, ms, message) => Promise.race([
                    promise,
                    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms))
                ]);

                try {
                    stream = await withTimeout(
                        navigator.mediaDevices.getUserMedia({
                            video: {
                                facingMode: { ideal: "environment" },
                                width: { ideal: 1280 },
                                height: { ideal: 720 }
                            },
                            audio: false
                        }),
                        20000,
                        "Camera did not respond."
                    );
                } catch (firstError) {
                    // Some phones reject the size/facing hints: retry with the simplest request.
                    if (firstError && firstError.name === "NotAllowedError") {
                        throw firstError;
                    }
                    stream = await withTimeout(
                        navigator.mediaDevices.getUserMedia({ video: true, audio: false }),
                        20000,
                        "Camera did not respond."
                    );
                }

                video.muted = true;
                video.setAttribute("playsinline", "");
                video.srcObject = stream;
                video.hidden = false;

                if (cameraFallback) {
                    cameraFallback.hidden = true;
                }

                await withTimeout(video.play(), 10000, "Camera preview did not start.");

                startCamera.disabled = true;
                stopCamera.disabled = false;
                captureCamera.disabled = false;

                if (cameraStatus) {
                    cameraStatus.textContent =
                        "Camera ready.";
                }

                /*
                   Refresh GPS when camera starts.
                */
                getAutomaticLocation();

            } catch (error) {

                console.error(error);

                stream?.getTracks().forEach((track) => track.stop());
                stream = null;
                startCamera.disabled = false;

                if (cameraStatus) {
                    cameraStatus.textContent =
                        (error && error.name === "NotAllowedError")
                            ? "Camera permission was denied. Allow camera access in the browser/site settings, or use “Take photo with phone camera” below."
                            : "Camera could not start (" + ((error && error.message) || "unknown error") + "). Use “Take photo with phone camera” below instead.";
                }
            }
        }
    );


    /* =========================
       STOP CAMERA
    ========================= */

    stopCamera?.addEventListener(
        "click",
        () => {

            stream?.getTracks().forEach(
                (track) => track.stop()
            );

            stream = null;

            if (video) {
                video.srcObject = null;
                video.hidden = true;
            }

            if (cameraFallback) {
                cameraFallback.hidden = false;
            }

            startCamera.disabled = false;
            stopCamera.disabled = true;
            captureCamera.disabled = true;

            if (cameraStatus) {
                cameraStatus.textContent =
                    "Camera stopped.";
            }
        }
    );


    /* =========================
       CAPTURE CAMERA IMAGE
    ========================= */

    captureCamera?.addEventListener(
        "click",
        async () => {

            try {

                if (!video.videoWidth || !video.videoHeight) {
                    throw new Error("Camera is not ready yet. Wait a second and try again.");
                }

                // Take a short burst and keep only the sharpest frames. A phone camera
                // needs a moment to focus and hand shake blurs single frames, so picking
                // the sharpest few is what lets small / far-away garbage be seen.
                const BURST_FRAMES = 6;
                const KEEP_FRAMES = 3;
                const scale = Math.min(1, 1024 / Math.max(video.videoWidth, video.videoHeight));
                const canvas = $("#cameraCanvas");
                canvas.width = Math.round(video.videoWidth * scale);
                canvas.height = Math.round(video.videoHeight * scale);

                const context =
                    canvas.getContext("2d");

                // Small scratch canvas used only to measure how sharp a frame is.
                const probe = document.createElement("canvas");
                const probeContext = probe.getContext("2d", { willReadFrequently: true });
                const sharpnessOf = (source) => {
                    const pw = 160;
                    const ph = Math.max(8, Math.round(pw * source.height / source.width));
                    probe.width = pw;
                    probe.height = ph;
                    probeContext.drawImage(source, 0, 0, pw, ph);
                    const px = probeContext.getImageData(0, 0, pw, ph).data;
                    const gray = new Float32Array(pw * ph);
                    for (let i = 0; i < gray.length; i += 1) {
                        gray[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
                    }
                    let sum = 0;
                    let sumSquares = 0;
                    let count = 0;
                    for (let y = 1; y < ph - 1; y += 1) {
                        for (let x = 1; x < pw - 1; x += 1) {
                            const i = y * pw + x;
                            const lap = gray[i - 1] + gray[i + 1] + gray[i - pw] + gray[i + pw] - 4 * gray[i];
                            sum += lap;
                            sumSquares += lap * lap;
                            count += 1;
                        }
                    }
                    const mean = sum / count;
                    return sumSquares / count - mean * mean;
                };

                // Forget the previous answer so an old result never sits under a new one.
                const previousResult = $("#cameraResult");
                if (previousResult) {
                    previousResult.hidden = true;
                    previousResult.innerHTML = "";
                }

                if (cameraStatus) {
                    cameraStatus.textContent =
                        "Hold steady… capturing (1/" + BURST_FRAMES + ")";
                }

                captureCamera.disabled = true;

                /*
                   Make sure GPS is available before
                   sending the detection.
                */
                if (
                    !latitudeInput.value ||
                    !longitudeInput.value
                ) {
                    getAutomaticLocation();
                }

                const captured = [];
                for (let index = 0; index < BURST_FRAMES; index += 1) {
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const score = sharpnessOf(canvas);
                    const blob = await new Promise((resolve) => {
                        canvas.toBlob(resolve, "image/jpeg", 0.88);
                    });
                    if (!blob) {
                        throw new Error("Camera frame could not be captured.");
                    }
                    captured.push({ blob, score });
                    if (cameraStatus) {
                        cameraStatus.textContent =
                            "Hold steady… capturing (" + Math.min(index + 2, BURST_FRAMES) + "/" + BURST_FRAMES + ")";
                    }
                    if (index < BURST_FRAMES - 1) {
                        await new Promise((resolve) => setTimeout(resolve, 200));
                    }
                }

                captured.sort((a, b) => b.score - a.score);
                const frames = captured.slice(0, KEEP_FRAMES).map((item) => item.blob);

                const formData =
                    new FormData();

                frames.forEach((frame, index) => {
                    formData.append("validation_frames", frame, `camera-${index}.jpg`);
                });

                formData.append(
                    "confidence",
                    $("#confidence")?.value || "0.25"
                );

                /*
                   Automatically attached GPS.
                */
                formData.append(
                    "latitude",
                    latitudeInput?.value || ""
                );

                formData.append(
                    "longitude",
                    longitudeInput?.value || ""
                );

                formData.append(
                    "address",
                    $("#address")?.value || ""
                );

                if (cameraStatus) {
                    cameraStatus.textContent =
                        "Analysing… please keep this screen open.";
                }

                const controller = new AbortController();
                const abortTimer = setTimeout(() => controller.abort(), 120000);
                let response;
                try {
                    response = await fetch(
                        "/api/camera-detect",
                        {
                            method: "POST",
                            body: formData,
                            signal: controller.signal
                        }
                    );
                } finally {
                    clearTimeout(abortTimer);
                }

                let data;
                try {
                    data = await response.json();
                } catch (parseError) {
                    data = {
                        ok: false,
                        error: response.status === 401 || response.redirected
                            ? "Your session expired. Log in again."
                            : response.status === 413
                                ? "The photo is too large to upload."
                                : "The server returned an unexpected response (" + response.status + "). Try again in a moment."
                    };
                }

                captureCamera.disabled = false;

                if (!data.ok) {

                    if (cameraStatus) {
                        cameraStatus.textContent =
                            data.error ||
                            "Detection failed.";
                    }

                    return;
                }

                if (cameraStatus) {
                    cameraStatus.textContent =
                        "✅ Detection saved with GPS location.";
                }

                const result =
                    $("#cameraResult");

                if (result) {

                    result.hidden = false;

                    result.innerHTML = `
                        <h2>Camera detection complete</h2>

                        <p>
                            <strong>
                                ${data.count}
                            </strong>
                            garbage item(s) detected
                        </p>

                        <p>
                            Average confidence:
                            ${
                                data.confidence === null
                                    ? "—"
                                    : data.confidence.toFixed(1) + "%"
                            }
                        </p>

                        <p>
                            Validation: ${data.validation_reason || "—"}
                        </p>

                        <p>
                            Risk: ${data.risk_level || "—"}${data.risk_score === null ? "" : ` (${data.risk_score}/100)`}
                        </p>

                        ${data.is_duplicate ? `<p>Repeated location: ${data.duplicate_reason || "Yes"}</p>` : ""}

                        <p>
                            📍 GPS:
                            ${
                                data.latitude !== null &&
                                data.longitude !== null
                                    ? `${data.latitude}, ${data.longitude}`
                                    : "Unavailable"
                            }
                        </p>

                        <img
                            class="camera-result"
                            src="${data.result_url}"
                            alt="AI garbage detection result"
                        >

                        ${data.case_id ? `<p><strong>Case #${data.case_id}</strong> created (PENDING).</p>` : `<p>No case created for this detection.</p>`}

                        <p>
                            ${data.case_url ? `<a class="button" href="${data.case_url}">View case</a> ` : ""}
                            <a
                                class="${data.case_url ? "button secondary" : "button"}"
                                href="${data.report_url}"
                            >
                                View saved report
                            </a>
                        </p>
                    `;
                }

            } catch (error) {

                console.error(error);

                captureCamera.disabled = false;

                if (cameraStatus) {
                    cameraStatus.textContent =
                        (error && error.name === "AbortError")
                            ? "The server took too long to answer. Check your connection and try again, or use “Take photo with phone camera”."
                            : ((error && error.message) || "An error occurred while sending the detection.");
                }
            }
        }
    );


    /* =====================================================
       LOCATION MAP
       ===================================================== */

    const mapElement = $("#map");

    if (mapElement && typeof L === "undefined") {
        // Map library did not load (offline / blocked): show the empty state instead of breaking the page.
        const emptyBox = $("#mapEmpty");
        if (emptyBox) {
            emptyBox.hidden = false;
            emptyBox.innerHTML = "<b>Map could not load</b><p>Check your internet connection and reload the page.</p>";
        }
    } else if (mapElement) {

        const records = JSON.parse(mapElement.dataset.records || "[]");
        const hotspots = JSON.parse(mapElement.dataset.hotspots || "[]");
        const mapEmpty = $("#mapEmpty");
        const filterEmpty = $("#mapFilterEmpty");
        const statusFilter = $("#mapStatusFilter");
        const riskFilter = $("#mapRiskFilter");
        const searchFilter = $("#mapSearchFilter");
        const applyFilters = $("#mapApplyFilters");
        const locationStatus = $("#mapLocationStatus");

        const adminLatRaw = mapElement.dataset.adminLat;
        const adminLngRaw = mapElement.dataset.adminLng;
        const adminLat = adminLatRaw ? parseFloat(adminLatRaw) : null;
        const adminLng = adminLngRaw ? parseFloat(adminLngRaw) : null;
        const hasAdminLocation = Number.isFinite(adminLat) && Number.isFinite(adminLng);

        const escapeHtml = (value) => String(value || "").replace(/[&<>'"]/g, (character) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
        }[character]));

        // Real geographic distance (never screen/pixel distance).
        const haversineKm = (lat1, lng1, lat2, lng2) => {
            const toRad = (value) => (value * Math.PI) / 180;
            const earthRadiusKm = 6371;
            const dLat = toRad(lat2 - lat1);
            const dLng = toRad(lng2 - lng1);
            const a =
                Math.sin(dLat / 2) ** 2 +
                Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
            return earthRadiusKm * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        };
        const formatDistanceKm = (km) => (km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`);

        // Real compass bearing from one real coordinate to another (never
        // confused with lat/lng — this is the direction of travel, in
        // degrees clockwise from true north, converted to a compass point).
        const bearingDeg = (lat1, lng1, lat2, lng2) => {
            const toRad = (value) => (value * Math.PI) / 180;
            const toDeg = (value) => (value * 180) / Math.PI;
            const y = Math.sin(toRad(lng2 - lng1)) * Math.cos(toRad(lat2));
            const x =
                Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
                Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(toRad(lng2 - lng1));
            return (toDeg(Math.atan2(y, x)) + 360) % 360;
        };
        const COMPASS_POINTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
        const COMPASS_LABELS = {
            N: "North", NE: "North-East", E: "East", SE: "South-East",
            S: "South", SW: "South-West", W: "West", NW: "North-West"
        };
        const compassPoint = (deg) => COMPASS_POINTS[Math.round(deg / 45) % 8];

        // A Google Maps turn-by-turn link is the real, mobile-friendly
        // "Navigate" action (opens the device's own maps app on a phone).
        // Straight-line Haversine distance is never claimed to be this
        // route's distance or ETA — the popups label it "straight-line".
        const directionsUrl = (lat, lng) => {
            const destination = `${lat},${lng}`;
            const params = new URLSearchParams({
                api: "1",
                destination,
                travelmode: "driving"
            });
            // No fixed "origin" is set — Google Maps fills that in from
            // whatever device opens the link, so this never depends on
            // this page having the visitor's own location.
            return `https://www.google.com/maps/dir/?${params.toString()}`;
        };

        // Detection timestamps are stored in UTC; render them the same way
        // the server's `india_datetime` filter does, without a page reload.
        const formatDetectedAt = (value) => {
            if (!value) return "—";
            const parsed = new Date(`${value.replace(" ", "T")}Z`);
            if (Number.isNaN(parsed.getTime())) return escapeHtml(value);
            try {
                return `${new Intl.DateTimeFormat("en-IN", {
                    day: "2-digit", month: "short", year: "numeric",
                    hour: "2-digit", minute: "2-digit", hour12: true,
                    timeZone: "Asia/Kolkata"
                }).format(parsed)} IST`;
            } catch (error) {
                return escapeHtml(value);
            }
        };

        if (!records.length && !hasAdminLocation) {
            mapElement.hidden = true;
            if (mapEmpty) mapEmpty.hidden = false;
            if (locationStatus) locationStatus.hidden = true;
        } else {
            const firstRecord = records[0];
            const initialView = firstRecord
                ? [firstRecord.lat, firstRecord.lng]
                : [adminLat, adminLng];
            // minZoom is a hard floor: even if a report's GPS coordinate is
            // badly wrong (a phone's IP-based location fallback landing
            // hundreds of km away, say), fitBounds() below can never zoom
            // the whole map out far enough to show that alongside the real
            // local reports — which is what used to make the map "fly off"
            // to a mostly-empty, zoomed-out view after a popup opened.
            const map = L.map("map", { minZoom: 8 }).setView(initialView, 13);
            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                maxZoom: 19,
                attribution: "© OpenStreetMap contributors"
            }).addTo(map);
            const markers = L.layerGroup().addTo(map);
            const clusterLayer = L.layerGroup().addTo(map);
            const peopleLayer = L.layerGroup().addTo(map);
            const lineLayer = L.layerGroup().addTo(map);
            const markerById = {};
            let activeDistanceLine = null;

            const PIN_SVG =
                '<svg viewBox="0 0 24 32" fill="currentColor" xmlns="http://www.w3.org/2000/svg">' +
                '<path d="M12 0C5.9 0 1 4.9 1 11c0 8.3 11 21 11 21s11-12.7 11-21C23 4.9 18.1 0 12 0z"/>' +
                '<circle cx="12" cy="11" r="4.6" fill="#fff"/></svg>';

            const markerClass = (risk) => {
                const value = (risk || "LOW").toLowerCase();
                return ["high", "medium", "low"].includes(value) ? value : "low";
            };

            const pinIcon = (risk) => L.divIcon({
                className: "map-marker-wrap",
                html: `<span class="gps-pin ${markerClass(risk)}">${PIN_SVG}</span>`,
                iconSize: [30, 40],
                iconAnchor: [15, 40],
                popupAnchor: [0, -36]
            });

            const dotIcon = (kind) => L.divIcon({
                className: "map-marker-wrap",
                html: `<span class="gps-dot ${kind}"></span>`,
                iconSize: [18, 18],
                iconAnchor: [9, 9]
            });

            const clusterIcon = (count, risk) => L.divIcon({
                className: "map-marker-wrap",
                html: `<span class="gps-cluster ${markerClass(risk)}" style="width:${Math.min(30 + count * 4, 56)}px;height:${Math.min(30 + count * 4, 56)}px">${count}</span>`,
                iconSize: [1, 1]
            });

            const sourceLabel = (source) => (source === "camera" ? "Live Camera" : "Upload");

            // A short "2.4 km NE" style label — used both inside popups and
            // on the floating dashed-line tooltip drawn between the Admin
            // Location and whichever marker's popup is currently open.
            const shortDistanceLabel = (lat1, lng1, lat2, lng2) => {
                const km = haversineKm(lat1, lng1, lat2, lng2);
                const direction = compassPoint(bearingDeg(lat1, lng1, lat2, lng2));
                return `${formatDistanceKm(km)} ${direction}`;
            };

            // Draws (or replaces) a dashed line from the fixed Admin
            // Location to whatever marker's popup is currently open, with a
            // floating distance/direction tooltip at the midpoint — purely
            // visual, never affects the underlying report/hotspot layers.
            const showDistanceLine = (lat, lng) => {
                clearDistanceLine();
                if (!hasAdminLocation) return;
                const midLat = (adminLat + lat) / 2;
                const midLng = (adminLng + lng) / 2;
                activeDistanceLine = L.polyline([[adminLat, adminLng], [lat, lng]], {
                    color: "#0f172a", weight: 2, opacity: 0.85, dashArray: "6,7"
                }).addTo(lineLayer);
                L.marker([midLat, midLng], {
                    icon: L.divIcon({
                        className: "map-marker-wrap",
                        html: `<span class="dist-tag">${shortDistanceLabel(adminLat, adminLng, lat, lng)}</span>`,
                        iconSize: [1, 1]
                    }),
                    interactive: false
                }).addTo(lineLayer);
            };
            const clearDistanceLine = () => { lineLayer.clearLayers(); activeDistanceLine = null; };

            // Distance + direction is computed fresh every time a popup
            // opens (see the function-based bindPopup calls below), always
            // measured from the one fixed Admin Location — the map never
            // asks the visitor's browser for its own GPS position.
            const distanceGrid = (lat, lng) => {
                if (!hasAdminLocation) {
                    return `<div class="map-popup-distgrid"><div class="map-popup-dist-cell full"><span class="map-popup-dist-label">🏢 Distance from admin</span><span class="map-popup-dist-value muted">Admin location not configured</span></div></div>`;
                }
                const direction = compassPoint(bearingDeg(adminLat, adminLng, lat, lng));
                return `<div class="map-popup-distgrid">` +
                    `<div class="map-popup-dist-cell full">` +
                    `<span class="map-popup-dist-label">🏢 Distance from admin office</span>` +
                    `<span class="map-popup-dist-value">${formatDistanceKm(haversineKm(adminLat, adminLng, lat, lng))} (${direction})</span>` +
                    `</div><div class="map-popup-dir">➤ ${COMPASS_LABELS[direction]} of admin office</div>` +
                    `</div>`;
            };

            const popup = (record) => {
                const actions = [];
                if (record.report_url) {
                    actions.push(`<a class="map-popup-btn primary" href="${record.report_url}">View Report</a>`);
                }
                if (record.case_url) {
                    actions.push(`<a class="map-popup-btn case" href="${record.case_url}">View Case</a>`);
                }
                actions.push(
                    `<a class="map-popup-btn navigate" target="_blank" rel="noopener" href="${directionsUrl(record.lat, record.lng)}">Get Directions</a>`
                );
                const photo = record.result_path
                    ? `<div class="map-popup-photo-wrap"><img class="map-popup-photo" src="/static/${record.result_path}" alt="Report #${record.detection_id} photo"></div>`
                    : "";
                return `<div class="map-popup rich">` +
                    photo +
                    `<div class="map-popup-body">` +
                    `<div class="map-popup-headrow">` +
                    `<span class="map-popup-title">Report #${record.detection_id}</span>` +
                    `<span class="map-popup-badge ${markerClass(record.risk_level)}">${escapeHtml(record.risk_level || "NOT ASSESSED")} RISK</span>` +
                    `</div>` +
                    `${record.case_id ? `<span class="map-popup-case">Case #${record.case_id}</span>` : ""}` +
                    `<div class="map-popup-fields">` +
                    `<span class="map-popup-row"><b>Garbage detected:</b> ${record.count ? `${record.count} item${record.count === 1 ? "" : "s"}` : "—"}</span>` +
                    `<span class="map-popup-row"><b>Confidence:</b> ${record.confidence === null || record.confidence === undefined ? "—" : record.confidence.toFixed(1) + "%"}</span>` +
                    `<span class="map-popup-row"><b>Source:</b> ${escapeHtml(sourceLabel(record.source))}</span>` +
                    `<span class="map-popup-row"><b>Date &amp; Time:</b> ${formatDetectedAt(record.created_at)}</span>` +
                    `${record.case_status ? `<span class="map-popup-row"><b>Case status:</b> ${escapeHtml(record.case_status)}</span>` : ""}` +
                    `${record.assignee_name ? `<span class="map-popup-row"><b>Assigned to:</b> ${escapeHtml(record.assignee_name)}</span>` : ""}` +
                    `${record.hotspot_count ? `<span class="map-popup-row"><b>Hotspot:</b> ${record.hotspot_count} nearby reports</span>` : ""}` +
                    `</div>` +
                    `<div class="map-popup-locrow">` +
                    `<span class="map-popup-loc-label">📍 Location</span>` +
                    `<span class="map-popup-loc-value">${escapeHtml(record.address || `${record.lat.toFixed(5)}, ${record.lng.toFixed(5)}`)}</span>` +
                    `<a class="map-popup-viewmap" href="#" data-view-lat="${record.lat}" data-view-lng="${record.lng}">View on Map</a>` +
                    `</div>` +
                    distanceGrid(record.lat, record.lng) +
                    `<div class="map-popup-actions">${actions.join("")}</div>` +
                    `</div></div>`;
            };

            const hotspotPopup = (hotspot) => {
                const idList = hotspot.report_ids || [];
                const idPreview = idList.slice(0, 6).map((id) => `#${id}`).join(", ") +
                    (idList.length > 6 ? ` +${idList.length - 6} more` : "");
                const caseList = hotspot.case_ids || [];
                const casePreview = caseList.length
                    ? caseList.slice(0, 6).map((id) => `#${id}`).join(", ") + (caseList.length > 6 ? ` +${caseList.length - 6} more` : "")
                    : "None yet";
                const sourcesPreview = (hotspot.sources || []).map(sourceLabel).join(", ") || "—";
                const actions = [];
                if (hotspot.report_url) {
                    actions.push(`<a class="map-popup-btn primary" href="${hotspot.report_url}">View Latest Report</a>`);
                }
                if (hotspot.history_url) {
                    actions.push(`<a class="map-popup-btn case" href="${hotspot.history_url}">View Hotspot Reports</a>`);
                }
                actions.push(
                    `<a class="map-popup-btn navigate" target="_blank" rel="noopener" href="${directionsUrl(hotspot.latitude, hotspot.longitude)}">🧭 Get Directions</a>`
                );
                return `<div class="map-popup rich hotspot-popup">` +
                    `<div class="map-popup-body">` +
                    `<span class="map-popup-badge ${markerClass(hotspot.highest_risk)}">${escapeHtml(hotspot.highest_risk || "LOW")} RISK HOTSPOT</span>` +
                    `<span class="map-popup-title">${escapeHtml(hotspot.area_label || "Hotspot Area")}</span>` +
                    `<span class="map-popup-row">${hotspot.detection_count} reports</span>` +
                    `<ul class="map-popup-breakdown">` +
                    `<li><i class="risk-dot high"></i>High Risk: ${hotspot.high_count || 0}</li>` +
                    `<li><i class="risk-dot medium"></i>Medium Risk: ${hotspot.medium_count || 0}</li>` +
                    `<li><i class="risk-dot low"></i>Low Risk: ${hotspot.low_count || 0}</li>` +
                    `</ul>` +
                    `<span class="map-popup-row"><b>Reports:</b> ${idPreview}</span>` +
                    `<span class="map-popup-row"><b>Cases:</b> ${casePreview}</span>` +
                    `<span class="map-popup-row"><b>Sources:</b> ${escapeHtml(sourcesPreview)}</span>` +
                    `<span class="map-popup-row"><b>Latest detection:</b> ${formatDetectedAt(hotspot.latest_detection)}</span>` +
                    `<span class="map-popup-row"><b>Active cases:</b> ${hotspot.active_cases} · <b>Resolved:</b> ${hotspot.completed_cases}</span>` +
                    (hasAdminLocation
                        ? `<span class="map-popup-row link">Distance from admin office: ${shortDistanceLabel(adminLat, adminLng, hotspot.latitude, hotspot.longitude)}</span>`
                        : "") +
                    `<div class="map-popup-actions">${actions.join("")}</div>` +
                    `</div></div>`;
            };

            // The admin marker is the map's one fixed reference point — it
            // never moves at runtime, so it is created once. Every other
            // marker's popup reports its distance and direction from here.
            const ensureAdminMarker = () => {
                if (!hasAdminLocation) return;
                const marker = L.marker([adminLat, adminLng], { icon: dotIcon("admin"), zIndexOffset: 999 })
                    .bindTooltip("Admin Location", { permanent: true, direction: "top", offset: [0, -6], className: "map-place-tag admin-tag" })
                    .bindPopup(() =>
                        `<div class="map-popup">` +
                        `<span class="map-popup-title">Admin / Municipal Office</span>` +
                        `<span class="map-popup-row">Coordinates: ${adminLat.toFixed(5)}, ${adminLng.toFixed(5)}</span>` +
                        `<span class="map-popup-row">All distances on this map are measured from this office.</span>` +
                        `<div class="map-popup-actions"><a class="map-popup-btn navigate" target="_blank" rel="noopener" href="${directionsUrl(adminLat, adminLng)}">Get Directions</a></div>` +
                        `</div>`
                    )
                    .addTo(peopleLayer);
                marker.on("popupclose", clearDistanceLine);
            };

            // Report pins + hotspot bubbles are (re)built only when filters
            // change — GPS updates never touch this function, so a phone
            // moving around does not re-render the whole marker set.
            // Report details card. Deliberately NOT a Leaflet popup: popups
            // live inside the moving map layer, so panning, auto-pan and
            // re-renders kept closing or displacing them. This card is a
            // fixed panel over the map that never moves, and stays open until
            // its X is pressed (or Esc), another marker is clicked, or the
            // filters change.
            const cardHost = mapElement.closest(".map-canvas-wrap") || mapElement.parentElement;
            cardHost.classList.add("has-info-card-host");
            const infoCard = document.createElement("div");
            infoCard.className = "map-info-card";
            infoCard.hidden = true;
            infoCard.setAttribute("role", "dialog");
            infoCard.setAttribute("aria-label", "Report details");
            cardHost.appendChild(infoCard);
            const closeInfoCard = () => {
                infoCard.hidden = true;
                infoCard.innerHTML = "";
                clearDistanceLine();
            };
            const openInfoCard = (html, lat, lng) => {
                infoCard.innerHTML = `<button type="button" class="map-info-card-close" aria-label="Close details">✕</button>${html}`;
                infoCard.hidden = false;
                infoCard.scrollTop = 0;
                showDistanceLine(lat, lng);
            };
            infoCard.addEventListener("click", (event) => {
                if (event.target.closest(".map-info-card-close")) {
                    closeInfoCard();
                    return;
                }
                const link = event.target.closest(".map-popup-viewmap");
                if (!link) return;
                event.preventDefault();
                const lat = parseFloat(link.dataset.viewLat);
                const lng = parseFloat(link.dataset.viewLng);
                if (Number.isFinite(lat) && Number.isFinite(lng)) map.flyTo([lat, lng], 18);
            });
            document.addEventListener("keydown", (event) => {
                if (event.key === "Escape" && !infoCard.hidden) closeInfoCard();
            });

            const renderMarkers = () => {
                const status = statusFilter?.value || "";
                const risk = riskFilter?.value || "";
                const search = (searchFilter?.value || "").trim().toLowerCase();
                const visible = records.filter((record) =>
                    (!status || record.case_status === status) &&
                    (!risk || record.risk_level === risk) &&
                    (!search || [record.address, record.detection_id, record.case_id]
                        .filter((value) => value !== null && value !== undefined)
                        .some((value) => String(value).toLowerCase().includes(search)))
                );
                closeInfoCard();
                markers.clearLayers();
                clusterLayer.clearLayers();
                Object.keys(markerById).forEach((key) => delete markerById[key]);
                const bounds = [];
                visible.forEach((record) => {
                    const marker = L.marker([record.lat, record.lng], { icon: pinIcon(record.risk_level), bubblingMouseEvents: false })
                        .on("click", () => openInfoCard(popup(record), record.lat, record.lng));
                    markers.addLayer(marker);
                    markerById[record.detection_id] = marker;
                    bounds.push([record.lat, record.lng]);
                });
                // Hotspot clusters (2+ nearby reports) drawn as numbered bubbles
                // above the individual pins. Clicking one simply opens its
                // popup in place - no zoom or re-centering - so the map never
                // moves away from the thing that was just clicked.
                (hotspots || []).forEach((hotspot) => {
                    const bubble = L.marker([hotspot.latitude, hotspot.longitude], {
                        icon: clusterIcon(hotspot.detection_count, hotspot.highest_risk),
                        zIndexOffset: 500,
                        bubblingMouseEvents: false
                    }).on("click", () => openInfoCard(hotspotPopup(hotspot), hotspot.latitude, hotspot.longitude));
                    clusterLayer.addLayer(bubble);
                });
                if (filterEmpty) filterEmpty.hidden = visible.length > 0;
                // Every real report still gets a pin above — this only
                // decides what the *auto-fit view* optimizes for. A single
                // report with a wildly wrong GPS reading (bad phone fix,
                // IP-based fallback, etc.) used to be included here too,
                // forcing fitBounds() to zoom the whole shared map out to
                // whatever huge, mostly-empty region contained both that
                // outlier and the real local reports. Reference distance is
                // the admin office when known, otherwise the median of the
                // points themselves, so one bad coordinate can no longer
                // hijack the view for every visitor.
                const median = (values) => {
                    const sorted = [...values].sort((a, b) => a - b);
                    const mid = Math.floor(sorted.length / 2);
                    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
                };
                const refLat = hasAdminLocation ? adminLat : median(bounds.map((point) => point[0]));
                const refLng = hasAdminLocation ? adminLng : median(bounds.map((point) => point[1]));
                const fitBounds = bounds.length > 1
                    ? bounds.filter(([lat, lng]) => haversineKm(refLat, refLng, lat, lng) <= 150)
                    : bounds;
                const boundsToUse = fitBounds.length ? fitBounds : bounds;
                if (boundsToUse.length > 1) {
                    map.fitBounds(boundsToUse, { padding: [25, 25] });
                } else if (boundsToUse.length === 1) {
                    map.setView(boundsToUse[0], 14);
                } else if (hasAdminLocation) {
                    map.setView([adminLat, adminLng], 13);
                }
            };

            // "Admin Office" control — recenters on the one fixed reference
            // point every popup's distance/direction is measured from.
            // Doesn't touch or request the visitor's own location at all.
            const AdminLocateControl = L.Control.extend({
                options: { position: "topleft" },
                onAdd() {
                    const button = L.DomUtil.create("button", "leaflet-bar map-locate-control");
                    button.type = "button";
                    button.title = "Admin Office";
                    button.setAttribute("aria-label", "Center map on the admin office");
                    button.innerHTML = "🏢";
                    L.DomEvent.disableClickPropagation(button);
                    button.addEventListener("click", () => {
                        if (!hasAdminLocation) return;
                        map.flyTo([adminLat, adminLng], Math.max(map.getZoom(), 14));
                    });
                    return button;
                }
            });
            if (hasAdminLocation) map.addControl(new AdminLocateControl());

            // Full-screen map toggle. Deliberately not the browser's native
            // Fullscreen API (Element.requestFullscreen) — support for that
            // is inconsistent on mobile Safari, which is where people using
            // this app to snap a photo of garbage are most likely to be. A
            // plain CSS "cover the whole viewport" overlay works identically
            // everywhere and needs no permission prompt.
            const mapShell = mapElement.closest(".map-canvas-wrap") || mapElement.parentElement;
            let exitButton = null;
            const onEscKey = (event) => {
                if (event.key === "Escape") setMapFullscreen(false);
            };
            const setMapFullscreen = (on) => {
                mapShell.classList.toggle("is-map-fullscreen", on);
                document.body.classList.toggle("map-fullscreen-open", on);
                if (fullscreenButton) {
                    fullscreenButton.title = on ? "Exit full screen" : "View full screen";
                    fullscreenButton.setAttribute("aria-label", fullscreenButton.title);
                    fullscreenButton.innerHTML = on ? "🗗" : "⛶";
                }
                if (on && !exitButton) {
                    exitButton = document.createElement("button");
                    exitButton.type = "button";
                    exitButton.className = "map-fullscreen-exit";
                    exitButton.innerHTML = "✕ Exit Full Screen";
                    exitButton.setAttribute("aria-label", "Exit full screen map");
                    exitButton.addEventListener("click", () => setMapFullscreen(false));
                    mapShell.appendChild(exitButton);
                } else if (!on && exitButton) {
                    exitButton.remove();
                    exitButton = null;
                }
                if (on) {
                    document.addEventListener("keydown", onEscKey);
                } else {
                    document.removeEventListener("keydown", onEscKey);
                }
                // The map card resizes with the viewport when it goes full
                // screen; Leaflet needs to be told so its tiles fill the new
                // size instead of leaving stale, wrongly-cropped tiles.
                setTimeout(() => map.invalidateSize(), 60);
            };
            let fullscreenButton = null;
            const FullscreenControl = L.Control.extend({
                options: { position: "topright" },
                onAdd() {
                    fullscreenButton = L.DomUtil.create("button", "leaflet-bar map-locate-control");
                    fullscreenButton.type = "button";
                    fullscreenButton.title = "View full screen";
                    fullscreenButton.setAttribute("aria-label", "View full screen");
                    fullscreenButton.innerHTML = "⛶";
                    L.DomEvent.disableClickPropagation(fullscreenButton);
                    fullscreenButton.addEventListener("click", () => {
                        setMapFullscreen(!mapShell.classList.contains("is-map-fullscreen"));
                    });
                    return fullscreenButton;
                }
            });
            map.addControl(new FullscreenControl());
            // Text link in the page heading (Hotspot Map page) that does the
            // same thing as the ⛶ button on the map. Still a real link to
            // /map?fullscreen=1, so it also works if JavaScript is blocked.
            const fullscreenLink = document.getElementById("mapFullscreenLink");
            if (fullscreenLink) {
                fullscreenLink.addEventListener("click", (event) => {
                    event.preventDefault();
                    setMapFullscreen(true);
                });
            }

            // Lets any link elsewhere in the app (e.g. the dashboard's
            // "View Map →" link) send the visitor straight into full-screen
            // mode by adding ?fullscreen=1, instead of needing a second click
            // once they land on this page.
            if (new URLSearchParams(window.location.search).get("fullscreen") === "1") {
                setTimeout(() => setMapFullscreen(true), 0);
            }

            // The "View on Map" link inside a popup zooms in tight on that
            // exact real coordinate without leaving the page.
            map.getContainer().addEventListener("click", (event) => {
                const link = event.target.closest(".map-popup-viewmap");
                if (!link) return;
                event.preventDefault();
                const lat = parseFloat(link.dataset.viewLat);
                const lng = parseFloat(link.dataset.viewLng);
                if (Number.isFinite(lat) && Number.isFinite(lng)) {
                    map.flyTo([lat, lng], 18);
                }
            });

            statusFilter?.addEventListener("change", renderMarkers);
            riskFilter?.addEventListener("change", renderMarkers);
            searchFilter?.addEventListener("input", renderMarkers);
            applyFilters?.addEventListener("click", renderMarkers);
            renderMarkers();
            ensureAdminMarker();
            map.invalidateSize();

            // Clicking a sidebar report (or its 📍 "View on map" link) just
            // re-centers the map on its real saved coordinates — it no
            // longer also pops open that marker's popup box, since that was
            // an unwanted extra message appearing on top of the map every
            // time someone just wanted to jump to a location. The "View
            // Report →" / "View →" link inside each row still does the
            // real navigation to that report's page.
            const goToCoordinates = (lat, lng) => {
                if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
                map.setView([lat, lng], 17);
            };
            document.querySelectorAll(".hotspot-row[data-goto-lat][data-goto-lng]").forEach((row) => {
                const goTo = () => goToCoordinates(parseFloat(row.dataset.gotoLat), parseFloat(row.dataset.gotoLng));
                row.addEventListener("click", (event) => {
                    if (event.target.closest("a")) return; // let a real link (View Report, View on map) handle its own click
                    goTo();
                });
                row.addEventListener("keydown", (event) => {
                    if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        goTo();
                    }
                });
            });
            document.querySelectorAll(".hotspot-view-map[data-goto-lat][data-goto-lng]").forEach((link) => {
                link.addEventListener("click", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    goToCoordinates(parseFloat(link.dataset.gotoLat), parseFloat(link.dataset.gotoLng));
                });
            });

            // This map never asks for the visitor's own GPS position —
            // every report's distance and direction is measured from the
            // fixed admin office instead, so it works the same for every
            // visitor without a location permission prompt.
            if (locationStatus) {
                locationStatus.textContent = hasAdminLocation
                    ? "Distances shown below are measured from the Admin Office."
                    : "Admin location not configured — distances are unavailable.";
            }
        }
    }

})();

/* Shrink big phone photos in the browser before uploading (a 6 MB photo becomes ~300 KB),
   then submit. Used by the file picker and by the "Take photo with phone camera" button. */
document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("uploadForm");
    const fileInput = document.getElementById("image");
    const nativeInput = document.getElementById("nativeCapture");
    if (!uploadForm || !fileInput) return;

    const status = () => document.getElementById("cameraStatus");
    let pending = null;

    const downscale = (file) => new Promise((resolve) => {
        const okType = ["image/jpeg", "image/png", "image/webp"].includes(file && file.type);
        if (!file || !file.type.startsWith("image/") || (okType && file.size < 1_200_000)) { resolve(file); return; }
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            try {
                const scale = Math.min(1, 1600 / Math.max(img.naturalWidth, img.naturalHeight));
                const canvas = document.createElement("canvas");
                canvas.width = Math.round(img.naturalWidth * scale);
                canvas.height = Math.round(img.naturalHeight * scale);
                canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
                canvas.toBlob((blob) => {
                    URL.revokeObjectURL(url);
                    if (!blob || (okType && blob.size >= file.size)) { resolve(file); return; }
                    resolve(new File([blob], (file.name || "photo").replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" }));
                }, "image/jpeg", 0.85);
            } catch (error) { URL.revokeObjectURL(url); resolve(file); }
        };
        img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
        img.src = url;
    });

    const setFile = (file) => {
        const transfer = new DataTransfer();
        transfer.items.add(file);
        fileInput.files = transfer.files;
    };

    fileInput.addEventListener("change", () => {
        const file = fileInput.files && fileInput.files[0];
        if (!file) return;
        pending = downscale(file).then((small) => {
            if (small !== file) { try { setFile(small); } catch (error) { /* keep original */ } }
        }).finally(() => { pending = null; });
    });

    // If the user taps submit while a photo is still being shrunk, wait for it.
    uploadForm.addEventListener("submit", (event) => {
        if (!pending) return;
        event.preventDefault();
        pending.then(() => (uploadForm.requestSubmit ? uploadForm.requestSubmit() : uploadForm.submit()));
    });

    if (nativeInput) {
        nativeInput.addEventListener("change", async () => {
            const file = nativeInput.files && nativeInput.files[0];
            if (!file) return;
            if (status()) status().textContent = "Photo taken. Preparing…";
            const small = await downscale(file);
            try {
                setFile(small);
                fileInput.dispatchEvent(new Event("input", { bubbles: true }));
            } catch (error) { return; }
            if (status()) status().textContent = "Uploading and analysing…";
            if (uploadForm.requestSubmit) uploadForm.requestSubmit(); else uploadForm.submit();
        });
    }
});
