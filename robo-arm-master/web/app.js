// web/app.js
// Frontend Logic for Robotic Arm 3D Autonomy Cockpit

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const cameraStreamImg = document.getElementById("cameraStreamImg");
    const depthStreamImg = document.getElementById("depthStreamImg");
    const viewportContainer = document.getElementById("viewportContainer");
    const cameraPanel = document.getElementById("cameraPanel");
    const depthPanel = document.getElementById("depthPanel");

    const btnShowCamera = document.getElementById("btnShowCamera");
    const btnShowDepth = document.getElementById("btnShowDepth");
    const btnShowDual = document.getElementById("btnShowDual");

    const btnRefreshFeed = document.getElementById("btnRefreshFeed");
    const btnParkSurvey = document.getElementById("btnParkSurvey");
    const btnParkRest = document.getElementById("btnParkRest");
    const btnScanObjects = document.getElementById("btnScanObjects");

    const perceptionTableBody = document.getElementById("perceptionTableBody");
    const chatLog = document.getElementById("chatLog");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");

    const hudX = document.getElementById("hudX");
    const hudY = document.getElementById("hudY");
    const hudZ = document.getElementById("hudZ");
    const hudClaw = document.getElementById("hudClaw");
    const activeScenarioLabel = document.getElementById("activeScenarioLabel");
    const modelPresetSelect = document.getElementById("modelPresetSelect");

    const sliderX = document.getElementById("sliderX");
    const sliderY = document.getElementById("sliderY");
    const sliderZ = document.getElementById("sliderZ");
    const valX = document.getElementById("valX");
    const valY = document.getElementById("valY");
    const valZ = document.getElementById("valZ");
    const btnMoveArm = document.getElementById("btnMoveArm");
    const btnToggleClaw = document.getElementById("btnToggleClaw");

    let clawState = "open";

    // Feed Switching
    function setViewportMode(mode) {
        btnShowCamera.classList.remove("active");
        btnShowDepth.classList.remove("active");
        btnShowDual.classList.remove("active");

        if (mode === "camera") {
            btnShowCamera.classList.add("active");
            viewportContainer.classList.remove("dual-mode");
            cameraPanel.classList.remove("hidden");
            depthPanel.classList.add("hidden");
        } else if (mode === "depth") {
            btnShowDepth.classList.add("active");
            viewportContainer.classList.remove("dual-mode");
            cameraPanel.classList.add("hidden");
            depthPanel.classList.remove("hidden");
        } else if (mode === "dual") {
            btnShowDual.classList.add("active");
            viewportContainer.classList.add("dual-mode");
            cameraPanel.classList.remove("hidden");
            depthPanel.classList.remove("hidden");
        }
    }

    btnShowCamera.addEventListener("click", () => setViewportMode("camera"));
    btnShowDepth.addEventListener("click", () => setViewportMode("depth"));
    btnShowDual.addEventListener("click", () => setViewportMode("dual"));

    // Refresh Camera & Depth Streams
    function refreshFeeds() {
        const ts = Date.now();
        cameraStreamImg.src = `/capture?t=${ts}`;
        depthStreamImg.src = `/capture_depth?t=${ts}`;
    }

    btnRefreshFeed.addEventListener("click", refreshFeeds);

    // Update Telemetry
    async function fetchTelemetry() {
        try {
            const res = await fetch("/telemetry");
            if (res.ok) {
                const data = await res.json();
                hudX.textContent = `${data.end_effector_cm.x} cm`;
                hudY.textContent = `${data.end_effector_cm.y} cm`;
                hudZ.textContent = `${data.end_effector_cm.z} cm`;

                clawState = data.claw_state || "open";
                hudClaw.textContent = clawState.toUpperCase();
                hudClaw.style.color = clawState === "closed" ? "var(--accent-coral)" : "var(--accent-emerald)";

                // Update scenario label
                const scNames = {
                    "sorting": "Standard Sorting",
                    "mars_rover": "Mars Rover Curation",
                    "chemistry_lab": "Chemistry Lab Bench",
                    "manufacturing_plant": "Electronics Plant"
                };
                activeScenarioLabel.textContent = scNames[data.active_scenario] || data.active_scenario;
            }
        } catch (e) {
            console.warn("Telemetry fetch error:", e);
        }
    }

    // Refresh telemetry every 1.5 seconds
    setInterval(fetchTelemetry, 1500);
    fetchTelemetry();

    // Scan Objects & Depth Fusion
    async function scanObjects() {
        try {
            btnScanObjects.textContent = "Scanning...";
            const res = await fetch("/api/scan", { method: "POST" });
            if (res.ok) {
                const data = await res.json();
                renderPerceptionTable(data.objects || []);
                refreshFeeds();
            }
        } catch (err) {
            console.error("Scan error:", err);
        } finally {
            btnScanObjects.textContent = "Run SAM 2.1 & Depth Fusion";
        }
    }

    btnScanObjects.addEventListener("click", scanObjects);

    function renderPerceptionTable(objects) {
        if (!objects || objects.length === 0) {
            perceptionTableBody.innerHTML = `<tr><td colspan="6" class="empty-state">No objects detected.</td></tr>`;
            return;
        }

        perceptionTableBody.innerHTML = "";
        objects.forEach(obj => {
            const c = obj.centroid_cm;
            const dims = obj.dimensions_cm || {};
            const tr = document.createElement("tr");

            tr.innerHTML = `
                <td><span style="color: var(--accent-cyan); font-weight: 700;">#${obj.id}</span></td>
                <td>(${c.x}, ${c.y}, ${c.z || 2.0}) cm</td>
                <td style="color: var(--accent-amber); font-weight: 600;">${obj.surface_top_z_cm || 4.0} cm</td>
                <td>${dims.width_cm || 0} × ${dims.length_cm || 0} × ${obj.height_cm || 0} cm</td>
                <td style="color: var(--accent-emerald); font-weight: 700;">${obj.grasp_recommended_z_cm || 2.5} cm</td>
                <td>
                    <button class="btn btn-small" onclick="quickGrabObject(${obj.id}, ${c.x}, ${c.y}, ${obj.grasp_recommended_z_cm || 2.5})">
                        Target Grasp
                    </button>
                </td>
            `;
            perceptionTableBody.appendChild(tr);
        });
    }

    window.quickGrabObject = async (id, x, y, z) => {
        appendMessage("USER", `Targeting Object #${id} via 3D Depth coordinate: X=${x}, Y=${y}, Z=${z}cm`);
        try {
            const res = await fetch("/execute_tool", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    tool: "move_arm",
                    arguments: { x_cm: x, y_cm: y, z_cm: z }
                })
            });
            const data = await res.json();
            appendMessage("ROBOT", `Action executed: ${data.status.toUpperCase()}`, null, {
                tool: "move_arm",
                args: { x_cm: x, y_cm: y, z_cm: z }
            });
            refreshFeeds();
            fetchTelemetry();
        } catch (e) {
            appendMessage("ERROR", `Failed to move arm: ${e}`);
        }
    };

    // Chat Message Helpers
    function appendMessage(author, body, reasoning = null, tool = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `msg ${author.toLowerCase()}-msg`;

        let html = `<div class="msg-author">${author}</div><div class="msg-body">${body}</div>`;

        if (reasoning) {
            html += `<div class="reasoning-box"><strong>Cognitive Reasoning:</strong> ${reasoning}</div>`;
        }

        if (tool) {
            html += `<div class="tool-chip">⚡ Tool: <strong>${tool.tool || tool.name}</strong> ${JSON.stringify(tool.args || tool.arguments || {})}</div>`;
        }

        msgDiv.innerHTML = html;
        chatLog.appendChild(msgDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    // Handle Chat Submit
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const cmd = chatInput.value.trim();
        if (!cmd) return;

        chatInput.value = "";
        appendMessage("USER", cmd);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ command: cmd })
            });

            if (res.ok) {
                const data = await res.json();
                appendMessage("AGENT", data.response_text || "Step complete.", data.reasoning, data.tool_call);
                if (data.detected_objects) {
                    renderPerceptionTable(data.detected_objects);
                }
            } else {
                appendMessage("ERROR", "Autonomy layer encountered an error processing request.");
            }
        } catch (err) {
            appendMessage("ERROR", `Connection error: ${err}`);
        } finally {
            refreshFeeds();
            fetchTelemetry();
        }
    });

    // Handle Prompt Chips
    document.querySelectorAll(".prompt-chips .chip").forEach(chip => {
        chip.addEventListener("click", () => {
            chatInput.value = chip.getAttribute("data-prompt");
            chatForm.dispatchEvent(new Event("submit"));
        });
    });

    // Scenario Switching
    document.querySelectorAll(".scenario-pill").forEach(pill => {
        pill.addEventListener("click", async () => {
            const sc = pill.getAttribute("data-scenario");
            document.querySelectorAll(".scenario-pill").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");

            appendMessage("SYSTEM", `Switching cognitive scenario environment to <strong>${sc}</strong>...`);
            try {
                const res = await fetch("/switch_scenario", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ scenario: sc })
                });
                if (res.ok) {
                    appendMessage("SYSTEM", `Environment loaded: <strong>${sc}</strong>.`);
                    refreshFeeds();
                    fetchTelemetry();
                    scanObjects();
                }
            } catch (err) {
                appendMessage("ERROR", `Failed to switch scenario: ${err}`);
            }
        });
    });

    // Park Buttons
    btnParkSurvey.addEventListener("click", async () => {
        appendMessage("USER", "Park arm in survey pose to clear camera frustum.");
        await fetch("/execute_tool", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tool: "park_arm", arguments: { pose: "survey" } })
        });
        refreshFeeds();
        fetchTelemetry();
    });

    btnParkRest.addEventListener("click", async () => {
        appendMessage("USER", "Fold arm into compact rest position.");
        await fetch("/execute_tool", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tool: "park_arm", arguments: { pose: "rest" } })
        });
        refreshFeeds();
        fetchTelemetry();
    });

    // Manual Jog Sliders
    sliderX.addEventListener("input", (e) => valX.textContent = parseFloat(e.target.value).toFixed(1));
    sliderY.addEventListener("input", (e) => valY.textContent = parseFloat(e.target.value).toFixed(1));
    sliderZ.addEventListener("input", (e) => valZ.textContent = parseFloat(e.target.value).toFixed(1));

    btnMoveArm.addEventListener("click", async () => {
        const x = parseFloat(sliderX.value);
        const y = parseFloat(sliderY.value);
        const z = parseFloat(sliderZ.value);

        await fetch("/api/manual_move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "move", x: x, y: y, z: z })
        });
        refreshFeeds();
        fetchTelemetry();
    });

    btnToggleClaw.addEventListener("click", async () => {
        const nextState = clawState === "closed" ? "open" : "closed";
        await fetch("/api/manual_move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "claw", state: nextState })
        });
        refreshFeeds();
        fetchTelemetry();
    });

    // Initial scan
    setTimeout(() => {
        scanObjects();
    }, 1000);
});
