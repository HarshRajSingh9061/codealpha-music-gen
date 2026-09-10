/* MusicAI — CodeAlpha Task 3 | app.js */
"use strict";

let pollTimer = null;
let lossHistory = [];
let accHistory  = [];

// ── Poll status every 2s ─────────────────────────
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollStatus, 2000);
}

async function pollStatus() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    updateUI(d);
  } catch {}
}

// ── Update entire UI from status ─────────────────
function updateUI(d) {
  const state = d.state || "idle";

  // Status pill
  const dot   = document.getElementById("statusDot");
  const label = document.getElementById("statusLabel");
  const busy  = ["generating_midi","preprocessing","training","generating"].includes(state);
  dot.className = "status-dot " + (busy ? "busy" : "ok");
  label.textContent = {
    idle:             "Ready",
    generating_midi:  "Generating MIDI…",
    preprocessing:    "Preprocessing…",
    building_model:   "Building model…",
    training:         "Training…",
    generating:       "Generating music…",
    done:             "Done",
    error:            "Error",
  }[state] || state;

  // Pipeline steps
  setPipeStep(1, d.midi_data_ready);
  setPipeStep(2, d.cache_ready);
  setPipeStep(3, d.model_ready);
  setPipeStep(4, false);

  // Ready badges
  setVisible("midiReady",  d.midi_data_ready);
  setVisible("cacheReady", d.cache_ready);
  setVisible("modelReady", d.model_ready);

  // Preprocess stats
  if (d.sequences) {
    const el = document.getElementById("preprocessStats");
    el.style.display = "flex";
    el.innerHTML = `
      <div class="stat-pill">${d.sequences.toLocaleString()} sequences</div>
      <div class="stat-pill">${d.vocab_size} unique tokens</div>
      <div class="stat-pill">${(d.total_notes||0).toLocaleString()} total notes</div>`;
  }

  // Training progress
  if (state === "training" && d.latest) {
    showTrainingProgress(d);
  }
  if (state === "done" && d.history && d.history.length > 0) {
    showTrainingProgress(d);
  }

  // Button states
  const busy1 = state === "generating_midi";
  const busy2 = state === "preprocessing";
  const busy3 = state === "training" || state === "building_model";
  const busy4 = state === "generating";

  setBtnBusy("btnGenMidi",   busy1, "Generating…",       "<i class='fa-solid fa-bolt'></i> Generate MIDI Data");
  setBtnBusy("btnPreprocess",busy2, "Preprocessing…",    "<i class='fa-solid fa-gears'></i> Preprocess Data");
  setBtnBusy("btnTrain",     busy3, "Training…",         "<i class='fa-solid fa-brain'></i> Start Training");
  setBtnBusy("btnGenerate",  busy4, "Generating music…", "<i class='fa-solid fa-wand-magic-sparkles'></i> Generate Music");

  // Output files
  if (d.output_files && d.output_files.length > 0) {
    renderOutputFiles(d.output_files);
  }

  // Error toast
  if (state === "error" && d.message) {
    showToast("Error: " + d.message, "error");
  }
}

function setPipeStep(num, done) {
  const el = document.getElementById(`pipeStep${num}`);
  el.className = "pipe-step " + (done ? "done" : "");
}

function showTrainingProgress(d) {
  const el = document.getElementById("trainingProgress");
  el.style.display = "block";

  const ep    = d.epoch || 0;
  const total = d.total_epochs || 1;
  const pct   = d.progress_pct || 0;
  const lat   = d.latest || {};

  document.getElementById("epochLabel").textContent = `Epoch ${ep} / ${total}`;
  document.getElementById("pctLabel").textContent   = pct + "%";
  document.getElementById("progressFill").style.width = pct + "%";

  document.getElementById("mLoss").textContent   = lat.loss    ?? "—";
  document.getElementById("mAcc").textContent    = (lat.accuracy ?? "—") + (lat.accuracy != null ? "%" : "");
  document.getElementById("mValLoss").textContent = lat.val_loss ?? "—";
  document.getElementById("mValAcc").textContent  = (lat.val_acc ?? "—") + (lat.val_acc != null ? "%" : "");

  // Update chart data
  if (d.history && d.history.length > 0) {
    lossHistory = d.history.map(h => h.loss);
    accHistory  = d.history.map(h => h.accuracy);
    drawLossChart();
  }
}

// ── Mini Canvas Loss Chart ────────────────────────
function drawLossChart() {
  const canvas = document.getElementById("lossChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = 80;

  ctx.clearRect(0, 0, W, H);

  if (lossHistory.length < 2) return;

  const maxL = Math.max(...lossHistory, 0.01);
  const minL = Math.min(...lossHistory);

  // Background
  ctx.fillStyle = "rgba(255,255,255,0.03)";
  ctx.beginPath();
  ctx.roundRect(0,0,W,H,8);
  ctx.fill();

  // Draw loss line
  function drawLine(data, color, maxV, minV) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * W;
      const y = H - ((v - minV) / (maxV - minV + 1e-6)) * (H - 8) - 4;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawLine(lossHistory, "#6C63FF", maxL, minL);
  // Label
  ctx.fillStyle = "rgba(240,240,255,0.4)";
  ctx.font = "10px Inter";
  ctx.fillText("Loss", 6, 14);
}

// ── Render Output Files ───────────────────────────
function renderOutputFiles(files) {
  const wrap = document.getElementById("outputFiles");
  const list = document.getElementById("filesList");
  wrap.style.display = "block";
  list.innerHTML = "";
  files.slice(0, 8).forEach(f => {
    const div = document.createElement("div");
    div.className = "file-item";
    div.innerHTML = `
      <span class="file-name"><i class="fa-solid fa-file-audio"></i>${f}</span>
      <div class="file-actions">
        <a href="/play/${f}" target="_blank">
          <button class="file-btn play"><i class="fa-solid fa-play"></i> Play</button>
        </a>
        <a href="/download/${f}" download>
          <button class="file-btn dl"><i class="fa-solid fa-download"></i> Download</button>
        </a>
      </div>`;
    list.appendChild(div);
  });
}

// ── Button helpers ────────────────────────────────
function setBtnBusy(id, busy, busyLabel, idleHTML) {
  const btn = document.getElementById(id);
  if (!btn) return;
  btn.disabled = busy;
  btn.innerHTML = busy
    ? `<i class="fa-solid fa-spinner fa-spin"></i> ${busyLabel}`
    : idleHTML;
}

function setVisible(id, show) {
  const el = document.getElementById(id);
  if (el) el.style.display = show ? "flex" : "none";
}

// ── API Actions ───────────────────────────────────
async function generateMidi() {
  showToast("Generating MIDI training files…");
  await fetch("/generate-midi", { method: "POST" });
  startPolling();
}

async function preprocess() {
  showToast("Preprocessing MIDI sequences…");
  await fetch("/preprocess", { method: "POST" });
  startPolling();
}

async function startTraining() {
  const epochs    = parseInt(document.getElementById("epochs").value) || 50;
  const batchSize = parseInt(document.getElementById("batchSize").value) || 64;
  const lr        = parseFloat(document.getElementById("lr").value) || 0.001;

  showToast(`Starting LSTM training — ${epochs} epochs…`);
  lossHistory = []; accHistory = [];
  document.getElementById("trainingProgress").style.display = "block";

  const r = await fetch("/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ epochs, batch_size: batchSize, lr }),
  });
  const d = await r.json();
  if (!d.ok) { showToast(d.message, "error"); return; }
  startPolling();
}

async function generateMusic() {
  const n_notes     = parseInt(document.getElementById("nNotes").value) || 100;
  const temperature = parseFloat(document.getElementById("temperature").value) || 1.0;
  const bpm         = parseInt(document.getElementById("bpm").value) || 90;

  showToast("Composing new music…");
  const r = await fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n_notes, temperature, bpm }),
  });
  const d = await r.json();
  if (!d.ok) { showToast(d.message, "error"); return; }
  startPolling();
}

// ── Toast ─────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = "ok") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast show" + (type === "error" ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3000);
}

// ── Init ──────────────────────────────────────────
pollStatus();
startPolling();
console.log("%c MusicAI — CodeAlpha Task 3", "font-size:16px;font-weight:bold;color:#6C63FF;");
