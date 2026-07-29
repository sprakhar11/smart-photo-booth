/* ============================================================================
   Frame Change
   ----------------------------------------------------------------------------
   Both hands become a four-corner frame:

       index tip ──────── top edge ──────── index tip
           │                                    │
       thumb tip ─────── bottom edge ────── thumb tip

   The topology is fixed, never re-sorted. That is deliberate: twist one hand
   and the top and bottom edges cross, so the shape becomes a bowtie and the
   graded region splits into two triangles. Nonzero winding fills both lobes.

   Gestures
     - Close your eyes briefly  -> next colour / view / light treatment.
       A deliberate hold is required because involuntary blinks (~100-150ms)
       would otherwise scramble the look every few seconds.
     - Gather all four tips     -> start recording, again to stop, then a
       download button appears.

   Smoothness comes from a One Euro filter per corner, stepped every animation
   frame rather than only when a detection lands, so 30fps inference still
   renders at display rate.
   ========================================================================== */

import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/vision_bundle.mjs";

/* ------------------------------ configuration ---------------------------- */

const MP_VERSION = "1.0.0";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// MediaPipe hand topology indices.
const L_WRIST = 0;
const L_THUMB_TIP = 4;
const L_INDEX_TIP = 8;
const L_MIDDLE_MCP = 9;

const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const TUNING = {
  graceMs: 170,      // keep the frame alive through short detection dropouts
  refilterMs: 400,   // gone longer than this -> snap instead of gliding in
  fadeTau: 0.085,    // frame opacity easing time constant (seconds)
  transitionMs: 430, // effect crossfade duration

  // Eye-hold to change effect. Blendshape score 0..1, hysteresis + a hold long
  // enough to reject the involuntary blinks everyone makes every few seconds.
  eyeCloseOn: 0.5,
  eyeCloseOff: 0.3,
  blinkHoldMs: 280,
  blinkCooldownMs: 700,

  // All four tips gathered together toggles recording. Measured as the widest
  // gap among the four corners, divided by hand size so range does not matter.
  clusterOn: 0.85,
  clusterOff: 1.25,
  clusterHoldMs: 350,
  recordCooldownMs: 1000,

  faceEveryNth: 2,   // face inference cadence; blinks are slow, hands are not
  recordFps: 30,
  recordBitrate: 8_000_000,
  cornerRadiusPct: 0.03,
  minBoxArea: 16,    // below this the frame has no interior worth grading
};

/* ------------------------------ One Euro filter -------------------------- */

const TWO_PI = Math.PI * 2;

class LowPass {
  constructor() { this.value = null; }
  reset() { this.value = null; }
  filter(x, alpha) {
    this.value = this.value === null ? x : alpha * x + (1 - alpha) * this.value;
    return this.value;
  }
}

class OneEuro {
  constructor(minCutoff = 2.2, beta = 0.03, dCutoff = 1.2) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xf = new LowPass();
    this.dxf = new LowPass();
    this.prev = null;
  }
  reset() { this.xf.reset(); this.dxf.reset(); this.prev = null; }
  setParams(minCutoff, beta) { this.minCutoff = minCutoff; this.beta = beta; }
  static alpha(cutoff, dt) {
    const tau = 1 / (TWO_PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(value, dt) {
    if (!(dt > 0)) dt = 1 / 60;
    const rawRate = this.prev === null ? 0 : (value - this.prev) / dt;
    this.prev = value;
    const rate = this.dxf.filter(rawRate, OneEuro.alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(rate);
    return this.xf.filter(value, OneEuro.alpha(cutoff, dt));
  }
}

class SmoothPoint {
  constructor() {
    this.fx = new OneEuro();
    this.fy = new OneEuro();
    this.x = 0;
    this.y = 0;
    this.primed = false;
  }
  reset() { this.fx.reset(); this.fy.reset(); this.primed = false; }
  setParams(mc, beta) { this.fx.setParams(mc, beta); this.fy.setParams(mc, beta); }
  update(x, y, dt) {
    if (!this.primed) { this.primed = true; this.fx.reset(); this.fy.reset(); }
    this.x = this.fx.filter(x, dt);
    this.y = this.fy.filter(y, dt);
  }
}

/* ------------------------------ effect catalog --------------------------- */
/* Each effect = a CSS filter applied to a second copy of the video frame plus
   an optional composited overlay for tint / light. Both go into an offscreen
   layer so the crossfade and blend modes stay correct.                     */

const EFFECTS = [
  {
    name: "Neon Bloom",
    accent: "#4ad9ff",
    filter: "saturate(2.4) contrast(1.22) hue-rotate(178deg) brightness(1.06)",
    paint(ctx, b) {
      const g = ctx.createLinearGradient(b.x, b.y, b.x + b.w, b.y + b.h);
      g.addColorStop(0, "rgba(0,170,255,0.34)");
      g.addColorStop(0.5, "rgba(120,60,255,0.20)");
      g.addColorStop(1, "rgba(255,0,190,0.34)");
      ctx.globalCompositeOperation = "screen";
      ctx.fillStyle = g;
      ctx.fillRect(b.x, b.y, b.w, b.h);
    },
  },
  {
    name: "Thermal",
    accent: "#ff8a3d",
    filter: "invert(1) hue-rotate(155deg) saturate(3.4) contrast(1.5) brightness(1.08)",
    paint(ctx, b) {
      const g = ctx.createLinearGradient(b.x, b.y + b.h, b.x, b.y);
      g.addColorStop(0, "rgba(30,0,90,0.35)");
      g.addColorStop(0.55, "rgba(255,90,0,0.16)");
      g.addColorStop(1, "rgba(255,240,120,0.24)");
      ctx.globalCompositeOperation = "overlay";
      ctx.fillStyle = g;
      ctx.fillRect(b.x, b.y, b.w, b.h);
    },
  },
  {
    name: "Night Vision",
    accent: "#4dff9b",
    filter: "grayscale(1) brightness(1.5) contrast(1.35)",
    paint(ctx, b) {
      ctx.globalCompositeOperation = "multiply";
      ctx.fillStyle = "rgba(70,255,140,0.92)";
      ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "rgba(0,0,0,0.16)";
      for (let y = b.y; y < b.y + b.h; y += 4) ctx.fillRect(b.x, y, b.w, 2);
      ctx.globalCompositeOperation = "multiply";
      ctx.fillStyle = vignette(ctx, b, 0.42, 0.95);
      ctx.fillRect(b.x, b.y, b.w, b.h);
    },
  },
  {
    name: "Spotlight",
    accent: "#ffe58a",
    filter: "brightness(1.42) contrast(1.12) saturate(1.24)",
    paint(ctx, b) {
      const g = ctx.createRadialGradient(b.cx, b.cy, 0, b.cx, b.cy, b.r);
      g.addColorStop(0, "rgba(255,240,200,0.42)");
      g.addColorStop(0.45, "rgba(255,225,160,0.14)");
      g.addColorStop(1, "rgba(255,210,120,0)");
      ctx.globalCompositeOperation = "screen";
      ctx.fillStyle = g;
      ctx.fillRect(b.x, b.y, b.w, b.h);
    },
  },
  {
    name: "Noir",
    accent: "#cfd6e6",
    filter: "grayscale(1) contrast(1.62) brightness(0.98)",
    paint(ctx, b) {
      ctx.globalCompositeOperation = "multiply";
      ctx.fillStyle = vignette(ctx, b, 0.34, 0.9);
      ctx.fillRect(b.x, b.y, b.w, b.h);
    },
  },
  {
    name: "X-Ray",
    accent: "#8fd2ff",
    filter: "invert(1) grayscale(1) brightness(1.22) contrast(1.7)",
    paint(ctx, b) {
      ctx.globalCompositeOperation = "screen";
      ctx.fillStyle = "rgba(40,120,220,0.30)";
      ctx.fillRect(b.x, b.y, b.w, b.h);
    },
  },
  {
    name: "Dreamwave",
    accent: "#c99bff",
    filter: "blur(3.5px) saturate(2.1) brightness(1.16) contrast(1.05) hue-rotate(-22deg)",
    paint(ctx, b) {
      const g = ctx.createLinearGradient(b.x, b.y, b.x + b.w, b.y + b.h);
      g.addColorStop(0, "rgba(255,120,220,0.26)");
      g.addColorStop(1, "rgba(110,150,255,0.30)");
      ctx.globalCompositeOperation = "screen";
      ctx.fillStyle = g;
      ctx.fillRect(b.x, b.y, b.w, b.h);
    },
  },
  {
    name: "Vintage",
    accent: "#e0b183",
    filter: "sepia(0.78) contrast(1.22) saturate(1.5) brightness(1.04)",
    paint(ctx, b) {
      ctx.globalCompositeOperation = "multiply";
      ctx.fillStyle = vignette(ctx, b, 0.4, 0.82);
      ctx.fillRect(b.x, b.y, b.w, b.h);
    },
  },
];

function vignette(ctx, b, innerStop, edgeAlpha) {
  const g = ctx.createRadialGradient(b.cx, b.cy, b.r * innerStop, b.cx, b.cy, b.r);
  const edge = Math.round(255 * (1 - edgeAlpha));
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(1, `rgba(${edge},${edge},${edge},1)`);
  return g;
}

/* ------------------------------ hand slots ------------------------------- */
/* Two persistent slots. Detections are matched to slots by nearest wrist, so a
   corner keeps its identity (and its filter history) even when hands cross or
   handedness flickers.                                                     */

class HandSlot {
  constructor(id) {
    this.id = id;
    this.thumb = new SmoothPoint();
    this.index = new SmoothPoint();
    this.wrist = { x: 0, y: 0 };
    this.span = 1;
    this.lastSeen = -1e9;
    this.landmarks = null;
    this.rawThumb = null;
    this.rawIndex = null;
  }
  resetFilters() { this.thumb.reset(); this.index.reset(); }
  isLive(now) { return now - this.lastSeen < TUNING.graceMs; }
}

/* ------------------------------ DOM ------------------------------------- */

const $ = (id) => document.getElementById(id);

const video = $("video");
const canvas = $("view");
const ctx = canvas.getContext("2d", { alpha: false });

const stage = $("stage");
const overlay = $("overlay");
const statusEl = $("status");
const startBtn = $("startBtn");
const hudTop = $("hudTop");
const hudBottom = $("hudBottom");
const hintEl = $("hint");
const hintText = $("hintText");
const effectNameEl = $("effectName");
const effectSwatch = $("effectSwatch");
const effectCounter = $("effectCounter");
const fpsEl = $("fps");
const trackDot = $("trackDot");
const trackText = $("trackText");
const chipsEl = $("chips");
const prevBtn = $("prevBtn");
const nextBtn = $("nextBtn");
const mirrorBtn = $("mirrorBtn");
const skeletonBtn = $("skeletonBtn");
const fullBtn = $("fullBtn");
const stopBtn = $("stopBtn");
const smoothRange = $("smoothRange");
const smoothOut = $("smoothOut");

const eyePill = $("eyePill");
const eyeText = $("eyeText");
const eyeHold = $("eyeHold");
const recPill = $("recPill");
const recTime = $("recTime");
const recordBtn = $("recordBtn");
const recordBtnLabel = $("recordBtnLabel");
const recPanel = $("recPanel");
const recMeta = $("recMeta");
const recDownload = $("recDownload");
const recDiscard = $("recDiscard");

// Offscreen layer where each effect is composited before being clipped in.
const fx = document.createElement("canvas");
const fxCtx = fx.getContext("2d", { alpha: false });

// ctx.filter drives the colour grades. Detect it so we can warn if missing.
const SUPPORTS_FILTER = (() => {
  const probe = document.createElement("canvas").getContext("2d");
  probe.filter = "blur(1px)";
  return probe.filter === "blur(1px)";
})();

/* ------------------------------ state ----------------------------------- */

const state = {
  hands: null,
  face: null,
  stream: null,
  running: false,
  rafId: 0,

  slots: [new HandSlot(0), new HandSlot(1)],

  W: 0,
  H: 0,

  mirror: true,
  skeleton: false,

  effect: 0,
  prevEffect: -1,
  transitionStart: -1e9,

  quadAlpha: 0,
  quad: null,      // [indexA, indexB, thumbB, thumbA] - fixed topology
  quadKinds: ["index", "index", "thumb", "thumb"],

  eye: {
    present: false,
    score: 0,
    closed: false,
    closedSince: 0,
    fired: false,
    lastFire: -1e9,
    progress: 0,
  },

  cluster: {
    ratio: Infinity,
    active: false,
    since: 0,
    fired: false,
    lastToggle: -1e9,
  },

  rec: {
    recorder: null,
    stream: null,
    chunks: [],
    active: false,
    startedAt: 0,
    duration: 0,
    blob: null,
    url: null,
    filename: "",
  },

  ripples: [],
  hintQueue: [],
  activeHint: null,
  introShown: false,

  lastVideoTime: -1,
  detectTick: 0,
  lastFrameTs: 0,
  fpsAvg: 0,
  lastHoldPct: -1,
};

/* ------------------------------ boot ------------------------------------ */

buildChips();
applySmoothing(Number(smoothRange.value));
syncEffectUI();
syncRecordUI();

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
prevBtn.addEventListener("click", () => cycleEffect(-1));
nextBtn.addEventListener("click", () => cycleEffect(1));
recordBtn.addEventListener("click", () => toggleRecording(performance.now(), "button"));
recDownload.addEventListener("click", saveRecording);
recDiscard.addEventListener("click", discardRecording);

mirrorBtn.addEventListener("click", () => {
  state.mirror = !state.mirror;
  mirrorBtn.classList.toggle("is-on", state.mirror);
  mirrorBtn.setAttribute("aria-pressed", String(state.mirror));
  // Mirroring flips x, so drop history to avoid a sweep across the screen.
  state.slots.forEach((s) => s.resetFilters());
});

skeletonBtn.addEventListener("click", () => {
  state.skeleton = !state.skeleton;
  skeletonBtn.classList.toggle("is-on", state.skeleton);
  skeletonBtn.setAttribute("aria-pressed", String(state.skeleton));
});

fullBtn.addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else stage.requestFullscreen?.();
});

smoothRange.addEventListener("input", () => applySmoothing(Number(smoothRange.value)));

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.code === "Space" || e.key === "ArrowRight") { e.preventDefault(); cycleEffect(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); cycleEffect(-1); }
  else if (e.key === "r" || e.key === "R") toggleRecording(performance.now(), "key");
  else if (e.key === "m" || e.key === "M") mirrorBtn.click();
  else if (e.key === "h" || e.key === "H") skeletonBtn.click();
  else if (e.key === "f" || e.key === "F") fullBtn.click();
  else if (/^[1-9]$/.test(e.key)) {
    const i = Number(e.key) - 1;
    if (i < EFFECTS.length) setEffect(i);
  }
});

/* ------------------------------ lifecycle -------------------------------- */

async function start() {
  startBtn.disabled = true;

  if (!navigator.mediaDevices?.getUserMedia) {
    return fail("This browser has no camera API. Try Chrome, Edge, Firefox or Safari.");
  }
  if (!window.isSecureContext) {
    return fail("Camera access needs https:// or localhost. Open the page from a server.");
  }

  try {
    setStatus("Requesting camera…", { busy: true });
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 60, min: 24 },
      },
    });
  } catch (err) {
    const name = err?.name || "";
    if (name === "NotAllowedError" || name === "SecurityError") {
      return fail("Camera permission was blocked. Allow it in the address bar, then retry.");
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return fail("No camera found on this device.");
    }
    if (name === "NotReadableError") {
      return fail("The camera is already in use by another app. Close it and retry.");
    }
    return fail(`Could not start the camera: ${err?.message || name || "unknown error"}`);
  }

  video.srcObject = state.stream;
  try { await video.play(); } catch { /* readiness awaited below */ }
  await waitForVideo();
  resize();

  if (!state.hands || !state.face) {
    try {
      setStatus("Loading hand and face models…", { busy: true });
      const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
      const [hands, face] = await Promise.all([
        state.hands ? Promise.resolve(state.hands) : createHands(fileset),
        state.face ? Promise.resolve(state.face) : createFace(fileset),
      ]);
      state.hands = hands;
      state.face = face;
    } catch (err) {
      return fail(`Could not load the models: ${err?.message || err}. Check your connection.`);
    }
  }

  overlay.classList.add("is-hidden");
  stage.classList.add("is-live");
  hudTop.hidden = false;
  hudBottom.hidden = false;

  if (!SUPPORTS_FILTER) {
    flashHint("This browser ignores canvas filters, so grades show as tints only.", 5000);
  }

  state.running = true;
  state.lastFrameTs = performance.now();
  state.rafId = requestAnimationFrame(loop);
}

/** GPU delegate is much smoother; fall back to CPU where unavailable. */
async function withDelegateFallback(make) {
  try { return await make("GPU"); } catch { return await make("CPU"); }
}

function createHands(fileset) {
  return withDelegateFallback((delegate) =>
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    }));
}

function createFace(fileset) {
  return withDelegateFallback((delegate) =>
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,          // eyeBlinkLeft / eyeBlinkRight
      outputFacialTransformationMatrixes: false,
    }));
}

function waitForVideo() {
  if (video.readyState >= 2 && video.videoWidth) return Promise.resolve();
  return new Promise((resolve) => {
    video.addEventListener("loadeddata", resolve, { once: true });
  });
}

function stop() {
  if (state.rec.active) stopRecording(performance.now());

  state.running = false;
  cancelAnimationFrame(state.rafId);
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;

  state.slots.forEach((s) => { s.resetFilters(); s.lastSeen = -1e9; s.landmarks = null; });
  state.quad = null;
  state.quadAlpha = 0;
  state.ripples.length = 0;
  state.lastVideoTime = -1;
  resetEye();

  hudTop.hidden = true;
  hudBottom.hidden = true;
  hintEl.hidden = true;
  stage.classList.remove("is-live");
  overlay.classList.remove("is-hidden");
  startBtn.disabled = false;
  setStatus("Camera stopped. Enable it again whenever you like.");
  ctx.clearRect(0, 0, state.W, state.H);
}

function fail(message) {
  setStatus(message, { error: true });
  startBtn.disabled = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
}

function setStatus(text, { error = false, busy = false } = {}) {
  statusEl.textContent = text;
  statusEl.classList.toggle("is-error", error);
  statusEl.classList.toggle("is-busy", busy);
}

function resize() {
  const w = video.videoWidth || 1280;
  const h = video.videoHeight || 720;
  if (w === state.W && h === state.H) return;
  state.W = canvas.width = fx.width = w;
  state.H = canvas.height = fx.height = h;
  canvas.style.aspectRatio = `${w} / ${h}`;
  ctx.imageSmoothingQuality = "high";
  fxCtx.imageSmoothingQuality = "high";
}

/* ------------------------------ main loop -------------------------------- */

function loop(now) {
  if (!state.running) return;
  state.rafId = requestAnimationFrame(loop);

  const dt = Math.min(0.1, Math.max(0.001, (now - state.lastFrameTs) / 1000));
  state.lastFrameTs = now;

  const inst = 1 / dt;
  state.fpsAvg = state.fpsAvg ? state.fpsAvg * 0.9 + inst * 0.1 : inst;
  fpsEl.textContent = `${Math.round(state.fpsAvg)} fps`;

  resize();

  // Inference only on genuinely new camera frames.
  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    state.detectTick++;

    try {
      const hands = state.hands.detectForVideo(video, now);
      if (hands) ingestHands(hands, now);
    } catch { /* skip this frame */ }

    // Hands need every frame for precision; eyelids move far slower.
    if (state.detectTick % TUNING.faceEveryNth === 0) {
      try {
        const face = state.face.detectForVideo(video, now);
        ingestFace(face, now);
      } catch { /* skip this frame */ }
    }
  }

  advance(dt, now);
  render(now);
}

/* ------------------------------ detection intake ------------------------- */

function ingestHands(result, now) {
  const lists = result.landmarks || [];
  const dets = [];

  for (let i = 0; i < lists.length && dets.length < 2; i++) {
    const lm = lists[i];
    if (!lm || lm.length < 21) continue;
    const wrist = toCanvas(lm[L_WRIST]);
    const mcp = toCanvas(lm[L_MIDDLE_MCP]);
    dets.push({
      thumb: toCanvas(lm[L_THUMB_TIP]),
      index: toCanvas(lm[L_INDEX_TIP]),
      wrist,
      span: Math.max(24, Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)),
      landmarks: lm,
    });
  }

  const [a, b] = state.slots;
  let pairs;

  if (dets.length === 0) {
    pairs = [];
  } else if (dets.length === 1) {
    pairs = [[dets[0], matchCost(dets[0], a, now) <= matchCost(dets[0], b, now) ? a : b]];
  } else {
    const straight = matchCost(dets[0], a, now) + matchCost(dets[1], b, now);
    const swapped = matchCost(dets[0], b, now) + matchCost(dets[1], a, now);
    pairs = straight <= swapped
      ? [[dets[0], a], [dets[1], b]]
      : [[dets[0], b], [dets[1], a]];
  }

  for (const [det, slot] of pairs) {
    if (now - slot.lastSeen > TUNING.refilterMs) slot.resetFilters();
    slot.wrist = det.wrist;
    slot.span = det.span;
    slot.landmarks = det.landmarks;
    slot.lastSeen = now;
    slot.rawThumb = det.thumb;
    slot.rawIndex = det.index;
  }
}

function matchCost(det, slot, now) {
  if (now - slot.lastSeen > 500) return 1e6;
  return Math.hypot(det.wrist.x - slot.wrist.x, det.wrist.y - slot.wrist.y);
}

function toCanvas(lm) {
  return {
    x: (state.mirror ? 1 - lm.x : lm.x) * state.W,
    y: lm.y * state.H,
  };
}

/* ------------------------------ blink trigger ---------------------------- */

/** Mean of the two eye-blink blendshapes: 0 wide open, 1 fully shut. */
function ingestFace(result, now) {
  const shapes = result?.faceBlendshapes?.[0]?.categories;
  const eye = state.eye;

  if (!shapes || !shapes.length) {
    resetEye();
    return;
  }

  let left = 0, right = 0;
  for (const c of shapes) {
    if (c.categoryName === "eyeBlinkLeft") left = c.score;
    else if (c.categoryName === "eyeBlinkRight") right = c.score;
  }

  eye.present = true;
  eye.score = (left + right) / 2;

  if (!eye.closed && eye.score > TUNING.eyeCloseOn) {
    eye.closed = true;
    eye.closedSince = now;
    eye.fired = false;
  } else if (eye.closed && eye.score < TUNING.eyeCloseOff) {
    eye.closed = false;
    eye.fired = false;
  }

  // Fire once the lids have stayed shut past a natural blink's duration.
  if (eye.closed && !eye.fired && now - eye.closedSince >= TUNING.blinkHoldMs) {
    eye.fired = true;
    if (now - eye.lastFire > TUNING.blinkCooldownMs) {
      eye.lastFire = now;
      cycleEffect(1);
      flashHint(`${EFFECTS[state.effect].name}`, 1100);
    }
  }
}

function resetEye() {
  const eye = state.eye;
  eye.present = false;
  eye.score = 0;
  eye.closed = false;
  eye.fired = false;
  eye.progress = 0;
}

/* ------------------------------ per-frame update ------------------------- */

function advance(dt, now) {
  for (const slot of state.slots) {
    if (slot.rawThumb && slot.isLive(now)) {
      // Corners track the real tips with no floor, so an edge can close fully.
      slot.thumb.update(slot.rawThumb.x, slot.rawThumb.y, dt);
      slot.index.update(slot.rawIndex.x, slot.rawIndex.y, dt);
    }
  }

  const [a, b] = state.slots;
  const bothLive = a.isLive(now) && b.isLive(now) && a.thumb.primed && b.thumb.primed;

  // Exponential ease -> framerate independent fade.
  const target = bothLive ? 1 : 0;
  state.quadAlpha += (target - state.quadAlpha) * (1 - Math.exp(-dt / TUNING.fadeTau));
  if (Math.abs(state.quadAlpha - target) < 0.002) state.quadAlpha = target;

  if (bothLive) {
    state.quad = buildQuad(a, b);
    updateCluster(a, b, now);
  } else {
    if (state.quadAlpha <= 0.002) state.quad = null;
    state.cluster.active = false;
    state.cluster.fired = false;
    state.cluster.ratio = Infinity;
  }

  // Eye-hold progress for the HUD meter.
  const eye = state.eye;
  eye.progress = eye.closed
    ? Math.min(1, (now - eye.closedSince) / TUNING.blinkHoldMs)
    : 0;

  if (state.rec.active) state.rec.duration = (now - state.rec.startedAt) / 1000;

  state.ripples = state.ripples.filter((r) => now - r.t0 < 620);

  updateBadges(now, bothLive);
}

/** Fixed topology: top edge joins the index tips, bottom edge the thumbs.
    Never angle-sorted, so twisting a hand crosses the edges into a bowtie. */
function buildQuad(a, b) {
  const [left, right] = a.wrist.x <= b.wrist.x ? [a, b] : [b, a];
  return [
    { x: left.index.x, y: left.index.y },
    { x: right.index.x, y: right.index.y },
    { x: right.thumb.x, y: right.thumb.y },
    { x: left.thumb.x, y: left.thumb.y },
  ];
}

/* ------------------------------ record gesture --------------------------- */

/** Widest gap among the four corners, divided by hand size. Small = gathered. */
function clusterRatio(a, b) {
  const pts = [a.thumb, a.index, b.thumb, b.index];
  let widest = 0;
  for (let i = 0; i < 4; i++) {
    for (let j = i + 1; j < 4; j++) {
      const d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
      if (d > widest) widest = d;
    }
  }
  return widest / Math.max(1, (a.span + b.span) / 2);
}

function updateCluster(a, b, now) {
  const c = state.cluster;
  c.ratio = clusterRatio(a, b);

  if (!c.active && c.ratio < TUNING.clusterOn) {
    c.active = true;
    c.since = now;
    c.fired = false;
  } else if (c.active && c.ratio > TUNING.clusterOff) {
    c.active = false;
    c.fired = false;
  }

  if (c.active && !c.fired && now - c.since >= TUNING.clusterHoldMs) {
    c.fired = true;
    if (now - c.lastToggle > TUNING.recordCooldownMs) {
      c.lastToggle = now;
      const pts = [a.thumb, a.index, b.thumb, b.index];
      state.ripples.push({
        x: pts.reduce((s, p) => s + p.x, 0) / 4,
        y: pts.reduce((s, p) => s + p.y, 0) / 4,
        t0: now,
      });
      toggleRecording(now, "gesture");
    }
  }
}

/* ------------------------------ recording -------------------------------- */

/* MP4/H.264 is preferred over WebM despite larger files, because the recording
   is something people take away and open elsewhere: MP4 plays in QuickTime,
   Photos, iMovie and on phones, while WebM does not open on macOS by default.
   MediaRecorder's WebM also reports duration as Infinity, which breaks seeking
   in some players. WebM remains the fallback for browsers without MP4 support. */
function pickMimeType() {
  const candidates = [
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4;codecs=avc1",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  for (const c of candidates) {
    if (window.MediaRecorder?.isTypeSupported?.(c)) return c;
  }
  return "";
}

function toggleRecording(now, source) {
  if (state.rec.active) stopRecording(now);
  else startRecording(now, source);
}

function startRecording(now, source) {
  if (state.rec.active) return;
  if (!window.MediaRecorder || !canvas.captureStream) {
    flashHint("This browser cannot record video.", 3200);
    return;
  }
  if (!state.running) {
    flashHint("Start the camera before recording.", 2600);
    return;
  }

  let stream;
  try {
    stream = canvas.captureStream(TUNING.recordFps);
  } catch {
    flashHint("Could not capture the canvas for recording.", 3200);
    return;
  }

  const mime = pickMimeType();
  let recorder;
  try {
    recorder = new MediaRecorder(
      stream,
      mime ? { mimeType: mime, videoBitsPerSecond: TUNING.recordBitrate } : undefined
    );
  } catch {
    try {
      recorder = new MediaRecorder(stream);
    } catch {
      stream.getTracks().forEach((t) => t.stop());
      flashHint("This browser cannot record video.", 3200);
      return;
    }
  }

  const rec = state.rec;
  rec.chunks = [];
  rec.recorder = recorder;
  rec.stream = stream;
  rec.active = true;
  rec.startedAt = now;
  rec.duration = 0;

  recorder.ondataavailable = (e) => { if (e.data && e.data.size) rec.chunks.push(e.data); };
  recorder.onstop = finalizeRecording;
  recorder.onerror = () => { flashHint("Recording stopped: recorder error.", 3200); };

  // Timeslice keeps chunks flowing so a crash still leaves usable data.
  recorder.start(1000);

  clearRecordingResult();
  syncRecordUI();
  flashHint(source === "gesture" ? "Recording started — gather four tips again to stop" : "Recording started", 2200);
}

function stopRecording(now) {
  const rec = state.rec;
  if (!rec.active) return;
  rec.duration = (now - rec.startedAt) / 1000;
  rec.active = false;
  try {
    if (rec.recorder && rec.recorder.state !== "inactive") rec.recorder.stop();
    else finalizeRecording();
  } catch {
    finalizeRecording();
  }
  syncRecordUI();
}

function finalizeRecording() {
  const rec = state.rec;
  rec.stream?.getTracks().forEach((t) => t.stop());
  rec.stream = null;

  const type = (rec.recorder?.mimeType || pickMimeType() || "video/webm").split(";")[0];
  const blob = new Blob(rec.chunks, { type });
  rec.chunks = [];
  rec.recorder = null;

  if (!blob.size) {
    flashHint("Nothing was recorded.", 2600);
    syncRecordUI();
    return;
  }

  if (rec.url) URL.revokeObjectURL(rec.url);
  rec.blob = blob;
  rec.url = URL.createObjectURL(blob);
  rec.filename = `frame-change-${timestamp()}.${type.includes("mp4") ? "mp4" : "webm"}`;

  recDownload.href = rec.url;
  recDownload.setAttribute("download", rec.filename);
  recMeta.textContent = `${formatDuration(rec.duration)} · ${formatBytes(blob.size)} · ${type.includes("mp4") ? "MP4" : "WebM"}`;
  recPanel.hidden = false;

  syncRecordUI();
  flashHint("Recording ready to download", 2400);
}

function discardRecording() {
  clearRecordingResult();
}

function clearRecordingResult() {
  const rec = state.rec;
  if (rec.url) {
    URL.revokeObjectURL(rec.url);
    rec.url = null;
  }
  rec.blob = null;
  rec.filename = "";
  recDownload.removeAttribute("href");
  recDownload.removeAttribute("download");
  recPanel.hidden = true;
}

/* Save through a throwaway anchor built at click time. Relying on the panel's
   own href/download attributes let a stale pair survive a restart, which once
   saved a file named after the blob UUID with no extension. */
function saveRecording(e) {
  const rec = state.rec;
  if (!rec.blob) return;
  e.preventDefault();

  const url = URL.createObjectURL(rec.blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = rec.filename || `frame-change-${timestamp()}.webm`;
  a.rel = "noopener";
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

function syncRecordUI() {
  const on = state.rec.active;
  recPill.hidden = !on;
  recordBtn.classList.toggle("is-on", on);
  recordBtn.setAttribute("aria-pressed", String(on));
  recordBtnLabel.textContent = on ? "Stop" : "Record";
}

function timestamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function formatDuration(sec) {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/* ------------------------------ HUD ------------------------------------- */

function updateBadges(now, bothLive) {
  const live = state.slots.filter((s) => s.isLive(now)).length;
  trackDot.classList.toggle("is-locked", bothLive);
  trackDot.classList.toggle("is-lost", live === 0);

  let steady = null;
  if (bothLive) {
    trackText.textContent = "Frame locked";
    if (!state.introShown) {
      state.introShown = true;
      flashHint("Close your eyes briefly to change the look", 3600);
      flashHint("Gather all four tips to start recording", 3600);
    }
  } else if (live === 1) {
    trackText.textContent = "1 hand";
    steady = "Show your other hand";
  } else {
    trackText.textContent = "Searching…";
    steady = "Show both hands, thumb and index spread like an L";
  }

  // Eye meter
  const eye = state.eye;
  eyePill.classList.toggle("is-open", eye.present && !eye.closed);
  eyePill.classList.toggle("is-closed", eye.present && eye.closed);
  const pct = Math.round(eye.progress * 100);
  if (pct !== state.lastHoldPct) {
    state.lastHoldPct = pct;
    eyeHold.style.setProperty("--hold", `${pct}%`);
  }
  const eyeLabel = !eye.present ? "No face" : eye.closed ? "Hold…" : "Eyes open";
  if (eyeText.textContent !== eyeLabel) eyeText.textContent = eyeLabel;

  if (state.rec.active) {
    const t = formatDuration(state.rec.duration);
    if (recTime.textContent !== t) recTime.textContent = t;
  }

  // A queued flash wins while it lasts, then steady guidance returns.
  if (!state.activeHint && state.hintQueue.length) {
    const next = state.hintQueue.shift();
    state.activeHint = { text: next.text, until: now + next.ms };
  }
  if (state.activeHint && now < state.activeHint.until) setHint(state.activeHint.text);
  else { state.activeHint = null; setHint(steady); }
}

function flashHint(text, ms) {
  // Cap the queue so rapid triggers cannot make hints lag behind reality.
  if (state.hintQueue.length >= 2) state.hintQueue.shift();
  state.hintQueue.push({ text, ms });
}

function setHint(text) {
  if (!text) { hintEl.hidden = true; return; }
  if (hintText.textContent !== text) hintText.textContent = text;
  hintEl.hidden = false;
}

/* ------------------------------ rendering -------------------------------- */

function render(now) {
  const { W, H } = state;
  if (!W || !H) return;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.filter = "none";

  // 1. Untouched camera frame.
  drawVideo(ctx);

  const quad = state.quad;
  const alpha = state.quadAlpha;

  if (quad && alpha > 0.004) {
    const box = bounds(quad);
    const radius = Math.min(box.w, box.h) * TUNING.cornerRadiusPct;

    // 2. Graded frame, clipped to the shape. Nonzero winding means a crossed
    //    (twisted) quad fills as two triangles rather than one region.
    if (box.w * box.h > TUNING.minBoxArea) {
      ctx.save();
      polyPath(ctx, quad, radius);
      ctx.clip();

      const t = transitionProgress(now);
      if (t < 1 && state.prevEffect >= 0) {
        ctx.globalAlpha = alpha;
        ctx.drawImage(renderLayer(EFFECTS[state.prevEffect], box), 0, 0);
      }
      ctx.globalAlpha = alpha * (t < 1 ? easeInOut(t) : 1);
      ctx.drawImage(renderLayer(EFFECTS[state.effect], box), 0, 0);
      ctx.restore();
      ctx.globalAlpha = 1;
    }

    // 3. Edges and corners.
    drawFrame(ctx, quad, radius, alpha);
  }

  if (state.skeleton) drawSkeletons(ctx, now);
  drawRipples(ctx, now);

  ctx.globalAlpha = 1;
}

function drawVideo(c) {
  c.save();
  if (state.mirror) { c.translate(state.W, 0); c.scale(-1, 1); }
  c.drawImage(video, 0, 0, state.W, state.H);
  c.restore();
}

function renderLayer(effect, box) {
  fxCtx.setTransform(1, 0, 0, 1, 0, 0);
  fxCtx.globalAlpha = 1;
  fxCtx.globalCompositeOperation = "source-over";
  fxCtx.filter = SUPPORTS_FILTER ? effect.filter : "none";
  drawVideo(fxCtx);
  fxCtx.filter = "none";
  if (effect.paint) {
    fxCtx.save();
    effect.paint(fxCtx, box);
    fxCtx.restore();
    fxCtx.globalCompositeOperation = "source-over";
  }
  return fx;
}

function drawFrame(c, quad, radius, alpha) {
  const scale = Math.min(state.W, state.H) / 720;
  const accent = EFFECTS[state.effect].accent;
  const pulse = state.eye.progress;

  c.save();
  c.lineJoin = "round";
  c.lineCap = "round";

  // Nothing here is recording-tinted on purpose: the canvas is what gets
  // captured, so a red border would be baked into the exported file. The
  // recording indicator lives in the HUD, which is not captured.
  c.globalAlpha = alpha * (0.4 + pulse * 0.45);
  c.strokeStyle = accent;
  c.shadowColor = accent;
  c.shadowBlur = (18 + pulse * 26) * scale;
  c.lineWidth = (5.5 + pulse * 3) * scale;
  polyPath(c, quad, radius);
  c.stroke();

  // Crisp core line.
  c.shadowBlur = 0;
  c.globalAlpha = alpha;
  c.strokeStyle = "rgba(255,255,255,0.94)";
  c.lineWidth = 2 * scale;
  polyPath(c, quad, radius);
  c.stroke();

  // Corners: index tips take the accent, thumbs read white so the fixed
  // top/bottom topology is visible at a glance.
  quad.forEach((p, i) => {
    const isIndex = state.quadKinds[i] === "index";
    c.globalAlpha = alpha;
    c.beginPath();
    c.arc(p.x, p.y, (isIndex ? 7.5 : 6) * scale, 0, TWO_PI);
    c.fillStyle = isIndex ? accent : "rgba(255,255,255,0.92)";
    c.shadowColor = isIndex ? accent : "rgba(255,255,255,0.8)";
    c.shadowBlur = 13 * scale;
    c.fill();

    c.shadowBlur = 0;
    c.beginPath();
    c.arc(p.x, p.y, 2.8 * scale, 0, TWO_PI);
    c.fillStyle = isIndex ? "#fff" : "#0b0d14";
    c.fill();
  });

  c.restore();
}

function drawSkeletons(c, now) {
  const scale = Math.min(state.W, state.H) / 720;
  c.save();
  c.lineCap = "round";
  for (const slot of state.slots) {
    if (!slot.landmarks || !slot.isLive(now)) continue;
    const pts = slot.landmarks.map(toCanvas);
    c.globalAlpha = 0.5;
    c.strokeStyle = "rgba(255,255,255,0.85)";
    c.lineWidth = 2.2 * scale;
    c.beginPath();
    for (const [i, j] of HAND_BONES) {
      c.moveTo(pts[i].x, pts[i].y);
      c.lineTo(pts[j].x, pts[j].y);
    }
    c.stroke();

    c.globalAlpha = 0.75;
    c.fillStyle = EFFECTS[state.effect].accent;
    for (const p of pts) {
      c.beginPath();
      c.arc(p.x, p.y, 2.8 * scale, 0, TWO_PI);
      c.fill();
    }
  }
  c.restore();
}

function drawRipples(c, now) {
  if (!state.ripples.length) return;
  const scale = Math.min(state.W, state.H) / 720;
  const accent = EFFECTS[state.effect].accent;
  c.save();
  for (const r of state.ripples) {
    const t = (now - r.t0) / 620;
    if (t >= 1) continue;
    const e = 1 - Math.pow(1 - t, 3);
    c.globalAlpha = (1 - t) * 0.85;
    c.strokeStyle = accent;
    c.lineWidth = (3.5 * (1 - t) + 0.6) * scale;
    c.shadowColor = accent;
    c.shadowBlur = 16 * scale;
    c.beginPath();
    c.arc(r.x, r.y, (12 + e * 66) * scale, 0, TWO_PI);
    c.stroke();
  }
  c.restore();
}

/* ------------------------------ geometry helpers ------------------------- */

/** Shoelace area. Signed contributions cancel on a bowtie, which is why it is
    only used for reporting, never to decide whether to draw. */
function polyArea(pts) {
  let a = 0;
  for (let i = 0, n = pts.length; i < n; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % n];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a) / 2;
}

function bounds(pts) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const w = Math.max(1, maxX - minX);
  const h = Math.max(1, maxY - minY);
  return { x: minX, y: minY, w, h, cx: minX + w / 2, cy: minY + h / 2, r: Math.max(w, h) * 0.62 };
}

/** Closed polygon with softly rounded corners (quadratic fillets). */
function polyPath(c, pts, radius) {
  const n = pts.length;
  c.beginPath();
  if (!radius || radius < 0.5) {
    c.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < n; i++) c.lineTo(pts[i].x, pts[i].y);
    c.closePath();
    return;
  }
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const cur = pts[i];
    const next = pts[(i + 1) % n];

    const d1 = Math.hypot(prev.x - cur.x, prev.y - cur.y) || 1;
    const d2 = Math.hypot(next.x - cur.x, next.y - cur.y) || 1;
    const r = Math.min(radius, d1 / 2, d2 / 2);

    const ax = cur.x + ((prev.x - cur.x) / d1) * r;
    const ay = cur.y + ((prev.y - cur.y) / d1) * r;
    const bx = cur.x + ((next.x - cur.x) / d2) * r;
    const by = cur.y + ((next.y - cur.y) / d2) * r;

    if (i === 0) c.moveTo(ax, ay);
    else c.lineTo(ax, ay);
    c.quadraticCurveTo(cur.x, cur.y, bx, by);
  }
  c.closePath();
}

function transitionProgress(now) {
  const t = (now - state.transitionStart) / TUNING.transitionMs;
  return t >= 1 ? 1 : Math.max(0, t);
}

function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

/* ------------------------------ effect switching ------------------------- */

function cycleEffect(dir) {
  setEffect((state.effect + dir + EFFECTS.length) % EFFECTS.length);
}

function setEffect(index) {
  if (index === state.effect) return;
  state.prevEffect = state.effect;
  state.effect = index;
  state.transitionStart = performance.now();
  syncEffectUI();
}

function buildChips() {
  chipsEl.innerHTML = "";
  EFFECTS.forEach((effect, i) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.style.setProperty("--chip-accent", effect.accent);
    chip.innerHTML = `<i></i><span></span>`;
    chip.querySelector("span").textContent = effect.name;
    chip.title = `${effect.name} (${i + 1})`;
    chip.setAttribute("role", "tab");
    chip.addEventListener("click", () => setEffect(i));
    chipsEl.appendChild(chip);
  });
}

function syncEffectUI() {
  const effect = EFFECTS[state.effect];
  effectNameEl.textContent = effect.name;
  effectCounter.textContent = `${state.effect + 1}/${EFFECTS.length}`;
  effectSwatch.style.background = effect.accent;
  effectSwatch.style.color = effect.accent;

  [...chipsEl.children].forEach((chip, i) => {
    const active = i === state.effect;
    chip.classList.toggle("is-active", active);
    chip.setAttribute("aria-selected", String(active));
  });

  chipsEl.children[state.effect]?.scrollIntoView({
    block: "nearest", inline: "nearest", behavior: "smooth",
  });
}

/* ------------------------------ smoothing control ------------------------ */

function applySmoothing(v) {
  const s = Math.min(1, Math.max(0, v / 100));
  // Low cutoff = very smooth; beta rises with it so fast motion still tracks.
  const minCutoff = 5.0 + (0.6 - 5.0) * s;
  const beta = 0.008 + (0.045 - 0.008) * s;
  for (const slot of state.slots) {
    slot.thumb.setParams(minCutoff, beta);
    slot.index.setParams(minCutoff, beta);
  }
  smoothOut.textContent = String(Math.round(v));
  smoothRange.style.setProperty("--pct", `${v}%`);
}

/* ------------------------------ housekeeping ----------------------------- */

/* Debug/tuning handle, e.g.
     frameChange.state.eye.score        // live blink blendshape
     frameChange.state.cluster.ratio    // four-tip gather closeness
     frameChange.setEffect(3)                                              */
window.frameChange = {
  state, EFFECTS, TUNING,
  setEffect, render, advance, ingestFace, buildQuad, clusterRatio, polyArea, bounds,
  startRecording, stopRecording,
};

document.addEventListener("visibilitychange", () => {
  // Timestamps jump while hidden; clear history so nothing lurches on return.
  if (!document.hidden) {
    state.lastFrameTs = performance.now();
    state.slots.forEach((s) => s.resetFilters());
  }
});

window.addEventListener("pagehide", () => {
  if (state.rec.active) stopRecording(performance.now());
  state.stream?.getTracks().forEach((t) => t.stop());
});
