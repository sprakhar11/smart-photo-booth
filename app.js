/* ============================================================================
   Frame Change
   ----------------------------------------------------------------------------
   Tracks both hands with MediaPipe HandLandmarker, takes the thumb tip and
   index tip of each hand as four corners, builds a smoothed quadrilateral,
   and re-renders the webcam feed *inside* that quad with a colour / light
   treatment. Touching thumb to index on either hand cycles the treatment
   with a crossfade.

   Smoothness comes from two things:
     1. A One Euro filter per tracked point (steady when still, snappy when
        moving - no constant-lag that a plain lerp gives you).
     2. The filter is stepped every animation frame, not only when a new
        detection lands, so 30fps inference still renders at display rate.
   ========================================================================== */

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/vision_bundle.mjs";

/* ------------------------------ configuration ---------------------------- */

const MP_VERSION = "1.0.0";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Landmark indices we care about (MediaPipe hand topology).
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
  graceMs: 170,        // keep the quad alive through short detection dropouts
  refilterMs: 400,     // gone longer than this -> snap filters instead of gliding
  fadeTau: 0.085,      // quad opacity easing time constant (seconds)
  transitionMs: 430,   // effect crossfade duration

  // Tips considered touching below pinchOn, released above pinchOff. Measured
  // relative to hand size, so it behaves the same near or far from the lens.
  pinchOn: 0.26,
  pinchOff: 0.44,
  pinchCooldownMs: 420,

  // Closing the tips is the trigger, but it would also collapse the corner it
  // owns and destroy the frame being recoloured. So slightly *before* the
  // trigger fires we freeze that hand's two corners relative to its wrist:
  // the frame keeps its shape and still follows the hand, then eases back to
  // the live tips on release.
  holdOn: 0.62,
  holdOff: 0.74,

  minAreaFrac: 0.0035, // below this the quad has no usable interior
  cornerRadiusPct: 0.03,
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
/* Each effect = a CSS filter applied to a second copy of the video frame,
   plus an optional composited overlay for tints / light. Both are rendered
   into an offscreen layer so the crossfade and blend modes stay correct.   */

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
      // scanlines + edge falloff for the goggle look
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

/** Radial darkening gradient used by several effects. */
function vignette(ctx, b, innerStop, edgeAlpha) {
  const g = ctx.createRadialGradient(b.cx, b.cy, b.r * innerStop, b.cx, b.cy, b.r);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(1, `rgba(${Math.round(255 * (1 - edgeAlpha))},${Math.round(255 * (1 - edgeAlpha))},${Math.round(255 * (1 - edgeAlpha))},1)`);
  return g;
}

/* ------------------------------ hand slots ------------------------------- */
/* Two persistent slots. Detections are matched to slots by nearest wrist so
   a point keeps its identity (and therefore its filter state) even when the
   hands cross over or handedness flickers.                                 */

class HandSlot {
  constructor(id) {
    this.id = id;
    this.thumb = new SmoothPoint();
    this.index = new SmoothPoint();
    this.wrist = { x: 0, y: 0 };
    this.span = 1;
    this.lastSeen = -1e9;
    this.pinching = false;
    this.pinchGlow = 0;
    this.landmarks = null;
    this.rawThumb = null;
    this.rawIndex = null;
    this.holding = false;     // corners frozen relative to the wrist
    this.holdOffsets = null;  // {thumb:{dx,dy}, index:{dx,dy}}
  }
  resetFilters() { this.thumb.reset(); this.index.reset(); this.holdOffsets = null; this.holding = false; }
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

// Offscreen layer where each effect is composited before being clipped in.
const fx = document.createElement("canvas");
const fxCtx = fx.getContext("2d", { alpha: false });

// ctx.filter is what drives the colour grades. Detect it so we can warn.
const SUPPORTS_FILTER = (() => {
  const probe = document.createElement("canvas").getContext("2d");
  probe.filter = "blur(1px)";
  return probe.filter === "blur(1px)";
})();

/* ------------------------------ state ----------------------------------- */

const state = {
  landmarker: null,
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
  lastCycle: -1e9,

  quadAlpha: 0,
  quad: null,

  ripples: [],
  pinchHintShown: false,
  transientHint: null,

  lastVideoTime: -1,
  lastFrameTs: 0,
  fpsAvg: 0,
};

/* ------------------------------ boot ------------------------------------ */

buildChips();
applySmoothing(Number(smoothRange.value));
syncEffectUI();

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
prevBtn.addEventListener("click", () => cycleEffect(-1));
nextBtn.addEventListener("click", () => cycleEffect(1));

mirrorBtn.addEventListener("click", () => {
  state.mirror = !state.mirror;
  mirrorBtn.classList.toggle("is-on", state.mirror);
  mirrorBtn.setAttribute("aria-pressed", String(state.mirror));
  // Mirroring flips x, so drop filter history to avoid a sweep across screen.
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
    return fail("Camera access needs https:// or localhost. Open the page from a local server.");
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
      return fail("Camera permission was blocked. Allow it in the address bar, then try again.");
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
  try {
    await video.play();
  } catch {
    /* some browsers resolve play() late; readiness is awaited below */
  }
  await waitForVideo();
  resize();

  if (!state.landmarker) {
    try {
      setStatus("Loading hand tracking model…", { busy: true });
      const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
      state.landmarker = await createLandmarker(fileset);
    } catch (err) {
      return fail(`Could not load the tracking model: ${err?.message || err}. Check your internet connection.`);
    }
  }

  overlay.classList.add("is-hidden");
  stage.classList.add("is-live");
  hudTop.hidden = false;
  hudBottom.hidden = false;

  if (!SUPPORTS_FILTER) {
    flashHint("This browser ignores canvas filters, so grades show as tints only.", 5200);
  }

  state.running = true;
  state.lastFrameTs = performance.now();
  state.rafId = requestAnimationFrame(loop);
}

/** GPU delegate is much smoother; fall back to CPU where it is unavailable. */
async function createLandmarker(fileset) {
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    return await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

function waitForVideo() {
  if (video.readyState >= 2 && video.videoWidth) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => { video.removeEventListener("loadeddata", done); resolve(); };
    video.addEventListener("loadeddata", done, { once: true });
  });
}

function stop() {
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

  // Run inference only on genuinely new camera frames.
  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    let result = null;
    try {
      result = state.landmarker.detectForVideo(video, now);
    } catch {
      result = null;
    }
    if (result) ingest(result, now);
  }

  // Filters advance every frame, so motion interpolates at display rate.
  advance(dt, now);
  render(now);
}

/* ------------------------------ detection intake ------------------------- */

function ingest(result, now) {
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

/** Distance from a detection to a slot's last known wrist; stale slots are neutral. */
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

/* ------------------------------ per-frame update ------------------------- */

function advance(dt, now) {
  for (const slot of state.slots) {
    if (slot.rawThumb && slot.isLive(now)) {
      // Closeness is always measured on the raw tips. The smoothed corners may
      // be frozen, so they cannot be trusted to report a touch.
      const ratio = tipRatio(slot);
      updateHold(slot, ratio);

      const target = cornerTargets(slot);
      slot.thumb.update(target.thumb.x, target.thumb.y, dt);
      slot.index.update(target.index.x, target.index.y, dt);

      detectPinch(slot, ratio, now);
    } else {
      slot.pinching = false;
      slot.holding = false;
      slot.holdOffsets = null;
    }
    slot.pinchGlow = Math.max(0, slot.pinchGlow - dt * 3.2);
  }

  const [a, b] = state.slots;
  const bothLive = a.isLive(now) && b.isLive(now) && a.thumb.primed && b.thumb.primed;

  // Exponential ease -> framerate independent fade of the whole overlay.
  const target = bothLive ? 1 : 0;
  state.quadAlpha += (target - state.quadAlpha) * (1 - Math.exp(-dt / TUNING.fadeTau));
  if (Math.abs(state.quadAlpha - target) < 0.002) state.quadAlpha = target;

  if (bothLive) {
    state.quad = orderQuad([
      { x: a.thumb.x, y: a.thumb.y },
      { x: a.index.x, y: a.index.y },
      { x: b.index.x, y: b.index.y },
      { x: b.thumb.x, y: b.thumb.y },
    ]);
  } else if (state.quadAlpha <= 0.002) {
    state.quad = null;
  }

  state.ripples = state.ripples.filter((r) => now - r.t0 < 620);

  updateTrackingBadge(now, bothLive);
}

/** Gap between this hand's tips, normalised by hand size so the reading is the
    same whether the hand is near the lens or far from it. */
function tipRatio(slot) {
  const d = Math.hypot(slot.rawThumb.x - slot.rawIndex.x, slot.rawThumb.y - slot.rawIndex.y);
  return d / slot.span;
}

/** Freeze this hand's corners (relative to its wrist) as the tips close in, so
    the act of triggering a change does not flatten the frame. */
function updateHold(slot, ratio) {
  if (!slot.holding && ratio < TUNING.holdOn && slot.thumb.primed) {
    slot.holding = true;
    slot.holdOffsets = {
      thumb: { dx: slot.thumb.x - slot.wrist.x, dy: slot.thumb.y - slot.wrist.y },
      index: { dx: slot.index.x - slot.wrist.x, dy: slot.index.y - slot.wrist.y },
    };
  } else if (slot.holding && ratio > TUNING.holdOff) {
    slot.holding = false;
    slot.holdOffsets = null;
  }
}

/** Where this hand's two corners should sit this frame. */
function cornerTargets(slot) {
  if (slot.holding && slot.holdOffsets) {
    const { thumb, index } = slot.holdOffsets;
    return {
      thumb: { x: slot.wrist.x + thumb.dx, y: slot.wrist.y + thumb.dy },
      index: { x: slot.wrist.x + index.dx, y: slot.wrist.y + index.dy },
    };
  }
  return { thumb: slot.rawThumb, index: slot.rawIndex };
}

/** Tips touching = advance the effect. Hysteresis means one touch counts once. */
function detectPinch(slot, ratio, now) {
  if (!slot.pinching && ratio < TUNING.pinchOn) {
    slot.pinching = true;
    slot.pinchGlow = 1;
    state.ripples.push({
      x: (slot.rawThumb.x + slot.rawIndex.x) / 2,
      y: (slot.rawThumb.y + slot.rawIndex.y) / 2,
      t0: now,
    });
    if (now - state.lastCycle > TUNING.pinchCooldownMs) {
      state.lastCycle = now;
      cycleEffect(1);
    }
  } else if (slot.pinching && ratio > TUNING.pinchOff) {
    slot.pinching = false;
  }
}

/** Sort corners by angle around their centroid so edges never cross. */
function orderQuad(pts) {
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p.x; cy += p.y; }
  cx /= pts.length; cy /= pts.length;
  return pts
    .map((p) => ({ ...p, a: Math.atan2(p.y - cy, p.x - cx) }))
    .sort((m, n) => m.a - n.a);
}

function updateTrackingBadge(now, bothLive) {
  const live = state.slots.filter((s) => s.isLive(now)).length;
  trackDot.classList.toggle("is-locked", bothLive);
  trackDot.classList.toggle("is-lost", live === 0);

  let steady = null;
  if (bothLive) {
    trackText.textContent = "Frame locked";
    if (!state.pinchHintShown) {
      state.pinchHintShown = true;
      flashHint("Touch thumb to index to change the look", 4200);
    }
    if (state.quad && polyArea(state.quad) < TUNING.minAreaFrac * state.W * state.H) {
      steady = "Spread thumb and index apart to open the frame";
    }
  } else if (live === 1) {
    trackText.textContent = "1 hand";
    steady = "Show your other hand";
  } else {
    trackText.textContent = "Searching…";
    steady = "Show both hands, thumb and index spread like an L";
  }

  // A timed message wins while it lasts, then the steady guidance returns.
  const flash = state.transientHint;
  if (flash && now < flash.until) setHint(flash.text);
  else { state.transientHint = null; setHint(steady); }
}

function flashHint(text, ms) {
  state.transientHint = { text, until: performance.now() + ms };
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
    const usable = polyArea(quad) >= TUNING.minAreaFrac * W * H;

    // 2. Graded frame, clipped to the quad.
    if (usable) {
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

    // 3. Frame edges + corners.
    drawFrame(ctx, quad, radius, alpha, now);
  }

  // 4. Optional skeleton + pinch feedback.
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

/** Composite one effect into the offscreen layer at full opacity. */
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

function drawFrame(c, quad, radius, alpha, now) {
  const scale = Math.min(state.W, state.H) / 720;
  const accent = EFFECTS[state.effect].accent;
  const pulse = Math.max(...state.slots.map((s) => s.pinchGlow), 0);

  c.save();
  c.lineJoin = "round";
  c.lineCap = "round";

  // Outer glow
  c.globalAlpha = alpha * (0.4 + pulse * 0.45);
  c.strokeStyle = accent;
  c.shadowColor = accent;
  c.shadowBlur = (18 + pulse * 26) * scale;
  c.lineWidth = (5.5 + pulse * 3) * scale;
  polyPath(c, quad, radius);
  c.stroke();

  // Crisp core line
  c.shadowBlur = 0;
  c.globalAlpha = alpha;
  c.strokeStyle = "rgba(255,255,255,0.94)";
  c.lineWidth = 2 * scale;
  polyPath(c, quad, radius);
  c.stroke();

  // Corner markers
  for (const p of quad) {
    c.globalAlpha = alpha;
    c.beginPath();
    c.arc(p.x, p.y, (7 + pulse * 4) * scale, 0, TWO_PI);
    c.fillStyle = accent;
    c.shadowColor = accent;
    c.shadowBlur = 14 * scale;
    c.fill();

    c.shadowBlur = 0;
    c.beginPath();
    c.arc(p.x, p.y, 3 * scale, 0, TWO_PI);
    c.fillStyle = "#fff";
    c.fill();
  }

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

/** Shoelace area of a closed polygon. */
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
  return {
    x: minX, y: minY, w, h,
    cx: minX + w / 2,
    cy: minY + h / 2,
    r: Math.max(w, h) * 0.62,
  };
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

  const active = chipsEl.children[state.effect];
  active?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
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

/* Debug/tuning handle: lets you inspect live tracking state or force a quad
   from the console, e.g.
     frameChange.state.slots[0].thumb        // smoothed corner
     frameChange.setEffect(3)                // jump to Spotlight            */
window.frameChange = { state, EFFECTS, TUNING, setEffect, render, advance };

document.addEventListener("visibilitychange", () => {
  // Timestamps jump while hidden; clear history so nothing lurches on return.
  if (!document.hidden) {
    state.lastFrameTs = performance.now();
    state.slots.forEach((s) => s.resetFilters());
  }
});

window.addEventListener("pagehide", () => {
  state.stream?.getTracks().forEach((t) => t.stop());
});
