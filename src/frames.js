// Frame sources: sample a video file into a fixed number of RGBA frames at a
// working resolution, or synthesize a demo clip procedurally.

/**
 * @typedef {Object} FrameSet
 * @property {number} width
 * @property {number} height
 * @property {number} count
 * @property {Uint8ClampedArray[]} frames  RGBA, width*height*4 each
 * @property {number[]} times              source timestamps in seconds
 * @property {number} sourceFps            frames per second of sampled frames in source time
 */

function workingSize(srcW, srcH, workWidth) {
  const w = Math.min(workWidth, srcW);
  const h = Math.max(16, Math.round((w * srcH) / srcW));
  return [w, h & ~1];
}

export async function loadVideo(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = url;
  await new Promise((res, rej) => {
    video.onloadedmetadata = () => res();
    video.onerror = () => rej(new Error('Could not decode this video in the browser. Try an H.264 .mp4 or a .webm.'));
  });
  if (!isFinite(video.duration) || video.duration <= 0) {
    // Some recordings report Infinity until we seek to the end.
    video.currentTime = 1e9;
    await new Promise((res) => (video.onseeked = () => res()));
    video.currentTime = 0;
  }
  return { video, url };
}

/**
 * Sample `count` frames evenly across the video.
 * @returns {Promise<FrameSet>}
 */
export async function sampleVideo(video, { count, workWidth, onProgress }) {
  const [w, h] = workingSize(video.videoWidth, video.videoHeight, workWidth);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const duration = video.duration;
  const frames = [];
  const times = [];
  const margin = Math.min(0.05, duration * 0.02);
  for (let i = 0; i < count; i++) {
    const t = margin + ((duration - 2 * margin) * i) / Math.max(1, count - 1);
    await seek(video, t);
    ctx.drawImage(video, 0, 0, w, h);
    frames.push(ctx.getImageData(0, 0, w, h).data.slice());
    times.push(t);
    onProgress?.((i + 1) / count);
  }
  const sourceFps = count > 1 ? (count - 1) / (times[count - 1] - times[0]) : 1;
  return { width: w, height: h, count, frames, times, sourceFps };
}

function seek(video, t) {
  return new Promise((res, rej) => {
    const done = () => {
      video.removeEventListener('seeked', done);
      // Give the decoder one frame to present the seeked picture.
      requestAnimationFrame(() => res());
    };
    video.addEventListener('seeked', done);
    video.onerror = () => rej(new Error('Video decode error while seeking.'));
    video.currentTime = t;
  });
}

/**
 * Synthesize a demo clip: a dim lawn in front of a tree line at dusk, with a
 * deer-like silhouette walking across. Exists so the app can be tried without
 * any footage.
 * @returns {FrameSet}
 */
export function makeDemoClip({ count = 72, workWidth = 320 } = {}) {
  const w = workWidth;
  const h = Math.round((w * 9) / 16) & ~1;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const frames = [];
  const times = [];
  const rng = mulberry32(7);
  const trunks = Array.from({ length: 14 }, () => ({ x: rng() * w, wdt: 3 + rng() * 6, hgt: 0.25 + rng() * 0.2 }));
  const tufts = Array.from({ length: 260 }, () => ({ x: rng() * w, y: h * 0.42 + rng() * h * 0.58, s: 0.5 + rng() }));
  const horizon = h * 0.42;

  for (let i = 0; i < count; i++) {
    const u = i / (count - 1);
    // Scene
    const sky = ctx.createLinearGradient(0, 0, 0, horizon);
    sky.addColorStop(0, '#0d1a1c');
    sky.addColorStop(1, '#1e3a2f');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, horizon);
    // tree canopy noise
    ctx.fillStyle = '#152a22';
    for (let k = 0; k < 40; k++) {
      const cx = ((k * 97) % w) + 10, cy = horizon * (0.2 + ((k * 37) % 10) / 14), r = 14 + ((k * 13) % 20);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    }
    for (const t of trunks) {
      ctx.fillStyle = '#2a2420';
      ctx.fillRect(t.x, horizon - t.hgt * h, t.wdt, t.hgt * h + 2);
    }
    const lawn = ctx.createLinearGradient(0, horizon, 0, h);
    lawn.addColorStop(0, '#48521f');
    lawn.addColorStop(1, '#8a9a3a');
    ctx.fillStyle = lawn;
    ctx.fillRect(0, horizon, w, h - horizon);
    ctx.fillStyle = 'rgba(255,255,200,0.08)';
    for (const t of tufts) ctx.fillRect(t.x, t.y, 2 * t.s, 1.2 * t.s);
    // spotlight vignette like a trail cam IR lamp
    const vg = ctx.createRadialGradient(w * 0.55, h * 0.7, h * 0.2, w * 0.55, h * 0.7, w * 0.75);
    vg.addColorStop(0, 'rgba(0,0,0,0)');
    vg.addColorStop(1, 'rgba(0,0,0,0.7)');
    ctx.fillStyle = vg;
    ctx.fillRect(0, 0, w, h);

    // Subject: deer walking right to left along a gentle arc, getting closer.
    const px = w * (0.95 - 0.9 * u);
    const depthFrac = 0.35 + 0.45 * Math.sin(u * Math.PI); // 0 far .. 1 near
    const footY = horizon + (h - horizon) * (0.18 + 0.72 * depthFrac);
    const scale = 0.35 + 0.9 * depthFrac;
    drawDeer(ctx, px, footY, scale, i);

    frames.push(ctx.getImageData(0, 0, w, h).data.slice());
    times.push(i / 12);
  }
  return { width: w, height: h, count, frames, times, sourceFps: 12 };
}

function drawDeer(ctx, x, footY, s, i) {
  const bodyH = 26 * s, bodyW = 44 * s, legH = 26 * s;
  const bob = Math.sin(i * 0.9) * 1.5 * s;
  ctx.save();
  ctx.translate(x, footY + bob);
  ctx.fillStyle = '#d9c7a6';
  // legs
  const stride = Math.sin(i * 0.9) * 6 * s;
  for (const [lx, ph] of [[-bodyW * 0.32, 1], [-bodyW * 0.18, -1], [bodyW * 0.18, 1], [bodyW * 0.32, -1]]) {
    ctx.fillRect(lx + ph * stride * 0.5 - 2 * s, -legH, 4 * s, legH);
  }
  // body
  ctx.beginPath();
  ctx.ellipse(0, -legH - bodyH * 0.5, bodyW * 0.5, bodyH * 0.5, 0, 0, Math.PI * 2);
  ctx.fill();
  // neck + head (facing left)
  ctx.beginPath();
  ctx.moveTo(-bodyW * 0.35, -legH - bodyH * 0.6);
  ctx.lineTo(-bodyW * 0.62, -legH - bodyH * 1.35);
  ctx.lineTo(-bodyW * 0.42, -legH - bodyH * 1.3);
  ctx.lineTo(-bodyW * 0.25, -legH - bodyH * 0.4);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(-bodyW * 0.62, -legH - bodyH * 1.4, 9 * s, 5 * s, -0.3, 0, Math.PI * 2);
  ctx.fill();
  // ears
  ctx.fillRect(-bodyW * 0.6, -legH - bodyH * 1.65, 2.5 * s, 8 * s);
  ctx.fillRect(-bodyW * 0.52, -legH - bodyH * 1.65, 2.5 * s, 8 * s);
  ctx.restore();
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
