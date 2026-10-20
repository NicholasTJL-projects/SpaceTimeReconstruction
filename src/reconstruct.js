// Reconstruction: background estimate, motion segmentation, depth lookup and
// unprojection of both the still scene and every frame's moving pixels.
import { aiDisparity, groundDisparity, disparityToDepth } from './depth.js';

export const NEAR = 1.0;
export const DEFAULT_FAR = 12.0;

const yieldToUI = () => new Promise((r) => requestAnimationFrame(() => r()));

/** Per-pixel temporal median of up to `maxSamples` evenly spaced frames. */
export function medianBackground(frameSet, maxSamples = 21) {
  const { width, height, frames, count, valid } = frameSet;
  const n = Math.min(maxSamples, count);
  const idx = Array.from({ length: n }, (_, i) => Math.round((i * (count - 1)) / Math.max(1, n - 1)));
  const out = new Uint8ClampedArray(width * height * 4);
  const bgValid = new Uint8Array(width * height);
  const buf = new Uint8Array(n);
  for (let p = 0; p < width * height; p++) {
    let m = 0;
    for (let k = 0; k < n; k++) if (!valid || valid[idx[k]][p]) m++;
    // Need a few real samples for a stable median; otherwise the pixel is unknown.
    if (m < Math.min(3, n)) continue;
    bgValid[p] = 1;
    const mid = m >> 1;
    for (let c = 0; c < 3; c++) {
      const o = p * 4 + c;
      let j = 0;
      for (let k = 0; k < n; k++) if (!valid || valid[idx[k]][p]) buf[j++] = frames[idx[k]][o];
      out[o] = medianU8(buf, m, mid);
    }
    out[p * 4 + 3] = 255;
  }
  return { image: out, valid: bgValid };
}

function medianU8(buf, n, mid) {
  // counting sort on 256 bins is overkill for n<=25; insertion sort is fast here
  for (let i = 1; i < n; i++) {
    const v = buf[i];
    let j = i - 1;
    while (j >= 0 && buf[j] > v) { buf[j + 1] = buf[j]; j--; }
    buf[j + 1] = v;
  }
  return buf[mid];
}

/** Motion mask: mean abs RGB difference vs background > thr, then open+dilate. */
export function motionMask(frame, bg, width, height, thr, frameValid, bgValid) {
  const n = width * height;
  const raw = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    if ((frameValid && !frameValid[p]) || (bgValid && !bgValid[p])) continue;
    const o = p * 4;
    const d = Math.abs(frame[o] - bg[o]) + Math.abs(frame[o + 1] - bg[o + 1]) + Math.abs(frame[o + 2] - bg[o + 2]);
    raw[p] = d > thr * 3 ? 1 : 0;
  }
  const er = erode(raw, width, height);
  const di = dilate(er, width, height);
  return dilate(di, width, height);
}

function erode(src, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    if (src[i] && src[i - 1] && src[i + 1] && src[i - w] && src[i + w]) out[i] = 1;
  }
  return out;
}
function dilate(src, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (src[i] || (x > 0 && src[i - 1]) || (x < w - 1 && src[i + 1]) || (y > 0 && src[i - w]) || (y < h - 1 && src[i + w])) out[i] = 1;
  }
  return out;
}

/** 4-connected components with min area. Returns [{pixels:Int32Array, bottom, cx}] */
export function components(mask, w, h, minArea) {
  const label = new Int32Array(w * h);
  const stack = new Int32Array(w * h);
  const comps = [];
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || label[s]) continue;
    let sp = 0, n = 0;
    const id = comps.length + 1;
    stack[sp++] = s; label[s] = id;
    const pix = [];
    let bottom = 0, sumX = 0;
    while (sp) {
      const i = stack[--sp];
      pix.push(i); n++;
      const x = i % w, y = (i - x) / w;
      if (y > bottom) bottom = y;
      sumX += x;
      if (x > 0 && mask[i - 1] && !label[i - 1]) { label[i - 1] = id; stack[sp++] = i - 1; }
      if (x < w - 1 && mask[i + 1] && !label[i + 1]) { label[i + 1] = id; stack[sp++] = i + 1; }
      if (y > 0 && mask[i - w] && !label[i - w]) { label[i - w] = id; stack[sp++] = i - w; }
      if (y < h - 1 && mask[i + w] && !label[i + w]) { label[i + w] = id; stack[sp++] = i + w; }
    }
    if (n >= minArea) comps.push({ pixels: Int32Array.from(pix), bottom, cx: Math.round(sumX / n) });
  }
  return comps;
}

/**
 * Run the full reconstruction.
 * @returns {Promise<Reconstruction>}
 */
export async function reconstruct(frameSet, opts, report) {
  const { width: w, height: h, frames, count } = frameSet;
  const focal = (w / 2) / Math.tan((opts.fov * Math.PI) / 360);
  const cx = w / 2, cy = h / 2;
  const FAR = Math.max(NEAR * 1.5, opts.far || DEFAULT_FAR);

  report({ phase: 'Estimating still background (temporal median)', frac: 0.02 });
  await yieldToUI();
  const { image: bg, valid: bgValid } = medianBackground(frameSet);
  const frameValid = frameSet.valid || null;

  // Background depth
  let bgDisp;
  if (opts.depthMode === 'ai') {
    report({ phase: 'Loading depth model', frac: 0.05 });
    await yieldToUI();
    bgDisp = await aiDisparity(bg, w, h);
  } else {
    bgDisp = groundDisparity(w, h, { horizon: opts.horizon, near: NEAR, far: FAR, focal });
  }
  report({ phase: 'Unprojecting still scene', frac: 0.2 });
  await yieldToUI();

  // Background point cloud (only pixels the median could establish)
  const bgPosArr = [], bgColArr = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (!bgValid[p]) continue;
    const z = disparityToDepth(bgDisp[p], NEAR, FAR);
    bgPosArr.push(((x + 0.5 - cx) / focal) * z, (-(y + 0.5 - cy) / focal) * z, -z);
    bgColArr.push(bg[p * 4] / 255, bg[p * 4 + 1] / 255, bg[p * 4 + 2] / 255);
  }
  const bgPos = Float32Array.from(bgPosArr), bgCol = Float32Array.from(bgColArr);

  // Subjects, frame by frame
  const subjPos = [], subjCol = [], subjT = [];
  let lastMask = null;
  const perFrameAI = opts.depthMode === 'ai' && opts.subjectDepth === 'perframe';
  const centroidAcc = [0, 0, 0, 0];
  let maskPixels = 0;
  for (let i = 0; i < count; i++) {
    const frame = frames[i];
    const mask = motionMask(frame, bg, w, h, opts.motionThr, frameValid?.[i], bgValid);
    const comps = components(mask, w, h, opts.minBlob);
    lastMask = mask;
    for (const c of comps) maskPixels += c.pixels.length;
    let frameDisp = null;
    if (perFrameAI && comps.length) frameDisp = await aiDisparity(frame, w, h);
    const t = count > 1 ? i / (count - 1) : 0;
    for (const c of comps) {
      // Ground contact: the subject stands on whatever the background is at its lowest pixel.
      const contactIdx = Math.min(h - 1, c.bottom) * w + c.cx;
      const zContact = disparityToDepth(bgDisp[contactIdx], NEAR, FAR);
      for (let k = 0; k < c.pixels.length; k++) {
        const p = c.pixels[k];
        const x = p % w, y = (p - x) / w;
        const z = frameDisp ? disparityToDepth(frameDisp[p], NEAR, FAR) : zContact;
        const X = ((x + 0.5 - cx) / focal) * z, Y = (-(y + 0.5 - cy) / focal) * z, Z = -z;
        subjPos.push(X, Y, Z);
        subjCol.push(frame[p * 4] / 255, frame[p * 4 + 1] / 255, frame[p * 4 + 2] / 255);
        subjT.push(t);
        centroidAcc[0] += X; centroidAcc[1] += Y; centroidAcc[2] += Z; centroidAcc[3]++;
      }
    }
    if (i % 4 === 0) {
      report({ phase: `Reconstructing moving subjects: frame ${i + 1} / ${count}`, frac: 0.2 + (0.78 * (i + 1)) / count });
      await yieldToUI();
    }
  }

  const nSubj = subjT.length;
  const center = nSubj
    ? [centroidAcc[0] / nSubj, centroidAcc[1] / nSubj, centroidAcc[2] / nSubj]
    : sceneCenter(bgPos);
  const extent = nSubj ? percentileDistance(subjPos, center, 0.9) : percentileDistance(bgPos, center, 0.9);
  const maskFraction = maskPixels / (count * w * h);

  report({ phase: 'Done', frac: 1 });
  return {
    width: w, height: h, count, focal, fov: opts.fov,
    background: { positions: bgPos, colors: bgCol, image: bg, disparity: bgDisp },
    subjects: { positions: Float32Array.from(subjPos), colors: Float32Array.from(subjCol), t: Float32Array.from(subjT), count: nSubj },
    center,
    extent,
    maskFraction,
    near: NEAR,
    far: FAR,
    lastMask,
    frameSet,
  };
}

/** Distance from `c` that encloses a given fraction of the points; robust to stray outliers. */
function percentileDistance(pos, c, q) {
  const n = pos.length / 3;
  if (!n) return 0;
  const step = Math.max(1, Math.floor(n / 20000));
  const d = [];
  for (let i = 0; i < n; i += step) {
    const dx = pos[i * 3] - c[0], dy = pos[i * 3 + 1] - c[1], dz = pos[i * 3 + 2] - c[2];
    d.push(dx * dx + dy * dy + dz * dz);
  }
  d.sort((a, b) => a - b);
  return Math.sqrt(d[Math.min(d.length - 1, Math.floor(d.length * q))]);
}

function sceneCenter(pos) {
  let x = 0, y = 0, z = 0;
  const n = pos.length / 3;
  for (let i = 0; i < n; i++) { x += pos[i * 3]; y += pos[i * 3 + 1]; z += pos[i * 3 + 2]; }
  return [x / n, y / n, z / n];
}

/** ASCII PLY export of scene + subjects, with a per-vertex time property. */
export function toPLY(recon) {
  const bgN = recon.background.positions.length / 3;
  const sN = recon.subjects.count;
  const lines = [
    'ply', 'format ascii 1.0', `element vertex ${bgN + sN}`,
    'property float x', 'property float y', 'property float z',
    'property uchar red', 'property uchar green', 'property uchar blue',
    'property float time', 'end_header',
  ];
  const push = (pos, col, i, t) => {
    lines.push(`${pos[i * 3].toFixed(4)} ${pos[i * 3 + 1].toFixed(4)} ${pos[i * 3 + 2].toFixed(4)} ${Math.round(col[i * 3] * 255)} ${Math.round(col[i * 3 + 1] * 255)} ${Math.round(col[i * 3 + 2] * 255)} ${t.toFixed(4)}`);
  };
  for (let i = 0; i < bgN; i++) push(recon.background.positions, recon.background.colors, i, -1);
  for (let i = 0; i < sN; i++) push(recon.subjects.positions, recon.subjects.colors, i, recon.subjects.t[i]);
  return lines.join('\n') + '\n';
}
