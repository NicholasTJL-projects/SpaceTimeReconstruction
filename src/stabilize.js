// Camera-motion compensation for handheld or panning footage.
//
// Every sampled frame is aligned to the first one with a 2D similarity
// transform (scale, rotation, translation) estimated by coarse-to-fine
// inverse-compositional Lucas-Kanade on grayscale pyramids, with a Huber
// weight so a moving subject does not drag the estimate. Frames are then
// resampled into the first frame's coordinates; pixels that fall outside the
// source are marked invalid so the median and motion mask can ignore them.

const HUBER = 18;

/** @param {Uint8ClampedArray} rgba */
function toGray(rgba, w, h) {
  const g = new Float32Array(w * h);
  for (let p = 0, o = 0; p < w * h; p++, o += 4) g[p] = 0.299 * rgba[o] + 0.587 * rgba[o + 1] + 0.114 * rgba[o + 2];
  return g;
}

function downsample(src, w, h) {
  const w2 = w >> 1, h2 = h >> 1;
  const out = new Float32Array(w2 * h2);
  for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
    const i = (2 * y) * w + 2 * x;
    out[y * w2 + x] = 0.25 * (src[i] + src[i + 1] + src[i + w] + src[i + w + 1]);
  }
  return out;
}

function pyramid(gray, w, h, minWidth = 32) {
  const levels = [{ g: gray, w, h }];
  while (levels[levels.length - 1].w >= minWidth * 2) {
    const l = levels[levels.length - 1];
    levels.push({ g: downsample(l.g, l.w, l.h), w: l.w >> 1, h: l.h >> 1 });
  }
  return levels; // index 0 = finest
}

/** Precompute steepest-descent images and inverse Hessian for a template level. */
function templateLevel(g, w, h) {
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  const n = w * h;
  const sd = new Float32Array(n * 4);
  const H = new Float64Array(16);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const tx = 0.5 * (g[i + 1] - g[i - 1]);
    const ty = 0.5 * (g[i + w] - g[i - w]);
    const rx = x - cx, ry = y - cy;
    const s0 = tx * rx + ty * ry, s1 = -tx * ry + ty * rx, s2 = tx, s3 = ty;
    sd[i * 4] = s0; sd[i * 4 + 1] = s1; sd[i * 4 + 2] = s2; sd[i * 4 + 3] = s3;
    const v = [s0, s1, s2, s3];
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) H[a * 4 + b] += v[a] * v[b];
  }
  const h22 = H[10] * H[15] - H[11] * H[14];
  const Hinv2 = Math.abs(h22) > 1e-9 ? [H[15] / h22, -H[11] / h22, -H[14] / h22, H[10] / h22] : null;
  return { g, w, h, cx, cy, sd, Hinv: invert4(H), Hinv2 };
}

function invert4(m) {
  // Gauss-Jordan with partial pivoting on a 4x4 (Float64Array of 16).
  const a = Float64Array.from(m), inv = new Float64Array(16);
  for (let i = 0; i < 4; i++) inv[i * 4 + i] = 1;
  for (let c = 0; c < 4; c++) {
    let piv = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(a[r * 4 + c]) > Math.abs(a[piv * 4 + c])) piv = r;
    if (Math.abs(a[piv * 4 + c]) < 1e-12) return null;
    if (piv !== c) for (let k = 0; k < 4; k++) { [a[c * 4 + k], a[piv * 4 + k]] = [a[piv * 4 + k], a[c * 4 + k]]; [inv[c * 4 + k], inv[piv * 4 + k]] = [inv[piv * 4 + k], inv[c * 4 + k]]; }
    const d = a[c * 4 + c];
    for (let k = 0; k < 4; k++) { a[c * 4 + k] /= d; inv[c * 4 + k] /= d; }
    for (let r = 0; r < 4; r++) if (r !== c) {
      const f = a[r * 4 + c];
      if (f) for (let k = 0; k < 4; k++) { a[r * 4 + k] -= f * a[c * 4 + k]; inv[r * 4 + k] -= f * inv[c * 4 + k]; }
    }
  }
  return inv;
}

// Similarity transform as affine {a, b, c, d, tx, ty}: [x', y'] = [a b; c d] [x y] + [tx ty], centered coords.
const identity = () => ({ a: 1, b: 0, c: 0, d: 1, tx: 0, ty: 0 });
function compose(m1, m2) { // m1 ∘ m2
  return {
    a: m1.a * m2.a + m1.b * m2.c, b: m1.a * m2.b + m1.b * m2.d,
    c: m1.c * m2.a + m1.d * m2.c, d: m1.c * m2.b + m1.d * m2.d,
    tx: m1.a * m2.tx + m1.b * m2.ty + m1.tx, ty: m1.c * m2.tx + m1.d * m2.ty + m1.ty,
  };
}
function inverse(m) {
  const det = m.a * m.d - m.b * m.c;
  const a = m.d / det, b = -m.b / det, c = -m.c / det, d = m.a / det;
  return { a, b, c, d, tx: -(a * m.tx + b * m.ty), ty: -(c * m.tx + d * m.ty) };
}
function scaleTranslation(m, f) { return { ...m, tx: m.tx * f, ty: m.ty * f }; }

function sampleBilinear(g, w, h, x, y) {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) return NaN;
  const fx = x - x0, fy = y - y0;
  const i = y0 * w + x0;
  return (g[i] * (1 - fx) + g[i + 1] * fx) * (1 - fy) + (g[i + w] * (1 - fx) + g[i + w + 1] * fx) * fy;
}

/** Align image pyramid `img` to template pyramid `tpl`; returns transform in finest-level pixels. */
function align(tplLevels, imgLevels, init, iters = 20, model = 'translation') {
  let m = scaleTranslation(init, 1 / 2 ** (tplLevels.length - 1));
  let overlap = 1;
  for (let L = tplLevels.length - 1; L >= 0; L--) {
    const T = tplLevels[L], I = imgLevels[L];
    if (!T.Hinv) continue;
    const { w, h, cx, cy, sd, g: tg } = T;
    for (let it = 0; it < iters; it++) {
      const b = new Float64Array(4);
      let used = 0, total = 0;
      for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
        total++;
        const rx = x - cx, ry = y - cy;
        const sx = m.a * rx + m.b * ry + m.tx + cx, sy = m.c * rx + m.d * ry + m.ty + cy;
        const v = sampleBilinear(I.g, I.w, I.h, sx, sy);
        if (Number.isNaN(v)) continue;
        used++;
        const i = y * w + x;
        let e = v - tg[i];
        const ae = Math.abs(e);
        if (ae > 4 * HUBER) continue; // a moving object, not camera motion
        const wgt = ae > HUBER ? HUBER / ae : 1;
        e *= wgt;
        b[0] += sd[i * 4] * e; b[1] += sd[i * 4 + 1] * e; b[2] += sd[i * 4 + 2] * e; b[3] += sd[i * 4 + 3] * e;
      }
      overlap = used / Math.max(1, total);
      if (used < 64) break;
      const dp = [0, 0, 0, 0];
      if (model === 'similarity') {
        const Hi = T.Hinv;
        for (let r = 0; r < 4; r++) dp[r] = Hi[r * 4] * b[0] + Hi[r * 4 + 1] * b[1] + Hi[r * 4 + 2] * b[2] + Hi[r * 4 + 3] * b[3];
      } else {
        const Hi = T.Hinv2;
        if (!Hi) break;
        dp[2] = Hi[0] * b[2] + Hi[1] * b[3];
        dp[3] = Hi[2] * b[2] + Hi[3] * b[3];
      }
      const ds = dp[0], dth = dp[1];
      const cs = Math.cos(dth), sn = Math.sin(dth), s = 1 + ds;
      const delta = { a: s * cs, b: -s * sn, c: s * sn, d: s * cs, tx: dp[2], ty: dp[3] };
      m = compose(m, inverse(delta));
      if (Math.abs(dp[2]) < 0.02 && Math.abs(dp[3]) < 0.02 && Math.abs(ds) < 1e-4 && Math.abs(dth) < 1e-4) break;
    }
    if (L > 0) m = scaleTranslation(m, 2);
  }
  return { m, overlap };
}

/** Brute-force integer translation at the coarsest level: a robust starting point for fast pans. */
function coarseTranslation(tplLevel, imgLevel) {
  const { g: T, w, h } = tplLevel, I = imgLevel.g;
  const R = Math.max(2, Math.round(0.3 * w));
  let best = 0, bestScore = Infinity;
  for (let dy = -Math.round(0.3 * h); dy <= Math.round(0.3 * h); dy++) for (let dx = -R; dx <= R; dx++) {
    let sum = 0, n = 0;
    for (let y = Math.max(0, -dy); y < Math.min(h, h - dy); y += 2) for (let x = Math.max(0, -dx); x < Math.min(w, w - dx); x += 2) {
      sum += Math.abs(I[(y + dy) * w + x + dx] - T[y * w + x]); n++;
    }
    if (n < 0.35 * (w * h) / 4) continue;
    const score = sum / n;
    if (score < bestScore) { bestScore = score; best = { dx, dy }; }
  }
  return best || { dx: 0, dy: 0 };
}

function sane(m, prev = identity(), w) {
  const rel = compose(m, inverse(prev));
  const s = Math.hypot(rel.a, rel.c), th = Math.abs(Math.atan2(rel.c, rel.a));
  return s > 0.8 && s < 1.25 && th < 0.18 && Math.abs(rel.tx) < 0.6 * w && Math.abs(rel.ty) < 0.6 * w;
}

/**
 * Stabilize a FrameSet. Every frame is aligned to the first and resampled
 * onto a mosaic canvas large enough to hold the whole pan (bounded to about
 * 2.5x the frame area; the canvas is downscaled beyond that and `focalScale`
 * reports the factor). The result carries `valid` masks, the per-frame
 * transforms and the principal point of the first frame inside the canvas.
 */
export async function stabilize(frameSet, { onProgress, maxAreaScale = 2.5, model = 'translation' } = {}) {
  const { width: w, height: h, frames, count } = frameSet;
  const grays = frames.map((f) => toGray(f, w, h));
  const pyrs = grays.map((g) => pyramid(g, w, h));
  const tplOf = (pyr) => pyr.map((l) => templateLevel(l.g, l.w, l.h));
  const tpl0 = tplOf(pyrs[0]);
  const transforms = [identity()];
  // Align against a keyframe rather than the previous frame, so drift only
  // accumulates when the view has moved far enough to need a new keyframe.
  let key = { tpl: tpl0, global: identity(), toKey: identity() };
  const L = tpl0.length - 1;
  const alignTo = (k, i, initRel) => {
    const t = coarseTranslation(k.tpl[L], pyrs[i][L]);
    const init = { ...initRel, tx: initRel.tx + 0, ty: initRel.ty + 0 };
    const fromSearch = { ...identity(), tx: t.dx * 2 ** L, ty: t.dy * 2 ** L };
    const a = align(k.tpl, pyrs[i], fromSearch, 20, model);
    const b = Math.abs(init.tx - fromSearch.tx) + Math.abs(init.ty - fromSearch.ty) > 2 ** L ? align(k.tpl, pyrs[i], init, 20, model) : null;
    // Prefer the candidate with more overlap once both are sane.
    let best = a;
    if (b && (!sane(a.m, identity(), w) || (sane(b.m, identity(), w) && b.overlap > a.overlap))) best = b;
    if (!sane(best.m, identity(), w)) best = { m: fromSearch, overlap: 0 };
    return best;
  };
  for (let i = 1; i < count; i++) {
    let res = alignTo(key, i, key.toKey);
    if (res.overlap < 0.55) {
      // View has moved on: promote the previous frame to keyframe and align to it instead.
      key = { tpl: tplOf(pyrs[i - 1]), global: transforms[i - 1], toKey: identity() };
      res = alignTo(key, i, identity());
    }
    key.toKey = res.m;
    const global = compose(res.m, key.global);
    transforms.push(global);
    if (i % 3 === 0) {
      onProgress?.((0.7 * i) / (count - 1));
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  // Mosaic canvas: bounding box, in first-frame centered coordinates, of every warped frame.
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  let minX = -cx, maxX = cx, minY = -cy, maxY = cy;
  for (const m of transforms) {
    const inv = inverse(m);
    for (const [px, py] of [[-cx, -cy], [cx, -cy], [cx, cy], [-cx, cy]]) {
      const x = inv.a * px + inv.b * py + inv.tx, y = inv.c * px + inv.d * py + inv.ty;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  }
  // Never let a runaway transform explode the canvas: clamp to 3x the frame in each direction.
  minX = Math.max(minX, -cx - w); maxX = Math.min(maxX, cx + w); minY = Math.max(minY, -cy - h); maxY = Math.min(maxY, cy + h);
  const fullW = maxX - minX + 1, fullH = maxY - minY + 1;
  const k = Math.min(1, Math.sqrt((maxAreaScale * w * h) / (fullW * fullH)));
  const W = Math.max(2, Math.round(fullW * k)) & ~1, H = Math.max(2, Math.round(fullH * k)) & ~1;
  const principal = [(-minX) * k, (-minY) * k];

  const outFrames = [], valids = [];
  for (let i = 0; i < count; i++) {
    const m = transforms[i];
    const rgba = new Uint8ClampedArray(W * H * 4), valid = new Uint8Array(W * H);
    const src = frames[i];
    for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
      const x = u / k + minX, y = v / k + minY; // first-frame centered coords
      const sx = m.a * x + m.b * y + m.tx + cx, sy = m.c * x + m.d * y + m.ty + cy;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) continue;
      const fx = sx - x0, fy = sy - y0;
      const s0 = (y0 * w + x0) * 4, o = (v * W + u) * 4;
      for (let c = 0; c < 3; c++) {
        rgba[o + c] = (src[s0 + c] * (1 - fx) + src[s0 + 4 + c] * fx) * (1 - fy) + (src[s0 + w * 4 + c] * (1 - fx) + src[s0 + w * 4 + 4 + c] * fx) * fy;
      }
      rgba[o + 3] = 255;
      valid[v * W + u] = 1;
    }
    outFrames.push(rgba);
    valids.push(valid);
    if (i % 6 === 0) {
      onProgress?.(0.7 + (0.3 * i) / count);
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  return { ...frameSet, width: W, height: H, frames: outFrames, valid: valids, transforms, principal, focalScale: k, stabilized: true };
}
