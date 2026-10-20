// Depth providers. Each returns a Float32Array of normalized disparity in
// [0, 1] (1 = nearest) for an RGBA frame at width x height.

let depthPipeline = null;
let loadingPromise = null;

export function hasWebGPU() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

/** navigator.gpu can exist while no adapter is usable; probe before trusting it. */
async function pickDevice() {
  if (!hasWebGPU()) return 'wasm';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? 'webgpu' : 'wasm';
  } catch {
    return 'wasm';
  }
}

/**
 * Lazily load Depth Anything V2 (small) through Transformers.js. The model is
 * fetched from the Hugging Face hub and cached by the browser.
 */
export async function loadAIDepth(onProgress) {
  if (depthPipeline) return depthPipeline;
  if (!loadingPromise) {
    loadingPromise = (async () => {
      const tf = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3/dist/transformers.min.js');
      tf.env.allowLocalModels = false;
      let device = await pickDevice();
      const files = new Map();
      const progress_callback = (p) => {
        if (p.status === 'progress' && p.file) {
          files.set(p.file, [p.loaded || 0, p.total || 0]);
          let loaded = 0, total = 0;
          for (const [l, t] of files.values()) { loaded += l; total += t; }
          onProgress?.({ phase: 'download', frac: total ? loaded / total : 0, mb: loaded / 1048576 });
        } else if (p.status === 'ready') {
          onProgress?.({ phase: 'ready', frac: 1 });
        }
      };
      const make = (dev) => tf.pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', { device: dev, progress_callback });
      let pipe;
      try {
        pipe = await make(device);
      } catch (e) {
        if (device === 'webgpu') { console.warn('WebGPU backend failed, falling back to WASM', e); device = 'wasm'; pipe = await make(device); }
        else throw e;
      }
      depthPipeline = { pipe, tf, device };
      return depthPipeline;
    })().catch((e) => { loadingPromise = null; throw e; });
  }
  return loadingPromise;
}

export async function aiDisparity(rgba, width, height) {
  const { pipe, tf } = await loadAIDepth();
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  const image = tf.RawImage.fromCanvas(canvas);
  const out = await pipe(image);
  const t = out.predicted_depth; // Float32 tensor [H, W] at input size
  const [th, tw] = t.dims.slice(-2);
  const data = t.data;
  const disp = new Float32Array(width * height);
  // Robust normalization: clip to the 1st..99th percentile so a few outliers
  // do not squash the rest of the scene into a thin slab.
  const sorted = Float32Array.from(data).sort();
  const mn = sorted[Math.floor(sorted.length * 0.01)];
  const mx = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.99))];
  const range = mx - mn || 1;
  // Resample (nearest) in case dims differ from our working size.
  for (let y = 0; y < height; y++) {
    const sy = Math.min(th - 1, Math.round((y * th) / height));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(tw - 1, Math.round((x * tw) / width));
      disp[y * width + x] = Math.min(1, Math.max(0, (data[sy * tw + sx] - mn) / range));
    }
  }
  return disp;
}

/**
 * Ground-plane heuristic: everything below the horizon is a flat floor seen by a
 * level camera; everything above is a distant backdrop. Returns disparity in
 * [0, 1] consistent with a near/far range of [near, far].
 */
export function groundDisparity(width, height, { horizon = 0.4, near = 1, far = 1, focal }) {
  const disp = new Float32Array(width * height);
  const vH = horizon * height;
  const camHeight = 1.0;
  const invNear = 1 / near, invFar = 1 / far;
  for (let y = 0; y < height; y++) {
    let z;
    if (y <= vH + 0.5) z = far;
    else z = (focal * camHeight) / (y + 0.5 - vH);
    z = Math.min(far, Math.max(near, z));
    const d = (1 / z - invFar) / (invNear - invFar);
    for (let x = 0; x < width; x++) disp[y * width + x] = d;
  }
  return disp;
}

export function disparityToDepth(d, near, far) {
  const invNear = 1 / near, invFar = 1 / far;
  return 1 / (invFar + d * (invNear - invFar));
}
