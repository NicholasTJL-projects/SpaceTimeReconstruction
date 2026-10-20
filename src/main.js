import { loadVideo, sampleVideo, makeDemoClip } from './frames.js';
import { reconstruct, toPLY } from './reconstruct.js';
import { loadAIDepth } from './depth.js';
import { Viewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const els = {
  file: $('file'), drop: $('drop'), dropText: $('dropText'), demoBtn: $('demoBtn'),
  frameCount: $('frameCount'), workWidth: $('workWidth'), depthMode: $('depthMode'), subjectDepth: $('subjectDepth'),
  motionThr: $('motionThr'), minBlob: $('minBlob'), fov: $('fov'), far: $('far'), horizon: $('horizon'),
  runBtn: $('runBtn'), bar: $('bar'), status: $('status'),
  reveal: $('reveal'), trail: $('trail'), subjSize: $('subjSize'), bgSize: $('bgSize'), bgDim: $('bgDim'), extrude: $('extrude'),
  fps: $('fps'), orbitSpeed: $('orbitSpeed'), aspect: $('aspect'), autoOrbit: $('autoOrbit'), showFrustum: $('showFrustum'), showCaption: $('showCaption'),
  recordBtn: $('recordBtn'), snapBtn: $('snapBtn'), exportPlyBtn: $('exportPlyBtn'),
  view: $('view'), viewerWrap: $('viewerWrap'), caption: $('caption'), recDot: $('recDot'), empty: $('empty'),
  playBtn: $('playBtn'), timeline: $('timeline'), frameLabel: $('frameLabel'),
  bgCanvas: $('bgCanvas'), depthCanvas: $('depthCanvas'), maskCanvas: $('maskCanvas'),
};

const state = { video: null, videoName: '', demo: false, frameSet: null, recon: null, busy: false };
const viewer = new Viewer(els.view, els.viewerWrap);
viewer.onUserInteract = () => { els.autoOrbit.checked = false; viewer.autoOrbit = false; };
viewer.onFrame = (i, n) => { els.timeline.value = i; els.frameLabel.textContent = `${i + 1} / ${n}`; };
viewer.onEnded = () => { els.playBtn.textContent = '▶'; };

function setStatus(msg) { els.status.textContent = msg; }
function setProgress(frac) { els.bar.style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`; }

// --- live outputs for sliders
for (const [id, fmt] of [['motionThr'], ['minBlob'], ['fov'], ['far'], ['horizon', (v) => Number(v).toFixed(2)], ['trail', (v) => (Number(v) >= 1 ? 'all' : Number(v).toFixed(2))],
  ['subjSize'], ['bgSize'], ['bgDim'], ['extrude'], ['fps'], ['orbitSpeed']]) {
  const el = $(id), out = $(id + 'Out');
  const upd = () => (out.textContent = fmt ? fmt(el.value) : el.value);
  el.addEventListener('input', upd); upd();
}

function viewSettings() {
  return {
    reveal: els.reveal.value, trail: Number(els.trail.value), subjSize: Number(els.subjSize.value), bgSize: Number(els.bgSize.value),
    bgDim: Number(els.bgDim.value), extrude: Number(els.extrude.value), fps: Number(els.fps.value), orbitSpeed: Number(els.orbitSpeed.value),
    autoOrbit: els.autoOrbit.checked, showFrustum: els.showFrustum.checked,
  };
}
for (const id of ['reveal', 'trail', 'subjSize', 'bgSize', 'bgDim', 'extrude', 'fps', 'orbitSpeed', 'autoOrbit', 'showFrustum']) {
  $(id).addEventListener('input', () => viewer.setSettings(viewSettings()));
}
els.showCaption.addEventListener('input', () => (els.caption.style.display = els.showCaption.checked ? '' : 'none'));
els.aspect.addEventListener('input', () => viewer.setAspect(els.aspect.value));
els.depthMode.addEventListener('input', () => { els.subjectDepth.disabled = els.depthMode.value !== 'ai'; });

// --- video input
function acceptFile(file) {
  if (!file) return;
  state.demo = false;
  state.videoName = file.name;
  els.dropText.textContent = `${file.name} (${(file.size / 1048576).toFixed(1)} MB)`;
  loadVideo(file).then(({ video }) => {
    state.video = video;
    const d = video.duration;
    setStatus(`Loaded ${video.videoWidth}×${video.videoHeight}, ${d.toFixed(1)} s. Sampling ${els.frameCount.value} frames ≈ ${(Number(els.frameCount.value) / d).toFixed(1)} fps.`);
    els.runBtn.disabled = false;
  }).catch((e) => { setStatus(e.message); els.runBtn.disabled = true; });
}
els.file.addEventListener('change', () => acceptFile(els.file.files[0]));
els.drop.addEventListener('dragover', (e) => { e.preventDefault(); els.drop.classList.add('over'); });
els.drop.addEventListener('dragleave', () => els.drop.classList.remove('over'));
els.drop.addEventListener('drop', (e) => { e.preventDefault(); els.drop.classList.remove('over'); acceptFile(e.dataTransfer.files[0]); });
els.demoBtn.addEventListener('click', () => {
  state.demo = true; state.video = null; state.videoName = 'demo';
  els.dropText.textContent = 'Synthetic demo clip (a deer-like silhouette crossing a lawn)';
  setStatus('Demo clip ready. Ground-plane depth is enough for it; AI depth also works.');
  els.runBtn.disabled = false;
});

// --- reconstruction
els.runBtn.addEventListener('click', async () => {
  if (state.busy) return;
  state.busy = true;
  els.runBtn.disabled = true;
  els.recordBtn.disabled = els.snapBtn.disabled = els.exportPlyBtn.disabled = true;
  viewer.playing = false;
  const opts = {
    fov: Number(els.fov.value), far: Number(els.far.value), horizon: Number(els.horizon.value), depthMode: els.depthMode.value, subjectDepth: els.subjectDepth.value,
    motionThr: Number(els.motionThr.value), minBlob: Number(els.minBlob.value),
  };
  const count = Math.max(8, Math.min(400, Number(els.frameCount.value) || 72));
  const workWidth = Number(els.workWidth.value);
  try {
    if (opts.depthMode === 'ai') {
      setStatus('Loading depth model…');
      const { device } = await loadAIDepth((p) => {
        if (p.phase === 'download') { setProgress(p.frac * 0.15); setStatus(`Downloading depth model… ${p.mb.toFixed(0)} MB`); }
      });
      setStatus(`Depth model ready on ${device === 'webgpu' ? 'WebGPU' : 'WASM'}.`);
    }
    setStatus('Sampling frames…');
    if (state.demo) {
      state.frameSet = makeDemoClip({ count, workWidth });
    } else {
      state.frameSet = await sampleVideo(state.video, { count, workWidth, onProgress: (f) => setProgress(0.15 + f * 0.2) });
    }
    const t0 = performance.now();
    state.recon = await reconstruct(state.frameSet, opts, (p) => { setStatus(p.phase); setProgress(0.35 + p.frac * 0.65); });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    showIntermediate(state.recon);
    viewer.load(state.recon, viewSettings());
    els.empty.hidden = true;
    els.timeline.max = state.recon.count - 1;
    els.timeline.value = 0;
    els.frameLabel.textContent = `1 / ${state.recon.count}`;
    const subjPts = state.recon.subjects.count;
    setStatus(`Done in ${secs} s. Scene: ${(state.recon.width * state.recon.height).toLocaleString()} points. Subjects: ${subjPts.toLocaleString()} points across ${state.recon.count} moments.` +
      (subjPts === 0 ? '\nNo moving pixels found. Lower the motion threshold or min blob size.' : ''));
    els.recordBtn.disabled = els.snapBtn.disabled = els.exportPlyBtn.disabled = false;
    viewer.playing = true;
    els.playBtn.textContent = '❚❚';
  } catch (e) {
    console.error(e);
    setStatus(`Failed: ${e.message || e}`);
  } finally {
    state.busy = false;
    els.runBtn.disabled = false;
  }
});

function showIntermediate(recon) {
  const { width: w, height: h } = recon;
  const draw = (canvas, rgba) => {
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').putImageData(new ImageData(rgba, w, h), 0, 0);
  };
  draw(els.bgCanvas, recon.background.image);
  const dep = new Uint8ClampedArray(w * h * 4);
  for (let p = 0; p < w * h; p++) { const v = recon.background.disparity[p] * 255; dep[p * 4] = dep[p * 4 + 1] = dep[p * 4 + 2] = v; dep[p * 4 + 3] = 255; }
  draw(els.depthCanvas, dep);
  const m = new Uint8ClampedArray(w * h * 4);
  for (let p = 0; p < w * h; p++) { const v = recon.lastMask?.[p] ? 255 : 0; m[p * 4] = m[p * 4 + 1] = m[p * 4 + 2] = v; m[p * 4 + 3] = 255; }
  draw(els.maskCanvas, m);
}

// --- transport
els.playBtn.addEventListener('click', () => {
  if (!state.recon) return;
  if (!viewer.playing && viewer.time >= 1) viewer.setFrame(0);
  viewer.playing = !viewer.playing;
  els.playBtn.textContent = viewer.playing ? '❚❚' : '▶';
});
els.timeline.addEventListener('input', () => { viewer.playing = false; els.playBtn.textContent = '▶'; viewer.setFrame(Number(els.timeline.value)); });

// --- export
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
const baseName = () => (state.videoName || 'scene').replace(/\.[^.]+$/, '') + '-spacetime';
els.recordBtn.addEventListener('click', async () => {
  if (!state.recon) return;
  els.recordBtn.disabled = true;
  els.recDot.hidden = false;
  els.playBtn.textContent = '❚❚';
  setStatus('Recording one playback… keep this tab visible.');
  try {
    const blob = await viewer.recordPlayback();
    download(blob, `${baseName()}.webm`);
    setStatus(`Saved ${(blob.size / 1048576).toFixed(1)} MB WebM. Convert to MP4 with: ffmpeg -i in.webm -c:v libx264 -pix_fmt yuv420p out.mp4`);
  } catch (e) {
    setStatus(`Recording failed: ${e.message || e}`);
  } finally {
    els.recDot.hidden = true;
    els.recordBtn.disabled = false;
  }
});
els.snapBtn.addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = viewer.snapshot();
  a.download = `${baseName()}.png`;
  a.click();
});
els.exportPlyBtn.addEventListener('click', () => {
  if (!state.recon) return;
  download(new Blob([toPLY(state.recon)], { type: 'text/plain' }), `${baseName()}.ply`);
});

// Expose for automated tests.
window.__spacetime = { state, viewer };
