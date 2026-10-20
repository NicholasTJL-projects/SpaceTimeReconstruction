// Three.js viewer for a space + time reconstruction.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { NEAR } from './reconstruct.js';

const subjectVert = /* glsl */ `
  attribute float t;
  uniform float uTime;      // normalized playback time 0..1
  uniform float uTrail;     // trail length in normalized time (>=1 means everything)
  uniform float uMode;      // 0 grow, 1 all at once
  uniform float uSize;
  uniform float uScale;     // pixel scale for size attenuation
  uniform float uExtrude;   // offset along x per unit time
  uniform float uFrameDt;   // 1/(count-1)
  varying vec3 vColor;
  varying float vAlpha;
  varying float vHot;
  void main() {
    vColor = color;
    vec3 p = position;
    p.x += (t - 0.5) * uExtrude;
    float age = uTime - t;
    float vis = uMode > 0.5 ? 1.0 : step(-0.5 * uFrameDt, age);
    float within = uTrail >= 1.0 ? 1.0 : (1.0 - step(uTrail, age));
    vis *= within;
    float fade = uTrail >= 1.0 ? 1.0 : clamp(age / uTrail, 0.0, 1.0);
    vAlpha = vis * mix(1.0, 0.55, fade);
    vHot = 1.0 - smoothstep(0.0, 1.5 * uFrameDt, abs(age));
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_PointSize = uSize * uScale / max(0.1, -mv.z);
    gl_Position = projectionMatrix * mv;
  }
`;
const subjectFrag = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  varying float vHot;
  void main() {
    if (vAlpha <= 0.001) discard;
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    vec3 col = vColor * (1.0 + 0.25 * vHot);
    gl_FragColor = vec4(col, vAlpha);
  }
`;
const bgVert = /* glsl */ `
  uniform float uSize;
  uniform float uScale;
  uniform float uDim;
  varying vec3 vColor;
  void main() {
    vColor = color * uDim;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize * uScale / max(0.1, -mv.z);
    gl_Position = projectionMatrix * mv;
  }
`;
const bgFrag = /* glsl */ `
  varying vec3 vColor;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    gl_FragColor = vec4(vColor, 1.0);
  }
`;

export class Viewer {
  constructor(canvas, wrap) {
    this.canvas = canvas;
    this.wrap = wrap;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
    this.camera.position.set(0, 2, 6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.addEventListener('start', () => this.onUserInteract?.());

    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.recon = null;
    this.time = 0;
    this.playing = false;
    this.fps = 12;
    this.autoOrbit = true;
    this.orbitSpeed = 0.12;
    this.orbitAngle = 0;
    this.aspect = 'fill';
    this.onFrame = null;
    this._lastTs = performance.now();
    this._currentFrameIdx = -1;
    this.setSettings({});
    this.resize();
    window.addEventListener('resize', () => this.resize());
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  resize() {
    const rect = this.wrap.getBoundingClientRect();
    let w = Math.max(1, Math.floor(rect.width)), h = Math.max(1, Math.floor(rect.height));
    if (this.aspect !== 'fill') {
      const [a, b] = this.aspect.split('/').map(Number);
      const r = a / b;
      if (w / h > r) w = Math.floor(h * r); else h = Math.floor(w / r);
    }
    this.renderer.setSize(w, h, true);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const scale = (h * this.renderer.getPixelRatio()) / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    if (this.subjMat) this.subjMat.uniforms.uScale.value = scale * 0.01;
    if (this.bgMat) this.bgMat.uniforms.uScale.value = scale * 0.01;
  }

  setAspect(a) {
    this.aspect = a;
    this.wrap.classList.toggle('fixed-aspect', a !== 'fill');
    this.resize();
  }

  setSettings(s) {
    Object.assign(this, s);
    if (this.subjMat) {
      const u = this.subjMat.uniforms;
      if (s.subjSize != null) u.uSize.value = s.subjSize;
      if (s.trail != null) u.uTrail.value = s.trail;
      if (s.reveal != null) u.uMode.value = s.reveal === 'all' ? 1 : 0;
      if (s.extrude != null) u.uExtrude.value = s.extrude;
    }
    if (this.bgMat) {
      if (s.bgSize != null) this.bgMat.uniforms.uSize.value = s.bgSize;
      if (s.bgDim != null) this.bgMat.uniforms.uDim.value = s.bgDim;
    }
    if (this.frustum && s.showFrustum != null) this.frustum.visible = s.showFrustum;
  }

  load(recon, settings) {
    this.recon = recon;
    this.group.clear();
    this.subjMat?.dispose();
    this.bgMat?.dispose();

    // Still scene
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute('position', new THREE.BufferAttribute(recon.background.positions, 3));
    bgGeo.setAttribute('color', new THREE.BufferAttribute(recon.background.colors, 3));
    this.bgMat = new THREE.ShaderMaterial({ vertexShader: bgVert, fragmentShader: bgFrag, vertexColors: true,
      uniforms: { uSize: { value: 2.4 }, uScale: { value: 1 }, uDim: { value: 0.55 } } });
    this.group.add(new THREE.Points(bgGeo, this.bgMat));

    // Subjects over time
    const sGeo = new THREE.BufferGeometry();
    sGeo.setAttribute('position', new THREE.BufferAttribute(recon.subjects.positions, 3));
    sGeo.setAttribute('color', new THREE.BufferAttribute(recon.subjects.colors, 3));
    sGeo.setAttribute('t', new THREE.BufferAttribute(recon.subjects.t, 1));
    this.subjMat = new THREE.ShaderMaterial({ vertexShader: subjectVert, fragmentShader: subjectFrag, vertexColors: true,
      transparent: true, depthWrite: false,
      uniforms: { uTime: { value: 0 }, uTrail: { value: 1 }, uMode: { value: 0 }, uSize: { value: 2.2 }, uScale: { value: 1 },
        uExtrude: { value: 0 }, uFrameDt: { value: 1 / Math.max(1, recon.count - 1) } } });
    this.group.add(new THREE.Points(sGeo, this.subjMat));

    // Camera frustum with the live frame on its image plane
    this.frustum = new THREE.Group();
    const camDist = Math.hypot(...recon.center);
    const d = Math.min(3, Math.max(NEAR * 0.5, camDist * 0.35));
    const hw = (recon.width / 2 / recon.focal) * d, hh = (recon.height / 2 / recon.focal) * d;
    const c = [[-hw, hh, -d], [hw, hh, -d], [hw, -hh, -d], [-hw, -hh, -d]];
    const pts = [];
    for (let i = 0; i < 4; i++) { pts.push(0, 0, 0, ...c[i]); pts.push(...c[i], ...c[(i + 1) % 4]); }
    const lGeo = new THREE.BufferGeometry();
    lGeo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    this.frustum.add(new THREE.LineSegments(lGeo, new THREE.LineBasicMaterial({ color: 0xbfe3ff, transparent: true, opacity: 0.85 })));
    this.frameCanvas = document.createElement('canvas');
    this.frameCanvas.width = recon.width;
    this.frameCanvas.height = recon.height;
    this.frameTex = new THREE.CanvasTexture(this.frameCanvas);
    this.frameTex.colorSpace = THREE.SRGBColorSpace;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(hw * 2, hh * 2),
      new THREE.MeshBasicMaterial({ map: this.frameTex, side: THREE.DoubleSide, transparent: true, opacity: 0.92 }));
    plane.position.set(0, 0, -d);
    this.frustum.add(plane);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    this.frustum.add(dot);
    this.group.add(this.frustum);

    this.center = new THREE.Vector3(...recon.center);
    this.controls.target.copy(this.center);
    // Orbit far enough out to frame the camera frustum and the whole trail:
    // roughly the bounding sphere of {camera origin, trail} seen with a 50° lens.
    const sphere = this.center.length() * 0.5 + (recon.extent || 0);
    this.orbitRadius = Math.max(2.5, sphere * 2.2);
    this.orbitAngle = 0.35;
    this.time = 0;
    this._currentFrameIdx = -1;
    this.setSettings(settings);
    this.resize();
    this._showFrame(0);
  }

  get frameIndex() {
    if (!this.recon) return 0;
    return Math.round(this.time * (this.recon.count - 1));
  }
  setFrame(i) {
    if (!this.recon) return;
    this.time = this.recon.count > 1 ? i / (this.recon.count - 1) : 0;
    this._showFrame(i);
  }
  _showFrame(i) {
    if (!this.recon || i === this._currentFrameIdx) return;
    this._currentFrameIdx = i;
    const fs = this.recon.frameSet;
    const ctx = this.frameCanvas.getContext('2d');
    ctx.putImageData(new ImageData(fs.frames[i], fs.width, fs.height), 0, 0);
    this.frameTex.needsUpdate = true;
    this.onFrame?.(i, this.recon.count);
  }

  _loop(ts) {
    requestAnimationFrame(this._loop);
    const dt = Math.min(0.1, (ts - this._lastTs) / 1000);
    this._lastTs = ts;
    if (this.recon) {
      if (this.playing) {
        const n = this.recon.count - 1;
        this.time += (dt * this.fps) / Math.max(1, n);
        if (this.time >= 1) { this.time = 1; this.playing = false; this.onEnded?.(); }
      }
      this.subjMat.uniforms.uTime.value = this.time;
      this._showFrame(this.frameIndex);
      if (this.autoOrbit) {
        this.orbitAngle += dt * this.orbitSpeed * Math.PI * 2 * 0.5;
        const el = 0.38 + 0.25 * Math.sin(this.orbitAngle * 0.7);
        const r = this.orbitRadius;
        this.camera.position.set(
          this.center.x + r * Math.cos(el) * Math.sin(this.orbitAngle),
          this.center.y + r * Math.sin(el),
          this.center.z + r * Math.cos(el) * Math.cos(this.orbitAngle),
        );
        this.controls.target.copy(this.center);
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Record one full playback of the current view to a WebM blob. */
  recordPlayback({ fps = 30, holdMs = 1200 } = {}) {
    return new Promise((resolve, reject) => {
      if (!('MediaRecorder' in window)) return reject(new Error('MediaRecorder is not supported in this browser.'));
      // Render at 2x while recording so the exported clip is sharp even on 1x displays.
      const prevRatio = this.renderer.getPixelRatio();
      this.renderer.setPixelRatio(Math.max(2, prevRatio));
      this.resize();
      const restore = () => { this.renderer.setPixelRatio(prevRatio); this.resize(); };
      const stream = this.canvas.captureStream(fps);
      const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find((t) => MediaRecorder.isTypeSupported(t));
      const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 12_000_000 });
      const chunks = [];
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      rec.onstop = () => { restore(); resolve(new Blob(chunks, { type })); };
      rec.onerror = (e) => { restore(); reject(e.error || new Error('Recording failed')); };
      this.setFrame(0);
      this.playing = true;
      const prevEnded = this.onEnded;
      this.onEnded = () => { this.onEnded = prevEnded; setTimeout(() => rec.stop(), holdMs); prevEnded?.(); };
      rec.start(250);
    });
  }

  snapshot() {
    return this.canvas.toDataURL('image/png');
  }
}
