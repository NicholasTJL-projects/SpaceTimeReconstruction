# Space + Time Reconstruction

Turn a video into a 4D scene: the still background is reconstructed once as a point cloud, every frame of the moving subject is reconstructed and stacked at the place it was, and a camera orbits the result so you see every moment at once. The source frame plays on the image plane of a camera frustum, the way it does in the "space + time" clips that inspired this.

Everything runs in the browser. No server, no upload, no build step.

## Try it

Serve the folder with any static server and open it:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

Or open the GitHub Pages deployment once the `Deploy to GitHub Pages` workflow has run (enable Pages with source "GitHub Actions" in the repository settings).

1. Drop in a video, or click **Use a synthetic demo clip**.
2. Pick settings. The defaults work for a fixed camera and a subject that crosses the frame.
3. **Reconstruct in 4D**. The first AI-depth run downloads the Depth Anything V2 small model (about 25 MB on WASM, 100 MB on WebGPU) and caches it.
4. Orbit with the mouse, scrub the timeline, then **Record orbit video** to save a WebM of one full playback. Choose the 9:16 aspect for a short-form clip.

## What it does

| Stage | Method |
| --- | --- |
| Frame sampling | N frames spread evenly across the clip, scaled to the working width |
| Camera stabilization (optional) | Every frame is aligned to the first with a 2D similarity transform (scale, rotation, translation) found by coarse-to-fine inverse-compositional Lucas-Kanade with a Huber weight, then resampled into the first frame's coordinates. Pixels that leave the frame are marked invalid and ignored downstream |
| Still scene | Per-pixel temporal median of up to 21 frames, which removes anything that moves |
| Scene depth | Depth Anything V2 small via Transformers.js (WebGPU or WASM), or a ground-plane heuristic that needs no download |
| Moving subjects | Absolute difference against the median background, threshold, morphological open and dilate, 4-connected components with a minimum area |
| Subject depth | Ground contact: each blob stands on the background depth at its lowest pixel, so the silhouette is placed where its feet touch the ground. Optional per-frame AI depth gives per-pixel relief at the cost of one inference per frame |
| Unprojection | Pinhole camera with the chosen horizontal field of view; depth range 1 to 12 scene units |
| Rendering | Three.js point clouds. Subject points carry a per-vertex time; a shader reveals them as playback advances, keeps a configurable trail, and brightens the current moment |
| Export | WebM via MediaRecorder, PNG snapshot, ASCII PLY with a `time` property per vertex |

## Good footage

- A camera that does not move: tripod, trail camera, dashcam at rest, CCTV. The reconstruction assumes a fixed viewpoint. For handheld or panning shots, enable **Stabilize camera**: it compensates pans, small drifts and zooms, but not a camera that travels through the scene.
- One or a few subjects that cross the frame rather than loiter in one spot, so the median background does not include them.
- Even lighting. Auto-exposure jumps and moving shadows register as motion. Raise the motion threshold if the mask is noisy, lower it if the subject is missing.

## Settings

- **Frames to sample**: more frames give a denser trail and a longer reconstruction. 72 is a good start.
- **Working width**: processing resolution. 320 is fast; 480 gives finer silhouettes.
- **Stabilize camera**: align all frames to the first before everything else. Use it when the status line reports that most of the frame counts as moving.
- **Depth source**: AI depth gives real relief in the still scene. The ground-plane heuristic treats everything below the horizon slider as flat ground and everything above it as a distant backdrop, which already works well for lawns, roads and floors.
- **Time extrusion**: offsets each moment sideways in proportion to its time, for subjects that move little in space.
- **Reveal**: grow the trail with playback, or show all moments at once.

## Files

- `index.html`, `style.css`: the page.
- `src/frames.js`: video sampling and the synthetic demo clip.
- `src/depth.js`: AI and heuristic depth providers.
- `src/reconstruct.js`: background, motion mask, components, unprojection, PLY export.
- `src/viewer.js`: Three.js scene, shaders, orbit, recorder.
- `src/main.js`: UI wiring.

Dependencies load from CDNs through an import map: Three.js 0.170 and Transformers.js 3.3.3. The model comes from the Hugging Face hub.

## Example footage

Two free clips from Mixkit (Mixkit License, free for commercial and personal use) were used to test the pipeline on real drifting:

- [Red toy car spinning around a race track](https://mixkit.co/free-stock-video/red-toy-car-spinning-around-a-race-track-45166/): fixed ground-level camera. Defaults work; subject depth "standing on the ground".
- [A drifting car in the countryside](https://mixkit.co/free-stock-video/a-drifting-car-in-the-countryside-46422/): static top-down drone. Use subject depth "on the background surface", motion threshold about 70, min blob about 80 to drop most of the dust, and a depth range of 3 so the flat ground is not stretched into a ramp.

Footage that follows the car with a wide pan and tilt (most handheld drift videos) changes the viewpoint too much for a single reconstruction; stabilization handles pans of up to a frame width or two, not a camera that tracks the subject across a large scene.

## Testing

- Click **Use a synthetic demo clip** and reconstruct. The ground-plane mode finishes in about a second; AI depth adds the one-time model download.
- Upload any fixed-camera clip. The status line reports the decoded size and the effective sampling rate before you run.
- Headless Chromium builds without proprietary codecs cannot decode H.264. If an .mp4 fails to load there, transcode it to VP9 WebM: `ffmpeg -i in.mp4 -c:v libvpx-vp9 -b:v 2M out.webm`. Chrome, Edge and Safari decode H.264 directly.

## Converting the recording

The browser saves WebM. For an MP4:

```bash
ffmpeg -i scene-spacetime.webm -r 30 -c:v libx264 -pix_fmt yuv420p -crf 18 scene-spacetime.mp4
```

## Limits

- Monocular depth is relative, not metric. Scale and perspective are plausible rather than measured.
- Fixed or stabilizable camera only. Stabilization handles pans, tilts, roll and zoom as a 2D similarity; a camera that moves through the scene (parallax) would need structure-from-motion, which is outside the scope of this in-browser pipeline.
- Subjects are segmented by motion, so anything else that moves (branches, water, shadows) becomes part of the trail.
