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
| Still scene | Per-pixel temporal median of up to 21 frames, which removes anything that moves |
| Scene depth | Depth Anything V2 small via Transformers.js (WebGPU or WASM), or a ground-plane heuristic that needs no download |
| Moving subjects | Absolute difference against the median background, threshold, morphological open and dilate, 4-connected components with a minimum area |
| Subject depth | Ground contact: each blob stands on the background depth at its lowest pixel, so the silhouette is placed where its feet touch the ground. Optional per-frame AI depth gives per-pixel relief at the cost of one inference per frame |
| Unprojection | Pinhole camera with the chosen horizontal field of view; depth range 1 to 12 scene units |
| Rendering | Three.js point clouds. Subject points carry a per-vertex time; a shader reveals them as playback advances, keeps a configurable trail, and brightens the current moment |
| Export | WebM via MediaRecorder, PNG snapshot, ASCII PLY with a `time` property per vertex |

## Good footage

- A camera that does not move: tripod, trail camera, dashcam at rest, CCTV. The reconstruction assumes a fixed viewpoint. A moving camera smears the background and the trail will be wrong.
- One or a few subjects that cross the frame rather than loiter in one spot, so the median background does not include them.
- Even lighting. Auto-exposure jumps and moving shadows register as motion. Raise the motion threshold if the mask is noisy, lower it if the subject is missing.

## Settings

- **Frames to sample**: more frames give a denser trail and a longer reconstruction. 72 is a good start.
- **Working width**: processing resolution. 320 is fast; 480 gives finer silhouettes.
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

## Converting the recording

The browser saves WebM. For an MP4:

```bash
ffmpeg -i scene-spacetime.webm -c:v libx264 -pix_fmt yuv420p -crf 18 scene-spacetime.mp4
```

## Limits

- Monocular depth is relative, not metric. Scale and perspective are plausible rather than measured.
- Fixed camera only. Camera motion would need structure-from-motion, which is outside the scope of this in-browser pipeline.
- Subjects are segmented by motion, so anything else that moves (branches, water, shadows) becomes part of the trail.
