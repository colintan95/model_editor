import { useEffect, useRef, useState } from "react";
import {
  buildArrows,
  perspective,
  translation,
  rotationX,
  rotationY,
  multiply,
  BYTE_STRIDE,
  FLOATS_PER_VERTEX,
  AXIS_DIRS,
  GIZMO_TIP_DIST,
  GIZMO_SCALE,
  transform4,
  rayCubeNearT,
  invert,
  rayFromCursor,
  raySphere,
  rayCapsule,
  defaultMesh,
  buildMeshData,
  extrudeFace,
  faceCentroid,
  rayPoly,
} from "./geometry";
import type { Vec3, DynMesh } from "./geometry";

// ── WGSL shader ─────────────────────────────────────────────────────────────
// A vertex buffer supplies a position (vec3), a normal (vec3) and a color
// (vec4) per vertex. uMVP (projection * view * model) turns the local cube into
// screen space, and uModel (the model rotation) orients each normal into world
// space so a fixed directional light can shade it. The faces are all one flat
// grey; the lighting — not the colour — is what tells them apart.

const shader = /* wgsl */ `
@group(0) @binding(0) var<uniform> uMVP : mat4x4f;
@group(0) @binding(1) var<uniform> uModel : mat4x4f;

struct VSOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) worldNormal : vec3f,
};

// A fixed directional light in world space, plus a soft ambient floor so the
// faces turned away from it never go fully black.
const LIGHT_DIR = normalize(vec3f(0.4, 0.8, 0.6));
const AMBIENT = 0.35;

@vertex
fn vertexMain(
  @location(0) position : vec3f,
  @location(1) normal : vec3f,
  @location(2) color : vec4f,
) -> VSOut {
  var out : VSOut;
  out.position = uMVP * vec4f(position, 1.0);
  // model is a pure rotation, so its 3x3 part orients the normal without
  // stretching it; the translation columns are ignored.
  out.worldNormal = mat3x3f(uModel[0].xyz, uModel[1].xyz, uModel[2].xyz) * normal;
  out.color = color;
  return out;
}

@fragment
fn fragmentMain(in : VSOut) -> @location(0) vec4f {
  let diff = max(dot(normalize(in.worldNormal), LIGHT_DIR), 0.0);
  let intensity = AMBIENT + (1.0 - AMBIENT) * diff;
  return vec4f(in.color.rgb * intensity, in.color.a);
}
`;

// ── Gizmo shader (flat / unlit) ────────────────────────────────────────────
// The translation gizmo is a UI overlay, not part of the lit scene, so it is
// drawn flat: its X/Y/Z axis colours are output as-is, with no lighting and no
// normals. It shares the cube's xyz + nrm + rgba vertex layout but reads only
// the position and colour (the normal at location 1 is skipped).
const gizmoShader = /* wgsl */ `
@group(0) @binding(0) var<uniform> uMVP : mat4x4f;

struct VSOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
};

@vertex
fn vertexMain(
  @location(0) position : vec3f,
  @location(2) color : vec4f,
) -> VSOut {
  var out : VSOut;
  out.position = uMVP * vec4f(position, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fragmentMain(in : VSOut) -> @location(0) vec4f {
  return in.color;
}
`;

// ── Component ───────────────────────────────────────────────────────────────

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null); // 2D highlight layer
  const statusSpanRef = useRef<HTMLSpanElement>(null);

  const [status, setStatus] = useState<string>("Initialising WebGPU…");
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);

  // Editing mode: select + move a single vertex, or a whole face (its 4
  // corners together). Switching modes clears the other kind of selection.
  const [mode, setMode] = useState<"vertex" | "face">("vertex");
  // Face currently selected in face mode (null = none).
  const [faceSel, setFaceSel] = useState<number | null>(null);

  // Live, mutable rotation state shared between the drag handlers and the
  // render loop without forcing a re-render (or restarting the effect).
  const yawRef = useRef(0.6);
  const pitchRef = useRef(0.35);
  const draggingRef = useRef(false);
  const lastPointerRef = useRef({ x: 0, y: 0 });
  const downPointerRef = useRef({ x: 0, y: 0 });
  const didDragRef = useRef(false);
  const distanceRef = useRef(3); // camera distance — the wheel zooms this
  const panXRef = useRef(0); // camera pan in world units — shift+drag
  const panYRef = useRef(0);

  // Live dynamic mesh (verts + faces).  Starts as the unit cube; grows when a
  // face is extruded.  A ref so the render loop + gizmo see the latest value
  // without a re-render.
  const meshRef = useRef<DynMesh>(defaultMesh());
  // Set while vertices have moved or faces have been added, so the GPU buffer
  // gets rebuilt.
  const dirtyMeshRef = useRef(false);
  // Track whether the mesh structure (face/vert count) changed, requiring a
  // new GPU buffer allocation.
  const meshStructureDirtyRef = useRef(false);
  // Active gizmo drag: which corners (one for a vertex, four for a face) and
  // which local axis they are sliding along. `axis` is locked the moment the
  // gizmo is grabbed (ray-cast), so it never changes for the duration of the
  // drag.
  const gizmoDragRef = useRef<{
    // Corner indices the drag slides (one for a vertex, four for a face).
    // `axis` is locked the moment the gizmo is grabbed (ray-cast), so it never
    // changes for the duration of the drag.
    corners: number[];
    axis: [number, number, number];
  } | null>(null);

  // Which corner is currently selected (-1 / null = none). A ref so the render
  // loop + overlay reader always see the latest value without a re-render.
  const selectedRef = useRef<number | null>(null);

  // Mirror of `mode` so the drag handlers + render loop read the latest value
  // without a re-render.
  const modeRef = useRef<"vertex" | "face">("vertex");
  // Face currently selected (null = none) — the face-mode analogue of
  // selectedRef, read by the handlers/render loop without a re-render.
  const faceSelRef = useRef<number | null>(null);

  // The most recent camera transforms, captured each frame so that click
  // handlers (which fire between frames) and the overlay use consistent maths.
  const frameRef = useRef<{
    mvp: Float32Array;
    invModel: Float32Array;
    invViewProj: Float32Array | null;
    cam: [number, number, number];
  } | null>(null);

  // Mode switching (V / F) and deselect on Escape / Clear.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (e.key === "Escape") {
        selectedRef.current = null;
        setSelected(null);
        faceSelRef.current = null;
        setFaceSel(null);
        return;
      }
      if (k === "v") {
        modeRef.current = "vertex";
        setMode("vertex");
        faceSelRef.current = null;
        setFaceSel(null);
      } else if (k === "f") {
        modeRef.current = "face";
        setMode("face");
        selectedRef.current = null;
        setSelected(null);
      } else if (k === "e") {
        // Extrude the selected face outward along its normal.
        if (modeRef.current === "face" && faceSelRef.current !== null) {
          const capIdx = extrudeFace(meshRef.current, faceSelRef.current, 0.3);
          faceSelRef.current = capIdx;
          setFaceSel(capIdx);
          dirtyMeshRef.current = true;
          meshStructureDirtyRef.current = true;
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let frame = 0;

    async function init() {
      try {
        if (!navigator.gpu) {
          throw new Error("WebGPU is not available in this browser.");
        }

        // 1. Adapter & device
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No suitable GPU adapter found.");
        const device = await adapter.requestDevice();
        if (cancelled) return;

        device.lost
          .then((info) => {
            if (!cancelled) {
              setError(`GPU device lost: ${info.message || "unknown reason"}`);
            }
          })
          .catch(() => {
            /* ignore rejection if cleanup already ran */
          });

        // 2. Canvas + context
        const canvas = canvasRef.current;
        if (!canvas) return;
        // Captured non-null so the nested render closure stays narrowed.
        const el = canvas;
        const overlay = overlayRef.current;
        if (!overlay) return;
        const octx = overlay.getContext("2d");
        if (!octx) throw new Error("Could not obtain a 2D context.");

        // Depth-stencil texture must match the canvas; (re)created on resize.
        let depthTexture = device.createTexture({
          size: [Math.max(1, canvas.width), Math.max(1, canvas.height)],
          format: "depth24plus",
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        let depthView = depthTexture.createView();

        const resize = () => {
          const dpr = window.devicePixelRatio || 1;
          canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
          canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
          // The 2D overlay mirrors the canvas backing store so its pixels line
          // up exactly with what the GPU drew.
          overlay.width = canvas.width;
          overlay.height = canvas.height;
          depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
          });
          depthView = depthTexture.createView();
        };
        resize();
        window.addEventListener("resize", resize);

        const context = canvas.getContext("webgpu") as
          | GPUCanvasContext
          | null;
        if (!context) throw new Error("Could not obtain a WebGPU context.");
        const format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format });

        // 3. Vertex + uniform buffers
        // Start with a generous buffer so a few extrudes fit without realloc.
        const MESH_BUF_BYTES = 64 * 1024;
        let vertexBuffer = device.createBuffer({
          size: MESH_BUF_BYTES,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(vertexBuffer, 0, buildMeshData(meshRef.current));

        const uniformBuffer = device.createBuffer({
          size: 64, // one mat4x4f
          usage:
            GPUBufferUsage.UNIFORM |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        });
        // The model rotation, shared by the cube and the gizmo (the gizmo only
        // adds a translation, which doesn't affect normals). Used to orient the
        // per-vertex normals into world space for lighting.
        const modelUniform = device.createBuffer({
          size: 64, // one mat4x4f
          usage:
            GPUBufferUsage.UNIFORM |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        });

        // 4. Pipeline + bind group
        const shaderModule = device.createShaderModule({ code: shader });
        const bindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.VERTEX,
              buffer: { type: "uniform" },
            },
            {
              binding: 1,
              visibility: GPUShaderStage.VERTEX,
              buffer: { type: "uniform" },
            },
          ],
        });

        // The gizmo is unlit, so it uses its own (flat) shader module and a
        // bind group layout that only needs the MVP — no model/lighting uniform.
        const gizmoShaderModule = device.createShaderModule({
          code: gizmoShader,
        });
        const gizmoBindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.VERTEX,
              buffer: { type: "uniform" },
            },
          ],
        });

        const bindGroup = device.createBindGroup({
          layout: bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: modelUniform } },
          ],
        });

        // Vertex translation gizmo. The arrows reuse the cube's shader/pipeline
        // (same xyz + nrm + rgba layout), so we only need their own vertex
        // buffer plus a second uniform to carry the gizmo's MVP (the cube's MVP
        // post-multiplies a translation onto the selected corner). The lighting
        // model uniform is shared, since the gizmo only adds a translation.
        const arrowData = buildArrows();
        const arrowBuffer = device.createBuffer({
          size: arrowData.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(arrowBuffer, 0, arrowData);
        const arrowCount = arrowData.length / FLOATS_PER_VERTEX;

        const gizmoUniform = device.createBuffer({
          size: 64, // one mat4x4f
          usage:
            GPUBufferUsage.UNIFORM |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        });
        const gizmoBindGroup = device.createBindGroup({
          layout: gizmoBindGroupLayout,
          entries: [{ binding: 0, resource: { buffer: gizmoUniform } }],
        });

        const pipeline = device.createRenderPipeline({
          layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
          }),
          vertex: {
            module: shaderModule,
            entryPoint: "vertexMain",
            buffers: [
              {
                arrayStride: BYTE_STRIDE,
                attributes: [
                  { shaderLocation: 0, offset: 0, format: "float32x3" },
                  { shaderLocation: 1, offset: 12, format: "float32x3" },
                  { shaderLocation: 2, offset: 24, format: "float32x4" },
                ],
              },
            ],
          },
          fragment: {
            module: shaderModule,
            entryPoint: "fragmentMain",
            targets: [{ format }],
          },
          primitive: {
            topology: "triangle-list",
          },
          depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "less",
          },
        });

        // A second pipeline for the gizmo. It uses the flat (unlit) shader so
        // the arrows keep their solid X/Y/Z colours, and it disables depth
        // testing so they always render on top of the cube — visible even when
        // they intersect or pass behind the geometry. The vertex format skips
        // the normal (location 1) the unlit shader doesn't need.
        const gizmoPipeline = device.createRenderPipeline({
          layout: device.createPipelineLayout({
            bindGroupLayouts: [gizmoBindGroupLayout],
          }),
          vertex: {
            module: gizmoShaderModule,
            entryPoint: "vertexMain",
            buffers: [
              {
                arrayStride: BYTE_STRIDE,
                attributes: [
                  { shaderLocation: 0, offset: 0, format: "float32x3" },
                  { shaderLocation: 2, offset: 24, format: "float32x4" },
                ],
              },
            ],
          },
          fragment: {
            module: gizmoShaderModule,
            entryPoint: "fragmentMain",
            targets: [{ format }],
          },
          primitive: {
            topology: "triangle-list",
          },
          depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: false,
            depthCompare: "always",
          },
        });

        // 5. Camera control: plain drag rotates, shift+drag pans, wheel zooms.
        // (Pointer events cover mouse + touch + pen.)
        const FOV = (60 * Math.PI) / 180; // shared by the projection + pan maths

        // ── Selection helpers (vertex or face, by mode) ────────────────
        // Corner indices the *current* selection covers: one for a vertex, its
        // four for a face. null when nothing is selected in this mode.
        const selectedCornerIndices = (): number[] | null => {
          if (modeRef.current === "face") {
            const f = faceSelRef.current;
            return f === null ? null : meshRef.current.faces[f].corners;
          }
          const v = selectedRef.current;
          return v === null ? null : [v];
        };
        // Local-space centre of the current selection — the gizmo's anchor
        // (the corner itself for a vertex, the face's centre for a face).
        const selectedCentroid = (): Vec3 | null => {
          if (modeRef.current === "face") {
            const f = faceSelRef.current;
            return f === null ? null : faceCentroid(meshRef.current, f);
          }
          const v = selectedRef.current;
          return v === null ? null : meshRef.current.verts[v];
        };
        // Anything selected in the active mode?
        const hasSelection = (): boolean => selectedCornerIndices() !== null;

        // Pick the corner the cursor's ray passes closest to. The ray is built in
        // world space from the inverse of (proj·view), then rotated into the cube's
        // local space (model is rotation-only, so a plain rotation is exact). Each
        // corner is a small sphere; occluded corners (a cube face in front) are
        // skipped, and of the visible ones the nearest along the ray wins.
        const PICK_RADIUS_PX = 24;
        const pick = (clientX: number, clientY: number) => {
          const st = frameRef.current;
          if (!st || !st.invViewProj) return;
          const rect = el.getBoundingClientRect();
          const ray = rayFromCursor(
            st.invViewProj,
            clientX - rect.left,
            clientY - rect.top,
            rect.width,
            rect.height,
          );
          if (!ray) return;
          // Rotate the world ray into the cube's local space.
          const lo = transform4(st.invModel, ray.ox, ray.oy, ray.oz, 1);
          const ld = transform4(st.invModel, ray.dx, ray.dy, ray.dz, 0);
          const ldLen = Math.hypot(ld[0], ld[1], ld[2]) || 1;
          const ldx = ld[0] / ldLen;
          const ldy = ld[1] / ldLen;
          const ldz = ld[2] / ldLen;

          // ── Face mode: ray-test every face quad, pick the nearest hit. ──
          if (modeRef.current === "face") {
            const mesh = meshRef.current;
            let bestFace = -1;
            let bestT = Infinity;
            for (let fi = 0; fi < mesh.faces.length; fi++) {
              const f = mesh.faces[fi];
              // Build the polygon points in local space.
              const pts: Vec3[] = f.corners.map((ci) => mesh.verts[ci]);
              const t = rayPoly(lo[0], lo[1], lo[2], ldx, ldy, ldz, pts);
              if (t !== null && t < bestT) {
                bestT = t;
                bestFace = fi;
              }
            }
            faceSelRef.current = bestFace >= 0 ? bestFace : null;
            setFaceSel(faceSelRef.current);
            return;
          }

          // ── Vertex mode: the nearest corner the ray passes by, skipping any
          // hidden behind a face. ──
          // A constant *screen* grab target: convert the pixel radius to local
          // units (model has no scale, so local units == world units here).
          const worldPerPx =
            (2 * distanceRef.current * Math.tan(FOV / 2)) / (el.clientHeight || 1);
          const radius = PICK_RADIUS_PX * worldPerPx;
          // Where the ray first enters the cube: a face in front ⇒ that corner
          // is occluded.
          const tCube = rayCubeNearT(lo[0], lo[1], lo[2], ldx, ldy, ldz);
          let best = -1;
          let bestT = Infinity;
          for (let i = 0; i < meshRef.current.verts.length; i++) {
            const c = meshRef.current.verts[i];
            const t = raySphere(lo[0], lo[1], lo[2], ldx, ldy, ldz, c[0], c[1], c[2], radius);
            if (t === null) continue;
            if (tCube !== null && tCube < t - 1e-2) continue; // a face is in front
            if (t < bestT) {
              bestT = t;
              best = i;
            }
          }
          selectedRef.current = best >= 0 ? best : null;
          setSelected(selectedRef.current);
        };

        // Screen-space pixels-per-unit of sliding the corner along a local axis,
        // found by projecting the corner and a one-unit step ahead through the
        // current MVP. Turns a 2D pointer drag into a 3D move along that axis.
        const screenAxis = (
          st: NonNullable<typeof frameRef.current>,
          corner: readonly number[],
          axis: readonly number[],
        ): [number, number] => {
          const rect = el.getBoundingClientRect();
          const a = transform4(st.mvp, corner[0], corner[1], corner[2], 1);
          const b = transform4(
            st.mvp,
            corner[0] + axis[0],
            corner[1] + axis[1],
            corner[2] + axis[2],
            1,
          );
          if (a[3] <= 0 || b[3] <= 0) return [0, 0];
          const ax = (a[0] / a[3] * 0.5 + 0.5) * rect.width;
          const ay = (1 - (a[1] / a[3] * 0.5 + 0.5)) * rect.height;
          const bx = (b[0] / b[3] * 0.5 + 0.5) * rect.width;
          const by = (1 - (b[1] / b[3] * 0.5 + 0.5)) * rect.height;
          return [bx - ax, by - ay];
        };

        // Is the press on the selected vertex's gizmo? Cast the cursor ray into
        // the cube's local space and test it against the three arrow capsules; a
        // hit means "grab the gizmo" rather than "rotate the cube", and the
        // capsule struck tells us *which axis* (so the drag can lock onto it
        // immediately). Each arrow is a capsule — a cylindrical shaft plus two
        // rounded ends — so the grab is exact and orientation-independent.
        // Returns the AXIS_DIRS index (0..2), or -1 when nothing is under the
        // cursor.
        // The invisible grab tube grows with the (now scaled-up) gizmo so it
        // stays as easy to grab as the original; 0.07 was the base target.
        const GRAB_RADIUS = 0.07 * GIZMO_SCALE; // local/world units
        const grabGizmo = (clientX: number, clientY: number): number => {
          const st = frameRef.current;
          if (!st || !st.invViewProj || !hasSelection()) return -1;
          const rect = el.getBoundingClientRect();
          const ray = rayFromCursor(
            st.invViewProj,
            clientX - rect.left,
            clientY - rect.top,
            rect.width,
            rect.height,
          );
          if (!ray) return -1;
          // Rotate the world ray into local space (model is a pure rotation).
          const lo = transform4(st.invModel, ray.ox, ray.oy, ray.oz, 1);
          const ld = transform4(st.invModel, ray.dx, ray.dy, ray.dz, 0);
          const ldLen = Math.hypot(ld[0], ld[1], ld[2]) || 1;
          const ldx = ld[0] / ldLen;
          const ldy = ld[1] / ldLen;
          const ldz = ld[2] / ldLen;
          // Anchor the three arrows on the selection's centre (the corner for a
          // vertex, the face's centre for a face) so the same grab works for
          // both modes.
          const c = selectedCentroid();
          if (!c) return -1;
          let bestAxis = -1;
          let bestT = Infinity;
          for (let a = 0; a < AXIS_DIRS.length; a++) {
            const d = AXIS_DIRS[a];
            const bx = c[0] + d[0] * GIZMO_TIP_DIST;
            const by = c[1] + d[1] * GIZMO_TIP_DIST;
            const bz = c[2] + d[2] * GIZMO_TIP_DIST;
            const t = rayCapsule(
              lo[0], lo[1], lo[2], ldx, ldy, ldz,
              c[0], c[1], c[2], bx, by, bz, GRAB_RADIUS,
            );
            if (t === null) continue;
            // Only the arrows are hit-tested — an arrow that overlaps the cube
            // body is still grabbable, so we take the nearest capsule along the
            // ray regardless of whether it sits inside the cube.
            if (t < bestT) {
              bestT = t;
              bestAxis = a;
            }
          }
          return bestAxis;
        };

        const onPointerDown = (e: PointerEvent) => {
          draggingRef.current = true;
          didDragRef.current = false;
          lastPointerRef.current = { x: e.clientX, y: e.clientY };
          downPointerRef.current = { x: e.clientX, y: e.clientY };
          // A press on the selected vertex's gizmo starts a vertex move instead
          // of a camera rotation. grabGizmo returns the axis the cursor struck,
          // which is locked for the whole drag.
          const grabAxis = grabGizmo(e.clientX, e.clientY);
          if (grabAxis >= 0) {
            const corners = selectedCornerIndices();
            if (corners) {
              gizmoDragRef.current = {
                corners,
                axis: AXIS_DIRS[grabAxis],
              };
            }
          }
          canvas.setPointerCapture(e.pointerId);
        };
        const onPointerMove = (e: PointerEvent) => {
          if (!draggingRef.current) return;
          const st = frameRef.current;
          const adx = e.clientX - downPointerRef.current.x;
          const ady = e.clientY - downPointerRef.current.y;
          // Any real drag (gizmo move or camera turn) is only a drag, not a
          // click, once the pointer travels far enough.
          if (Math.abs(adx) > 3 || Math.abs(ady) > 3) didDragRef.current = true;

          // ── Vertex gizmo drag ──
          // When the press grabbed the gizmo, slide the selected corner instead
          // of turning the camera. The motion axis was locked when the gizmo was
          // grabbed (ray-cast), so just project the per-frame screen step onto it
          // to get the local displacement along it.
          const g = gizmoDragRef.current;
          if (g && st) {
            const dx = e.clientX - lastPointerRef.current.x;
            const dy = e.clientY - lastPointerRef.current.y;
            // Project the locked local axis to screen from the *current* centre
            // of the moving corners, so the drag tracks the handle as it moves.
            // Centroid of the moving corners in the mesh.
            let cx = 0, cy = 0, cz = 0;
            for (const ci of g.corners) {
              const c = meshRef.current.verts[ci];
              cx += c[0]; cy += c[1]; cz += c[2];
            }
            const nn = g.corners.length;
            const s = screenAxis(
              st,
              [cx / nn, cy / nn, cz / nn],
              g.axis,
            );
            const len2 = s[0] * s[0] + s[1] * s[1];
            if (len2 > 1e-6) {
              const t = (dx * s[0] + dy * s[1]) / len2;
              // Slide every corner of the handle along the locked axis, clamped
              // to the same box a single vertex respects.
              for (const ci of g.corners) {
                const c = meshRef.current.verts[ci];
                c[0] = Math.max(-1.5, Math.min(1.5, c[0] + g.axis[0] * t));
                c[1] = Math.max(-1.5, Math.min(1.5, c[1] + g.axis[1] * t));
                c[2] = Math.max(-1.5, Math.min(1.5, c[2] + g.axis[2] * t));
              }
              dirtyMeshRef.current = true;
            }
            lastPointerRef.current = { x: e.clientX, y: e.clientY };
            return;
          }

          // ── Camera drag ──
          const dx = e.clientX - lastPointerRef.current.x;
          const dy = e.clientY - lastPointerRef.current.y;
          lastPointerRef.current = { x: e.clientX, y: e.clientY };
          if (e.shiftKey) {
            // Shift+drag pans the camera in the screen plane. Convert the pixel
            // delta to world units via the visible height at the current zoom.
            const worldPerPx =
              (2 * distanceRef.current * Math.tan(FOV / 2)) / (el.clientHeight || 1);
            panXRef.current -= dx * worldPerPx;
            panYRef.current += dy * worldPerPx;
          } else {
            // Plain drag rotates the cube.
            yawRef.current += dx * 0.01;
            pitchRef.current += dy * 0.01;
          }
        };
        const endDrag = (e: PointerEvent) => {
          draggingRef.current = false;
          // A finished gizmo drag must not re-pick, which could deselect the
          // vertex we just moved.
          const wasGizmo = gizmoDragRef.current !== null;
          gizmoDragRef.current = null;
          if (!wasGizmo && !didDragRef.current) {
            // A tap with no movement selects (or clears) the nearest vertex.
            pick(e.clientX, e.clientY);
          }
          try {
            canvas.releasePointerCapture(e.pointerId);
          } catch {
            /* pointer already released */
          }
        };

        // Wheel zooms the camera in/out. A multiplicative step feels natural;
        // clamp so the cube can't cross the near plane or fly past the far one.
        // passive:false is required so preventDefault() can stop page scrolling.
        const MIN_DISTANCE = 1.2;
        const MAX_DISTANCE = 12;
        const onWheel = (e: WheelEvent) => {
          e.preventDefault();
          const next = distanceRef.current * (1 + e.deltaY * 0.001);
          distanceRef.current = Math.min(MAX_DISTANCE, Math.max(MIN_DISTANCE, next));
        };

        canvas.addEventListener("pointerdown", onPointerDown);
        canvas.addEventListener("pointermove", onPointerMove);
        canvas.addEventListener("pointerup", endDrag);
        canvas.addEventListener("pointercancel", endDrag);
        canvas.addEventListener("wheel", onWheel, { passive: false });

        // 6. Render loop
        // ── Occlusion test ──
        // Is the selected corner hidden *behind a face* of the cube? We test in
        // the cube's local space, where the cube is always the axis-aligned box
        // [-0.5,0.5]³. The corner we care about sits at t = 1 on the ray from
        // the (local) camera through it; if that ray enters the cube earlier
        // (t < 1) some face is in front, so the corner is occluded.
        const EPS = 1e-3;
        function cornerHidden(
          st: NonNullable<typeof frameRef.current>,
          corner: readonly [number, number, number],
        ): boolean {
          // Local-space camera = inverse(model) * cameraWorld. model is a pure
          // rotation (yaw∘pitch), so its inverse just negates the angles.
          const camL = transform4(
            st.invModel,
            st.cam[0],
            st.cam[1],
            st.cam[2],
            1,
          );
          const t = rayCubeNearT(
            camL[0],
            camL[1],
            camL[2],
            corner[0] - camL[0],
            corner[1] - camL[1],
            corner[2] - camL[2],
          );
          // t < 1 - EPS  ⇒ the cube's surface is reached before the corner.
          return t !== null && t < 1 - EPS;
        }

        // ── 2D highlight overlay ──
        // Each frame: clear, then — if a vertex is selected and currently
        // visible (in front of the camera and not behind a face) — paint a
        // marker at its projected screen position.
        // Project a local point to CSS-pixel screen coords (non-null overlay).
        const project = (
          st: NonNullable<typeof frameRef.current>,
          c: readonly number[],
        ): [number, number, boolean] => {
          const clip = transform4(st.mvp, c[0], c[1], c[2], 1);
          const inFront = clip[3] > 0;
          const px = (clip[0] / clip[3] * 0.5 + 0.5) * overlay!.width;
          const py = (1 - (clip[1] / clip[3] * 0.5 + 0.5)) * overlay!.height;
          return [px, py, inFront];
        };

        function drawOverlay() {
          octx!.clearRect(0, 0, overlay!.width, overlay!.height);
          const st = frameRef.current;
          if (!st) {
            if (statusSpanRef.current)
              statusSpanRef.current.textContent =
                modeRef.current === "face"
                  ? "no face selected"
                  : "no vertex selected";
            return;
          }

          // ── Face mode: outline the selected face's quad. ──
          if (modeRef.current === "face") {
            const f = faceSelRef.current;
            if (f === null) {
              if (statusSpanRef.current)
                statusSpanRef.current.textContent = "no face selected";
              return;
            }
            const dpr = window.devicePixelRatio || 1;
            const mesh = meshRef.current;
            const pts = mesh.faces[f].corners.map((ci) =>
              project(st, mesh.verts[ci]),
            );
            const allFront = pts.every((p) => p[2]);
            if (statusSpanRef.current)
              statusSpanRef.current.textContent = allFront
                ? `face ${f} · visible`
                : `face ${f} · back-facing (no highlight)`;
            if (!allFront) return;
            octx!.beginPath();
            pts.forEach((p, i) =>
              i === 0 ? octx!.moveTo(p[0], p[1]) : octx!.lineTo(p[0], p[1]),
            );
            octx!.closePath();
            octx!.fillStyle = "rgba(80,200,255,0.25)";
            octx!.fill();
            octx!.strokeStyle = "rgba(150,235,255,0.95)";
            octx!.lineWidth = 1.5 * dpr;
            octx!.stroke();
            return;
          }

          // ── Vertex mode: mark every vertex, highlight the selected one. ──
          const dpr = window.devicePixelRatio || 1;
          const mesh = meshRef.current;
          const sel = selectedRef.current;

          // Pass 1 — every visible, non-selected vertex gets a small filled dot
          // (drawn first so the selection paints on top). Vertices behind a face
          // or behind the camera are skipped.
          for (let i = 0; i < mesh.verts.length; i++) {
            if (i === sel) continue;
            const v = mesh.verts[i];
            const clip = transform4(st.mvp, v[0], v[1], v[2], 1);
            if (clip[3] <= 0) continue; // behind the camera
            if (cornerHidden(st, v)) continue; // behind a face
            const px = (clip[0] / clip[3] * 0.5 + 0.5) * overlay!.width;
            const py = (1 - (clip[1] / clip[3] * 0.5 + 0.5)) * overlay!.height;
            octx!.beginPath();
            octx!.arc(px, py, 3 * dpr, 0, Math.PI * 2);
            octx!.fillStyle = "rgba(255,255,255,0.9)";
            octx!.fill();
          }

          // Pass 2 — the selected vertex on top, as a ringed orange core.
          if (sel !== null) {
            const corner = mesh.verts[sel];
            const clip = transform4(st.mvp, corner[0], corner[1], corner[2], 1);
            const inFront = clip[3] > 0;
            const hidden = inFront ? cornerHidden(st, corner) : true;
            const visible = inFront && !hidden;

            if (statusSpanRef.current)
              statusSpanRef.current.textContent = visible
                ? `vertex ${sel} · visible`
                : `vertex ${sel} · hidden (no highlight)`;

            if (visible) {
              const px = (clip[0] / clip[3] * 0.5 + 0.5) * overlay!.width;
              const py = (1 - (clip[1] / clip[3] * 0.5 + 0.5)) * overlay!.height;
              octx!.beginPath();
              octx!.arc(px, py, 6 * dpr, 0, Math.PI * 2);
              octx!.strokeStyle = "rgba(255,255,255,0.95)";
              octx!.lineWidth = 1.5 * dpr;
              octx!.stroke();
              octx!.beginPath();
              octx!.arc(px, py, 3.5 * dpr, 0, Math.PI * 2);
              octx!.fillStyle = "rgba(255,165,0,1)";
              octx!.fill();
            }
          } else if (statusSpanRef.current) {
            statusSpanRef.current.textContent = "no vertex selected";
          }
        }

        function render() {
          if (cancelled) return;

          const aspect = el.width / el.height;
          const proj = perspective(FOV, aspect, 0.1, 100);
          const view = translation(
            -panXRef.current,
            -panYRef.current,
            -distanceRef.current,
          );
          const model = multiply(
            rotationY(yawRef.current),
            rotationX(pitchRef.current),
          );
          // proj·view once, so it can feed both the MVP and the picking ray's
          // inverse.
          const viewProj = multiply(proj, view);
          const mvp = multiply(viewProj, model);
          device.queue.writeBuffer(uniformBuffer, 0, mvp);
          // Feed the model rotation to the shared uniform so the shader can
          // orient normals for lighting (used by both the cube and the gizmo).
          device.queue.writeBuffer(modelUniform, 0, model);

          // Cache the transforms for the pick/grab handlers + occlusion test.
          // invViewProj turns a screen pixel into a world-space picking ray.
          frameRef.current = {
            mvp,
            invModel: multiply(rotationX(-pitchRef.current), rotationY(-yawRef.current)),
            invViewProj: invert(viewProj),
            cam: [panXRef.current, panYRef.current, distanceRef.current],
          };

          // Rebuild the mesh vertex buffer when vertices moved or faces were
          // added (extrude).  If the structure changed, the new data may be
          // larger than the current buffer, so reallocate if needed.
          if (dirtyMeshRef.current || meshStructureDirtyRef.current) {
            const data = buildMeshData(meshRef.current);
            if (meshStructureDirtyRef.current) {
              if (data.byteLength > vertexBuffer.size) {
                vertexBuffer.destroy();
                vertexBuffer = device.createBuffer({
                  size: data.byteLength,
                  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                });
              }
              meshStructureDirtyRef.current = false;
            }
            device.queue.writeBuffer(vertexBuffer, 0, data);
            dirtyMeshRef.current = false;
          }

          const encoder = device.createCommandEncoder();
          const pass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: context!.getCurrentTexture().createView(),
                clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1.0 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
            depthStencilAttachment: {
              view: depthView,
              depthClearValue: 1.0,
              depthLoadOp: "clear",
              depthStoreOp: "store",
            },
          });
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.setVertexBuffer(0, vertexBuffer);
          // Total vertices = Σ (corners−1)×3 over all faces (triangle fan).
          let drawCount = 0;
          for (const f of meshRef.current.faces) drawCount += (f.corners.length - 1) * 3;
          pass.draw(drawCount);

          // Translation gizmo: when a vertex or face is selected, draw the three
          // axis arrows anchored on the selection's centre. The arrows live in
          // the cube's local space, so post-multiplying the cube's MVP by a
          // translation to that centre places them on the handle and lets them
          // rotate with the cube.
          if (hasSelection()) {
            const c = selectedCentroid()!;
            const gizmoMVP = multiply(mvp, translation(c[0], c[1], c[2]));
            device.queue.writeBuffer(gizmoUniform, 0, gizmoMVP);
            pass.setPipeline(gizmoPipeline);
            pass.setBindGroup(0, gizmoBindGroup);
            pass.setVertexBuffer(0, arrowBuffer);
            pass.draw(arrowCount);
          }

          pass.end();

          device.queue.submit([encoder.finish()]);
          drawOverlay();
          frame = requestAnimationFrame(render);
        }

        setStatus(
          "WebGPU running — drag to rotate · shift+drag to pan · scroll to zoom · click to select a vertex or face (V / F) · E to extrude a face · drag a gizmo arrow to move it ✦",
        );
        frame = requestAnimationFrame(render);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof Error
            ? err.message
            : "WebGPU not supported in this browser. Try Chrome/Edge 113+ or Safari 18+.",
        );
      }
    }

    init();
    return () => {
      cancelled = true;
      if (frame) cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-950 text-white">
      <h1 className="mb-4 text-2xl font-bold tracking-tight">
        WebGPU Cube
      </h1>

      <div className="w-full max-w-2xl aspect-video rounded-xl overflow-hidden shadow-2xl ring-1 ring-white/10 relative">
        <canvas
          ref={canvasRef}
          className="w-full h-full block bg-gray-900 cursor-grab active:cursor-grabbing select-none"
          style={{ touchAction: "none" }}
        />
        {/* 2D highlight overlay — transparent, sits above the GPU canvas and
            never intercepts pointer events (so drags/clicks reach the canvas). */}
        <canvas
          ref={overlayRef}
          className="absolute inset-0 w-full h-full block pointer-events-none"
          style={{ touchAction: "none" }}
        />
      </div>

      {/* Mode toggle: choose what click-selection targets. */}
      <div className="mt-4 flex items-center gap-1 text-sm">
        <span className="text-gray-500 mr-1">mode</span>
        <button
          type="button"
          onClick={() => {
            modeRef.current = "vertex";
            setMode("vertex");
            faceSelRef.current = null;
            setFaceSel(null);
          }}
          className={
            "rounded-md border px-3 py-1 text-xs transition " +
            (mode === "vertex"
              ? "border-sky-400 bg-sky-500/20 text-sky-300"
              : "border-white/15 text-gray-300 hover:bg-white/10")
          }
        >
          Vertex
        </button>
        <button
          type="button"
          onClick={() => {
            modeRef.current = "face";
            setMode("face");
            selectedRef.current = null;
            setSelected(null);
          }}
          className={
            "rounded-md border px-3 py-1 text-xs transition " +
            (mode === "face"
              ? "border-sky-400 bg-sky-500/20 text-sky-300"
              : "border-white/15 text-gray-300 hover:bg-white/10")
          }
        >
          Face
        </button>
      </div>

      <div className="mt-3 flex items-center gap-3 text-sm text-gray-400 max-w-md text-center">
        <span className="inline-flex items-center gap-1.5">
          <span className="text-sky-400 font-semibold tabular-nums capitalize">
            {mode}
          </span>
          <span
            ref={statusSpanRef}
            className="text-gray-300 tabular-nums"
          >
            {mode === "face" ? "no face selected" : "no vertex selected"}
          </span>
        </span>
        <button
          type="button"
          onClick={() => {
            selectedRef.current = null;
            setSelected(null);
            faceSelRef.current = null;
            setFaceSel(null);
          }}
          disabled={selected === null && faceSel === null}
          className="rounded-md border border-white/15 px-2 py-0.5 text-xs text-gray-300 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Clear
        </button>
      </div>

      {error ? (
        <p className="mt-3 text-red-400 text-sm max-w-md text-center">
          {error}
        </p>
      ) : (
        <p className="mt-2 text-gray-500 text-xs">{status}</p>
      )}
    </div>
  );
}
