import { useEffect, useRef, useState } from "react";
import {
  buildCubeData,
  buildCubeDataFromCorners,
  defaultCorners,
  buildArrows,
  perspective,
  translation,
  rotationX,
  rotationY,
  multiply,
  BYTE_STRIDE,
  FLOATS_PER_VERTEX,
  CORNERS,
  AXIS_DIRS,
  GIZMO_TIP_DIST,
  transform4,
  rayCubeNearT,
} from "./geometry";

// ── WGSL shader ─────────────────────────────────────────────────────────────
// A vertex buffer supplies a position (vec3) and a color (vec4) per vertex.
// A single mat4 uniform (uMVP = projection * view * model) turns the local
// cube into screen space.

const shader = /* wgsl */ `
@group(0) @binding(0) var<uniform> uMVP : mat4x4f;

struct VSOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
};

@vertex
fn vertexMain(
  @location(0) position : vec3f,
  @location(1) color : vec4f,
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

  // Live positions of the 8 cube corners (an independent copy of CORNERS). A
  // gizmo drag mutates these so the dragged vertex reshapes its faces; a ref so
  // the render loop sees the latest value without a re-render.
  const cornersRef = useRef(defaultCorners());
  // Set while a corner has moved, so the cube's GPU buffer gets rebuilt.
  const dirtyCubeRef = useRef(false);
  // Active gizmo drag: which corner, and which local axis it is sliding along.
  // `axis` stays null until the first significant movement (so the axis can be
  // chosen from the drag direction) and then locks for the rest of the drag.
  const gizmoDragRef = useRef<{
    idx: number;
    axis: [number, number, number] | null;
  } | null>(null);

  // Which corner is currently selected (-1 / null = none). A ref so the render
  // loop + overlay reader always see the latest value without a re-render.
  const selectedRef = useRef<number | null>(null);

  // The most recent camera transforms, captured each frame so that click
  // handlers (which fire between frames) and the overlay use consistent maths.
  const frameRef = useRef<{
    mvp: Float32Array;
    invModel: Float32Array;
    cam: [number, number, number];
  } | null>(null);

  // Deselect on Escape / click a Clear button.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        selectedRef.current = null;
        setSelected(null);
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
        const cubeData = buildCubeData();
        const vertexBuffer = device.createBuffer({
          size: cubeData.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(vertexBuffer, 0, cubeData);

        const uniformBuffer = device.createBuffer({
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
          ],
        });
        const bindGroup = device.createBindGroup({
          layout: bindGroupLayout,
          entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
        });

        // Vertex translation gizmo. The arrows reuse the cube's shader/pipeline
        // (same xyz + rgba layout), so we only need their own vertex buffer plus
        // a second uniform to carry the gizmo's MVP (the cube's MVP post-multiply
        // a translation onto the selected corner).
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
          layout: bindGroupLayout,
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
                  { shaderLocation: 1, offset: 12, format: "float32x4" },
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

        // 5. Camera control: plain drag rotates, shift+drag pans, wheel zooms.
        // (Pointer events cover mouse + touch + pen.)
        const FOV = (60 * Math.PI) / 180; // shared by the projection + pan maths

        // Pick the corner nearest the cursor (in CSS pixels), within a threshold.
        // Uses the transforms captured on the previous frame (frameRef).
        const PICK_RADIUS = 24;
        const pick = (clientX: number, clientY: number) => {
          const st = frameRef.current;
          if (!st) return;
          const rect = el.getBoundingClientRect();
          const cx = clientX - rect.left;
          const cy = clientY - rect.top;
          let best = -1;
          let bestD = PICK_RADIUS;
          for (let i = 0; i < CORNERS.length; i++) {
            const [vx, vy, vz] = cornersRef.current[i];
            const clip = transform4(st.mvp, vx, vy, vz, 1);
            if (clip[3] <= 0) continue; // behind the camera — can't pick it
            const sx = (clip[0] / clip[3] * 0.5 + 0.5) * rect.width;
            const sy = (1 - (clip[1] / clip[3] * 0.5 + 0.5)) * rect.height;
            const d = Math.hypot(sx - cx, sy - cy);
            if (d < bestD) {
              bestD = d;
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

        // Of the three local axes, which one's screen projection best matches a
        // 2D drag delta? Returns the AXIS_DIRS index (0..2) or -1 if none is
        // projectable (all arrows behind the camera).
        const bestAxis = (
          st: NonNullable<typeof frameRef.current>,
          corner: readonly number[],
          dx: number,
          dy: number,
        ): number => {
          let best = -1;
          let bestScore = -Infinity;
          const dragLen = Math.hypot(dx, dy) || 1;
          for (let a = 0; a < AXIS_DIRS.length; a++) {
            const s = screenAxis(st, corner, AXIS_DIRS[a]);
            const len2 = s[0] * s[0] + s[1] * s[1];
            if (len2 < 1e-6) continue;
            const proj = Math.abs(dx * s[0] + dy * s[1]);
            const score = proj / (dragLen * Math.sqrt(len2));
            if (score > bestScore) {
              bestScore = score;
              best = a;
            }
          }
          return best;
        };

        // Is the press on the selected vertex's gizmo? Sample the corner and a
        // few points along each arrow shaft; a hit within the threshold means
        // "grab the gizmo" rather than "rotate the cube".
        const GRAB_RADIUS = 28;
        const grabGizmo = (clientX: number, clientY: number): boolean => {
          const st = frameRef.current;
          const idx = selectedRef.current;
          if (!st || idx === null) return false;
          const rect = el.getBoundingClientRect();
          const cx = clientX - rect.left;
          const cy = clientY - rect.top;
          const c = cornersRef.current[idx];
          const near = (p: number[]): boolean => {
            const clip = transform4(st.mvp, p[0], p[1], p[2], 1);
            if (clip[3] <= 0) return false;
            const sx = (clip[0] / clip[3] * 0.5 + 0.5) * rect.width;
            const sy = (1 - (clip[1] / clip[3] * 0.5 + 0.5)) * rect.height;
            return Math.hypot(sx - cx, sy - cy) < GRAB_RADIUS;
          };
          if (near(c)) return true;
          for (let a = 0; a < AXIS_DIRS.length; a++) {
            const d = AXIS_DIRS[a];
            for (let f = 0.4; f <= 1.0001; f += 0.3) {
              if (
                near([
                  c[0] + d[0] * GIZMO_TIP_DIST * f,
                  c[1] + d[1] * GIZMO_TIP_DIST * f,
                  c[2] + d[2] * GIZMO_TIP_DIST * f,
                ])
              ) {
                return true;
              }
            }
          }
          return false;
        };

        const onPointerDown = (e: PointerEvent) => {
          draggingRef.current = true;
          didDragRef.current = false;
          lastPointerRef.current = { x: e.clientX, y: e.clientY };
          downPointerRef.current = { x: e.clientX, y: e.clientY };
          // A press on the selected vertex's gizmo starts a vertex move instead
          // of a camera rotation.
          if (grabGizmo(e.clientX, e.clientY)) {
            gizmoDragRef.current = {
              idx: selectedRef.current as number,
              axis: null,
            };
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
          // of turning the camera. The motion axis is the local axis whose
          // screen projection best matches the drag; it is chosen on the first
          // significant movement, then held for the rest of the drag.
          const g = gizmoDragRef.current;
          if (g && st) {
            if (g.axis === null) {
              // Not enough movement yet to choose a stable axis.
              if (Math.hypot(adx, ady) < 3) {
                lastPointerRef.current = { x: e.clientX, y: e.clientY };
                return;
              }
              const ai = bestAxis(st, cornersRef.current[g.idx], adx, ady);
              if (ai < 0) {
                lastPointerRef.current = { x: e.clientX, y: e.clientY };
                return;
              }
              g.axis = AXIS_DIRS[ai];
            }
            // Project the per-frame screen step onto the chosen axis to get the
            // local displacement along it.
            const dx = e.clientX - lastPointerRef.current.x;
            const dy = e.clientY - lastPointerRef.current.y;
            const s = screenAxis(st, cornersRef.current[g.idx], g.axis);
            const len2 = s[0] * s[0] + s[1] * s[1];
            if (len2 > 1e-6) {
              const t = (dx * s[0] + dy * s[1]) / len2;
              const c = cornersRef.current[g.idx];
              c[0] = Math.max(-1.5, Math.min(1.5, c[0] + g.axis[0] * t));
              c[1] = Math.max(-1.5, Math.min(1.5, c[1] + g.axis[1] * t));
              c[2] = Math.max(-1.5, Math.min(1.5, c[2] + g.axis[2] * t));
              dirtyCubeRef.current = true;
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
        function drawOverlay() {
          octx!.clearRect(0, 0, overlay!.width, overlay!.height);
          const idx = selectedRef.current;
          const st = frameRef.current;
          if (idx === null || !st) {
            if (statusSpanRef.current)
              statusSpanRef.current.textContent = "no vertex selected";
            return;
          }
          const corner = cornersRef.current[idx];

          const clip = transform4(st.mvp, corner[0], corner[1], corner[2], 1);
          const inFront = clip[3] > 0;
          const hidden = inFront ? cornerHidden(st, corner) : true;
          const visible = inFront && !hidden;

          if (statusSpanRef.current)
            statusSpanRef.current.textContent = visible
              ? `vertex ${idx} · visible`
              : `vertex ${idx} · hidden (no highlight)`;

          if (!visible) return;

          const dpr = window.devicePixelRatio || 1;
          const px = (clip[0] / clip[3] * 0.5 + 0.5) * overlay!.width;
          const py = (1 - (clip[1] / clip[3] * 0.5 + 0.5)) * overlay!.height;
          // Filled core
          octx!.beginPath();
          octx!.arc(px, py, 3.5 * dpr, 0, Math.PI * 2);
          octx!.fillStyle = "rgba(255,165,0,1)";
          octx!.fill();
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
          const mvp = multiply(multiply(proj, view), model);
          device.queue.writeBuffer(uniformBuffer, 0, mvp);

          // Cache the transforms for the pick handler + occlusion test.
          frameRef.current = {
            mvp,
            invModel: multiply(rotationX(-pitchRef.current), rotationY(-yawRef.current)),
            cam: [panXRef.current, panYRef.current, distanceRef.current],
          };

          // Rebuild the cube's vertex buffer when a gizmo drag moved a corner,
          // so the dragged vertex reshapes the faces that share it.
          if (dirtyCubeRef.current) {
            device.queue.writeBuffer(
              vertexBuffer,
              0,
              buildCubeDataFromCorners(cornersRef.current),
            );
            dirtyCubeRef.current = false;
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
          pass.draw(36);

          // Vertex translation gizmo: when a vertex is selected, draw the three
          // axis arrows anchored on that corner. The arrows live in the cube's
          // local space, so post-multiplying the cube's MVP by a translation to
          // the corner places them on the vertex and lets them rotate with the
          // cube. (Dragging is not wired up yet — this is the visual handle.)
          const sel = selectedRef.current;
          if (sel !== null) {
            const c = cornersRef.current[sel];
            const gizmoMVP = multiply(mvp, translation(c[0], c[1], c[2]));
            device.queue.writeBuffer(gizmoUniform, 0, gizmoMVP);
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
          "WebGPU running — drag to rotate · shift+drag to pan · scroll to zoom · click a corner to select · drag a gizmo arrow to move it ✦",
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

      <div className="mt-4 flex items-center gap-3 text-sm text-gray-400 max-w-md text-center">
        <span className="inline-flex items-center gap-1.5">
          <span className="text-sky-400 font-semibold tabular-nums">
            vertex
          </span>
          <span
            ref={statusSpanRef}
            className="text-gray-300 tabular-nums"
          >
            no vertex selected
          </span>
        </span>
        <button
          type="button"
          onClick={() => {
            selectedRef.current = null;
            setSelected(null);
          }}
          disabled={selected === null}
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
