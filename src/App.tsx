import { useEffect, useRef, useState } from "react";
import {
  buildCubeData,
  perspective,
  translation,
  rotationX,
  rotationY,
  multiply,
  BYTE_STRIDE,
} from "./cube";

// ── WGSL shader ───────────────────────────────────────────────────────────────
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

// ── Component ─────────────────────────────────────────────────────────────────

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<string>("Initialising WebGPU…");
  const [error, setError] = useState<string | null>(null);

  // Live, mutable rotation state shared between the drag handlers and the
  // render loop without forcing a re-render (or restarting the effect).
  const yawRef = useRef(0.6);
  const pitchRef = useRef(0.35);
  const draggingRef = useRef(false);
  const lastPointerRef = useRef({ x: 0, y: 0 });
  const distanceRef = useRef(3); // camera distance — the wheel zooms this
  const panXRef = useRef(0); // camera pan in world units — shift+drag
  const panYRef = useRef(0);

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
        const onPointerDown = (e: PointerEvent) => {
          draggingRef.current = true;
          lastPointerRef.current = { x: e.clientX, y: e.clientY };
          canvas.setPointerCapture(e.pointerId);
        };
        const onPointerMove = (e: PointerEvent) => {
          if (!draggingRef.current) return;
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
        function render() {
          if (cancelled) return;

          const aspect = el.width / el.height;
          const proj = perspective(FOV, aspect, 0.1, 100);
          const view = translation(-panXRef.current, -panYRef.current, -distanceRef.current);
          const model = multiply(
            rotationY(yawRef.current),
            rotationX(pitchRef.current),
          );
          const mvp = multiply(multiply(proj, view), model);
          device.queue.writeBuffer(uniformBuffer, 0, mvp);

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
          pass.end();

          device.queue.submit([encoder.finish()]);
          frame = requestAnimationFrame(render);
        }

        setStatus("WebGPU running — drag to rotate · shift+drag to pan · scroll to zoom ✔");
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

      <div className="w-full max-w-2xl aspect-video rounded-xl overflow-hidden shadow-2xl ring-1 ring-white/10">
        <canvas
          ref={canvasRef}
          className="w-full h-full block bg-gray-900 cursor-grab active:cursor-grabbing select-none"
          style={{ touchAction: "none" }}
        />
      </div>

      {error ? (
        <p className="mt-4 text-red-400 text-sm max-w-md text-center">
          {error}
        </p>
      ) : (
        <p className="mt-4 text-gray-400 text-sm">{status}</p>
      )}
    </div>
  );
}
