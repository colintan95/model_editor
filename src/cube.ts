// ── Cube geometry + 4×4 matrix math (column-major, WebGPU convention) ─────────
// Pure, framework-free functions so the 3D maths can be unit-tested in isolation.

export type Vec3 = [number, number, number];

export interface Face {
  color: [number, number, number, number];
  verts: Vec3[];
}

const HALF = 0.5;

// 6 faces × 2 triangles × 3 vertices = 36 vertices, coloured per face.
// Interleaved layout per vertex: [x, y, z, r, g, b, a] → stride 28 bytes.
export const FACES: Face[] = [
  {
    // +Z (front)
    color: [1.0, 0.12, 0.12, 1.0],
    verts: [
      [-HALF, -HALF, HALF], [HALF, -HALF, HALF], [HALF, HALF, HALF],
      [-HALF, -HALF, HALF], [HALF, HALF, HALF], [-HALF, HALF, HALF],
    ],
  },
  {
    // -Z (back)
    color: [0.12, 1.0, 0.12, 1.0],
    verts: [
      [HALF, -HALF, -HALF], [-HALF, -HALF, -HALF], [-HALF, HALF, -HALF],
      [HALF, -HALF, -HALF], [-HALF, HALF, -HALF], [HALF, HALF, -HALF],
    ],
  },
  {
    // +Y (top)
    color: [0.2, 0.4, 1.0, 1.0],
    verts: [
      [-HALF, HALF, HALF], [HALF, HALF, HALF], [HALF, HALF, -HALF],
      [-HALF, HALF, HALF], [HALF, HALF, -HALF], [-HALF, HALF, -HALF],
    ],
  },
  {
    // -Y (bottom)
    color: [1.0, 0.85, 0.1, 1.0],
    verts: [
      [-HALF, -HALF, -HALF], [HALF, -HALF, -HALF], [HALF, -HALF, HALF],
      [-HALF, -HALF, -HALF], [HALF, -HALF, HALF], [-HALF, -HALF, HALF],
    ],
  },
  {
    // +X (right)
    color: [0.1, 0.9, 0.9, 1.0],
    verts: [
      [HALF, -HALF, HALF], [HALF, -HALF, -HALF], [HALF, HALF, -HALF],
      [HALF, -HALF, HALF], [HALF, HALF, -HALF], [HALF, HALF, HALF],
    ],
  },
  {
    // -X (left)
    color: [0.9, 0.2, 0.95, 1.0],
    verts: [
      [-HALF, -HALF, -HALF], [-HALF, -HALF, HALF], [-HALF, HALF, HALF],
      [-HALF, -HALF, -HALF], [-HALF, HALF, HALF], [-HALF, HALF, -HALF],
    ],
  },
];

export const VERTICES_PER_FACE = 6;
export const FLOATS_PER_VERTEX = 7; // xyz + rgba
export const BYTE_STRIDE = FLOATS_PER_VERTEX * 4; // 28

export function buildCubeData(): Float32Array<ArrayBuffer> {
  const data = new Float32Array(FACES.length * VERTICES_PER_FACE * FLOATS_PER_VERTEX);
  let i = 0;
  for (const face of FACES) {
    for (const v of face.verts) {
      data[i++] = v[0];
      data[i++] = v[1];
      data[i++] = v[2];
      data[i++] = face.color[0];
      data[i++] = face.color[1];
      data[i++] = face.color[2];
      data[i++] = face.color[3];
    }
  }
  return data;
}

export function perspective(
  fovy: number,
  aspect: number,
  near: number,
  far: number,
): Float32Array<ArrayBuffer> {
  const f = 1 / Math.tan(fovy / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

export function translation(x: number, y: number, z: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(16);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[15] = 1;
  out[12] = x;
  out[13] = y;
  out[14] = z;
  return out;
}

export function rotationX(theta: number): Float32Array<ArrayBuffer> {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const out = new Float32Array(16);
  out[0] = 1;
  out[5] = c;
  out[6] = s;
  out[9] = -s;
  out[10] = c;
  out[15] = 1;
  return out;
}

export function rotationY(theta: number): Float32Array<ArrayBuffer> {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const out = new Float32Array(16);
  out[0] = c;
  out[2] = -s;
  out[4] = 0;
  out[5] = 1;
  out[8] = s;
  out[10] = c;
  out[15] = 1;
  return out;
}

// result = a * b  (column-major, applied as out = a * b * v)
export function multiply(a: Float32Array, b: Float32Array): Float32Array<ArrayBuffer> {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row] * b[col * 4 + k];
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}
