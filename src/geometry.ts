// ── 3D geometry + 4×4 matrix math (column-major, WebGPU convention) ─────────
// Cube + gizmo-arrow meshes, camera transforms, and picking/occlusion helpers.
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

// The 8 corners as a fresh, independently-mutable array (a deep copy of the
// canonical CORNERS), so a caller can drag individual vertices without touching
// the shared CORNERS array.
export function defaultCorners(): Vec3[] {
  return CORNERS.map((c) => [c[0], c[1], c[2]] as Vec3);
}

// Which of the 8 unique corners does a face vertex correspond to? FACES stores
// the canonical ±HALF positions, so a coordinate match finds the index.
function matchCorner(v: Vec3): number {
  for (let i = 0; i < CORNERS.length; i++) {
    const c = CORNERS[i];
    if (
      Math.abs(c[0] - v[0]) < 1e-6 &&
      Math.abs(c[1] - v[1]) < 1e-6 &&
      Math.abs(c[2] - v[2]) < 1e-6
    ) {
      return i;
    }
  }
  return 0;
}

// Like buildCubeData, but the 8 vertex positions come from `corners` instead of
// the canonical ±HALF values, so dragging one corner reshapes every face that
// shares it (per-vertex colour still comes from the face it belongs to).
export function buildCubeDataFromCorners(
  corners: Vec3[],
): Float32Array<ArrayBuffer> {
  const data = new Float32Array(
    FACES.length * VERTICES_PER_FACE * FLOATS_PER_VERTEX,
  );
  let i = 0;
  for (const face of FACES) {
    for (const v of face.verts) {
      const c = corners[matchCorner(v)];
      data[i++] = c[0];
      data[i++] = c[1];
      data[i++] = c[2];
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

// ── Picking / overlay helpers ────────────────────────────────────────────────

// The 8 unique cube corners (the 36 interleaved vertices in FACES duplicate
// these). Each is at (±HALF, ±HALF, ±HALF) in the cube's local space.
export const CORNERS: Vec3[] = [
  [-HALF, -HALF, -HALF],
  [HALF, -HALF, -HALF],
  [HALF, -HALF, HALF],
  [-HALF, -HALF, HALF],
  [-HALF, HALF, -HALF],
  [HALF, HALF, -HALF],
  [HALF, HALF, HALF],
  [-HALF, HALF, HALF],
];

// Transform a homogeneous point by a column-major 4×4 matrix.
// out_row = Σ_col m[col*4 + row] * v[col]  (matches the multiply() convention).
export function transform4(
  m: Float32Array | number[],
  x: number,
  y: number,
  z: number,
  w: number = 1,
): [number, number, number, number] {
  const ox = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
  const oy = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
  const oz = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
  const ow = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
  return [ox, oy, oz, ow];
}

// Near-intersection of a ray with the axis-aligned cube [-0.5, 0.5]³, returned
// as a parameter t along a (not necessarily unit) direction:
//   point(t) = origin + t * dir
// So the corner we are testing sits at t = 1, and any t < 1 means the ray poked
// through the cube *before* reaching the corner — i.e. the corner is hidden
// behind a face. Returns null when the ray misses the cube entirely.
export function rayCubeNearT(
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
): number | null {
  let t0 = 0;
  let t1 = Infinity;
  for (let axis = 0; axis < 3; axis++) {
    const o = axis === 0 ? ox : axis === 1 ? oy : oz;
    const d = axis === 0 ? dx : axis === 1 ? dy : dz;
    if (d === 0) {
      // Ray parallel to this slab: it only misses if the origin is outside it.
      if (o < -0.5 || o > 0.5) return null;
      continue;
    }
    const inv = 1 / d;
    let ta = (-0.5 - o) * inv;
    let tb = (0.5 - o) * inv;
    if (ta > tb) {
      const tmp = ta;
      ta = tb;
      tb = tmp;
    }
    if (ta > t0) t0 = ta;
    if (tb < t1) t1 = tb;
    if (t0 > t1) return null;
  }
  return t0 > 0 ? t0 : t1;
}

// ── Vertex translation gizmo ────────────────────────────────────────────────
// Three arrows — one per positive axis — that sit on the selected vertex and
// rotate with the cube. Each arrow is a short cylindrical shaft topped by a
// flared cone, built as a solid of revolution around its axis so a single
// routine produces all three. The per-vertex layout matches the cube
// (xyz + rgba, stride 28) so the gizmo can be drawn with the *same* pipeline.
// Colours follow the X = red, Y = green, Z = blue convention.

const ARROW_SHAFT_LEN = 0.3;
const ARROW_CONE_LEN = 0.085;
const ARROW_SHAFT_R = 0.016;
const ARROW_CONE_R = 0.045;
const ARROW_SEGMENTS = 8;

// Total length from an arrow's base to its tip. Used to place the heads and to
// sample points along the shafts when hit-testing a gizmo grab.
export const GIZMO_TIP_DIST = ARROW_SHAFT_LEN + ARROW_CONE_LEN;

export const AXIS_DIRS: Vec3[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

const AXIS_COLORS: [number, number, number, number][] = [
  [0.95, 0.16, 0.16, 1.0], // X
  [0.16, 0.9, 0.3, 1.0], // Y
  [0.22, 0.45, 0.96, 1.0], // Z
];

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

// Build the three gizmo arrows, centred on the origin and pointing along +X,
// +Y, +Z. The caller places them on the selected vertex and rotates them with
// the model by post-multiplying the gizmo MVP with a translation.
export function buildArrows(): Float32Array<ArrayBuffer> {
  // Solid-of-revolution profile: (radius, position-along-axis). A flat shaft,
  // then a flared cone that comes to a point.
  const rings: [number, number][] = [
    [ARROW_SHAFT_R, 0],
    [ARROW_SHAFT_R, ARROW_SHAFT_LEN],
    [ARROW_CONE_R, ARROW_SHAFT_LEN],
    [0, ARROW_SHAFT_LEN + ARROW_CONE_LEN],
  ];

  const out: number[] = [];

  // Append one xyz + rgba vertex.
  const vert = (
    x: number,
    y: number,
    z: number,
    c: [number, number, number, number],
  ) => {
    out.push(x, y, z, c[0], c[1], c[2], c[3]);
  };

  for (let a = 0; a < AXIS_DIRS.length; a++) {
    const dir = AXIS_DIRS[a];
    const color = AXIS_COLORS[a];

    // Orthonormal frame (u, v) spanning the plane perpendicular to dir, used to
    // lay ring vertices around the axis.
    const up: Vec3 = Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = normalize(cross(up, dir));
    const v = cross(dir, u);

    const n = ARROW_SEGMENTS;

    // The vertex of ring `ringIdx` at angular slot `seg`.
    const ringVertex = (ringIdx: number, seg: number): Vec3 => {
      const [r, len] = rings[ringIdx];
      const ang = (seg / n) * Math.PI * 2;
      const ca = Math.cos(ang) * r;
      const sa = Math.sin(ang) * r;
      return [
        dir[0] * len + u[0] * ca + v[0] * sa,
        dir[1] * len + u[1] * ca + v[1] * sa,
        dir[2] * len + u[2] * ca + v[2] * sa,
      ];
    };

    // Base cap: a fan at ring 0 closing the shaft end that rests on the vertex.
    for (let s = 0; s < n; s++) {
      const p0 = ringVertex(0, s);
      const p1 = ringVertex(0, s + 1);
      vert(0, 0, 0, color);
      vert(p0[0], p0[1], p0[2], color);
      vert(p1[0], p1[1], p1[2], color);
    }

    // Side bands connecting consecutive rings (shaft, then the flared cone).
    for (let i = 0; i < rings.length - 1; i++) {
      for (let s = 0; s < n; s++) {
        const a0 = ringVertex(i, s);
        const a1 = ringVertex(i, s + 1);
        const b0 = ringVertex(i + 1, s);
        const b1 = ringVertex(i + 1, s + 1);
        vert(a0[0], a0[1], a0[2], color);
        vert(a1[0], a1[1], a1[2], color);
        vert(b1[0], b1[1], b1[2], color);
        vert(a0[0], a0[1], a0[2], color);
        vert(b1[0], b1[1], b1[2], color);
        vert(b0[0], b0[1], b0[2], color);
      }
    }
  }

  return new Float32Array(out);
}
