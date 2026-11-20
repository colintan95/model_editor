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

// ── Ray-casting hit testing ─────────────────────────────────────────────────────
// The pointer is turned into a 3D ray and tested against the geometry directly,
// which is orientation- and zoom-independent (unlike projecting points to screen
// and measuring a pixel distance). The ray is built in *world* space from
// inv(proj·view); the cube's own helpers expect local space, so the caller
// rotates the ray by invModel before testing it against the axis-aligned cube.

// Inverse of a column-major 4×4 matrix (gl-matrix mat4.invert). Returns null for
// a singular matrix.
export function invert(m: Float32Array | number[]): Float32Array<ArrayBuffer> | null {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  // 2×2 minors of the top-left 3×3 block (and the matching bottom-right ones),
  // used to build the determinant and the adjugate below.
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  const det =
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-20) return null;
  const idet = 1 / det;

  // adjugate = transpose(cofactor): out[col·4+row] = (−1)^(row+col)·minor(col,row).
  const out = new Float32Array(16);
  out[0] = (a11 * (a22 * a33 - a32 * a23) - a21 * (a12 * a33 - a32 * a13) + a31 * (a12 * a23 - a22 * a13)) * idet;
  out[1] = -(a01 * (a22 * a33 - a32 * a23) - a21 * (a02 * a33 - a32 * a03) + a31 * (a02 * a23 - a22 * a03)) * idet;
  out[2] = (a01 * (a12 * a33 - a32 * a13) - a11 * (a02 * a33 - a32 * a03) + a31 * (a02 * a13 - a12 * a03)) * idet;
  out[3] = -(a01 * (a12 * a23 - a22 * a13) - a11 * (a02 * a23 - a22 * a03) + a21 * (a02 * a13 - a12 * a03)) * idet;
  out[4] = -(a10 * (a22 * a33 - a32 * a23) - a20 * (a12 * a33 - a32 * a13) + a30 * (a12 * a23 - a22 * a13)) * idet;
  out[5] = (a00 * (a22 * a33 - a32 * a23) - a20 * (a02 * a33 - a32 * a03) + a30 * (a02 * a23 - a22 * a03)) * idet;
  out[6] = -(a00 * (a12 * a33 - a32 * a13) - a10 * (a02 * a33 - a32 * a03) + a30 * (a02 * a13 - a12 * a03)) * idet;
  out[7] = (a00 * (a12 * a23 - a22 * a13) - a10 * (a02 * a23 - a22 * a03) + a20 * (a02 * a13 - a12 * a03)) * idet;
  out[8] = (a10 * (a21 * a33 - a31 * a23) - a20 * (a11 * a33 - a31 * a13) + a30 * (a11 * a23 - a21 * a13)) * idet;
  out[9] = -(a00 * (a21 * a33 - a31 * a23) - a20 * (a01 * a33 - a31 * a03) + a30 * (a01 * a23 - a21 * a03)) * idet;
  out[10] = (a00 * (a11 * a33 - a31 * a13) - a10 * (a01 * a33 - a31 * a03) + a30 * (a01 * a13 - a11 * a03)) * idet;
  out[11] = -(a00 * (a11 * a23 - a21 * a13) - a10 * (a01 * a23 - a21 * a03) + a20 * (a01 * a13 - a11 * a03)) * idet;
  out[12] = -(a10 * (a21 * a32 - a31 * a22) - a20 * (a11 * a32 - a31 * a12) + a30 * (a11 * a22 - a21 * a12)) * idet;
  out[13] = (a00 * (a21 * a32 - a31 * a22) - a20 * (a01 * a32 - a31 * a02) + a30 * (a01 * a22 - a21 * a02)) * idet;
  out[14] = -(a00 * (a11 * a32 - a31 * a12) - a10 * (a01 * a32 - a31 * a02) + a30 * (a01 * a12 - a11 * a02)) * idet;
  out[15] = (a00 * (a11 * a22 - a21 * a12) - a10 * (a01 * a22 - a21 * a02) + a20 * (a01 * a12 - a11 * a02)) * idet;
  return out;
}

// Build a picking ray in the space that `invViewProj` maps *from* (world space
// when it is the inverse of proj·view). `cx`/`cy` are CSS pixels within a
// `rectW`×`rectH` viewport. Returns a normalized [origin, dir], or null when the
// point is behind the camera.
export function rayFromCursor(
  invViewProj: Float32Array | number[],
  cx: number,
  cy: number,
  rectW: number,
  rectH: number,
): {
  ox: number;
  oy: number;
  oz: number;
  dx: number;
  dy: number;
  dz: number;
} | null {
  // CSS pixels → NDC. Y is flipped (screen top is +NDC-y).
  const ndcx = (cx / rectW) * 2 - 1;
  const ndcy = 1 - (cy / rectH) * 2;
  // Unproject near (z = -1) and far (z = +1) points, then divide by w.
  const near = transform4(invViewProj, ndcx, ndcy, -1, 1);
  const far = transform4(invViewProj, ndcx, ndcy, 1, 1);
  if (near[3] === 0 || far[3] === 0) return null;
  const nx = near[0] / near[3];
  const ny = near[1] / near[3];
  const nz = near[2] / near[3];
  const fx = far[0] / far[3];
  const fy = far[1] / far[3];
  const fz = far[2] / far[3];
  let dx = fx - nx;
  let dy = fy - ny;
  let dz = fz - nz;
  const dl = Math.hypot(dx, dy, dz) || 1;
  dx /= dl;
  dy /= dl;
  dz /= dl;
  return { ox: nx, oy: ny, oz: nz, dx, dy, dz };
}

// First intersection parameter t (≥ 0) of a ray with a sphere, or null on miss.
// Ray: P(t) = origin + t·dir, with `dir` normalized so t is in world units.
export function raySphere(
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  cx: number,
  cy: number,
  cz: number,
  r: number,
): number | null {
  const mx = ox - cx;
  const my = oy - cy;
  const mz = oz - cz;
  const b = mx * dx + my * dy + mz * dz;
  const c = mx * mx + my * my + mz * mz - r * r;
  const disc = b * b - c;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const t1 = -b - sq; // near intersection
  if (t1 > 1e-9) return t1;
  const t2 = -b + sq; // far intersection — the relevant one when the ray starts inside
  return t2 > 1e-9 ? t2 : null; // both behind the ray ⇒ miss
}

// First intersection parameter t (≥ 0) of a ray with a capsule — a segment
// a→b thickened to radius `r` (a cylinder capped by two hemispheres). Returns
// null on a miss. Used to hit-test the gizmo arrows exactly, which also reports
// *which* arrow (and therefore which axis) was struck.
export function rayCapsule(
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  r: number,
): number | null {
  // The two rounded ends are full spheres around each endpoint.
  let t = raySphere(ox, oy, oz, dx, dy, dz, ax, ay, az, r);
  const tb = raySphere(ox, oy, oz, dx, dy, dz, bx, by, bz, r);
  if (tb !== null) t = t === null ? tb : Math.min(t, tb);

  // Cylindrical middle: project the ray into the plane perpendicular to the
  // segment axis (which sits at the origin of that plane) and solve for where
  // its distance to the axis equals r, keeping only the hit whose axial
  // coordinate lies between the two endpoints.
  const kx = bx - ax;
  const ky = by - ay;
  const kz = bz - az;
  const L2 = kx * kx + ky * ky + kz * kz;
  if (L2 < 1e-12) return t; // degenerate segment — the caps above already cover it
  const L = Math.sqrt(L2);
  const ikx = kx / L;
  const iky = ky / L;
  const ikz = kz / L;

  // Ray origin relative to the segment start, split into along-axis + perp parts.
  const mx = ox - ax;
  const my = oy - ay;
  const mz = oz - az;
  const mDotK = mx * ikx + my * iky + mz * ikz;
  const mpx = mx - mDotK * ikx;
  const mpy = my - mDotK * iky;
  const mpz = mz - mDotK * ikz;
  const dDotK = dx * ikx + dy * iky + dz * ikz;
  const dpx = dx - dDotK * ikx;
  const dpy = dy - dDotK * iky;
  const dpz = dz - dDotK * ikz;

  const A = dpx * dpx + dpy * dpy + dpz * dpz;
  if (A < 1e-12) return t; // ray parallel to the axis — the caps already cover it
  const B = 2 * (mpx * dpx + mpy * dpy + mpz * dpz);
  const C = mpx * mpx + mpy * mpy + mpz * mpz - r * r;
  const disc = B * B - 4 * A * C;
  if (disc < 0) return t;
  const sq = Math.sqrt(disc);
  let s = (-B - sq) / (2 * A);
  if (s < 0) s = (-B + sq) / (2 * A);
  if (s < 0) return t;
  const zAxis = mDotK + s * dDotK; // axial position of the hit
  if (zAxis < 0 || zAxis > L) return t; // struck a rounded end, not the shaft
  return t === null ? s : Math.min(t, s);
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
