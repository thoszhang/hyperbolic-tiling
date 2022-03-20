const RADIUS = 800;
const CANVAS_SIZE = 2 * RADIUS;

type C = { readonly re: number; readonly im: number };
type C2 = readonly [C, C]; // Element of C^2, or CP^1.
type Mat = readonly [C, C, C, C]; // Row-major 2x2 matrix.

const zero = fromReal(0);
const one = fromReal(1);
const id: Mat = fromCol(zero, one);

function fromCol(v: C, w: C): Mat {
  return [conj(w), v, conj(v), w];
}

function rotation(angle: number): Mat {
  return fromCol(zero, fromPolar(1, -angle / 2));
}

function translation(dist: number): Mat {
  return fromCol(fromReal(Math.sinh(dist / 2)), fromReal(Math.cosh(dist / 2)));
}

function invDet1(m: Mat): Mat {
  return [m[3], neg(m[1]), neg(m[2]), m[0]];
}

function refl(line: Mat): Mat {
  return mulMat(line, invDet1(conjMat(line)));
}

function conjMat(m: Mat): Mat {
  return [conj(m[0]), conj(m[1]), conj(m[2]), conj(m[3])];
}

function mulMat(m1: Mat, m2: Mat): Mat {
  return [
    add(mul(m1[0], m2[0]), mul(m1[1], m2[2])),
    add(mul(m1[0], m2[1]), mul(m1[1], m2[3])),
    add(mul(m1[2], m2[0]), mul(m1[3], m2[2])),
    add(mul(m1[2], m2[1]), mul(m1[3], m2[3])),
  ];
}

function fromPolar(mod: number, arg: number): C {
  return { re: mod * Math.cos(arg), im: mod * Math.sin(arg) };
}

function fromReal(x: number): C {
  return { re: x, im: 0 };
}

function neg(z: C): C {
  return { re: -z.re, im: -z.im };
}

function conj(z: C): C {
  return { re: z.re, im: -z.im };
}

function modSq(z: C): number {
  return z.re ** 2 + z.im ** 2;
}

function add(z1: C, z2: C): C {
  return { re: z1.re + z2.re, im: z1.im + z2.im };
}

function realMul(a: number, z: C): C {
  return { re: a * z.re, im: a * z.im };
}

function mul(z1: C, z2: C): C {
  return {
    re: z1.re * z2.re - z1.im * z2.im,
    im: z1.im * z2.re + z1.re * z2.im,
  };
}

function div(z1: C, z2: C): C {
  return realMul(1 / modSq(z2), mul(z1, conj(z2)));
}

function apply(m: Mat, p: C2): C2 {
  return [
    add(mul(m[0], p[0]), mul(m[1], p[1])),
    add(mul(m[2], p[0]), mul(m[3], p[1])),
  ];
}

function quadraticRoots(
  a: number,
  b: number,
  c: number
): [number, number] | undefined {
  if (a === 0) {
    return [-b / c, -b / c];
  }
  const d = b * b - 4 * a * c;
  if (d < 0) {
    return undefined;
  }
  const sqrtD = Math.sqrt(d);
  if (b >= 0) {
    return [(-b - sqrtD) / (2 * a), (2 * c) / (-b - sqrtD)];
  } else {
    return [(2 * c) / (-b + sqrtD), (-b + sqrtD) / (2 * a)];
  }
}

function realLineIntxns(line: Mat): [number, number] | undefined {
  const v = line[1];
  const w = line[3];
  const roots = quadraticRoots(
    -v.re * w.im + v.im * w.re,
    2 * (v.re * v.im - w.re * w.im),
    -v.re * w.im + v.im * w.re
  );
  if (!roots) {
    return undefined;
  }
  const intersections = roots.map((r) =>
    Math.sqrt(modSq(dehomogenize(apply(line, homogenize(fromReal(r))))))
  );
  if (intersections[1] > intersections[0]) {
    return [intersections[0], intersections[1]];
  } else {
    return [intersections[1], intersections[0]];
  }
}

function homogenize(z: C): C2 {
  return [z, one];
}

function dehomogenize(p: C2): C {
  return div(p[0], p[1]);
}

type Params = { p: number; q: number; r: number };
type Mode = "triangles" | "r|pq" | "rp|q" | "pqr|";

function modeIndex(m: Mode) {
  switch (m) {
    case "triangles":
      return 0;
    case "r|pq":
      return 1;
    case "rp|q":
      return 2;
    case "pqr|":
      return 3;
  }
}

type DrawData = {
  matA: Mat;
  matB: Mat;
  matC: Mat;
  invHalfPlA: Mat;
  invHalfPlB: Mat;
  invHalfPlC: Mat;
  t: undefined | r_pqData | rp_qData | pqr_Data;
};
type r_pqData = {
  invHalfPl1: Mat;
};
type rp_qData = {
  invHalfPl1: Mat;
  invHalfPl2: Mat;
};
type pqr_Data = {
  invHalfPl1: Mat;
  invHalfPl2: Mat;
  invHalfPl3: Mat;
};

function calculateData(params: Params, mode: Mode): DrawData {
  const a = Math.PI / params.p;
  const b = Math.PI / params.q;
  const c = Math.PI / params.r;
  const sin_a = Math.sin(a);
  const cos_a = Math.cos(a);
  const sin_b = Math.sin(b);
  const cos_b = Math.cos(b);
  const sin_c = Math.sin(c);
  const cos_c = Math.cos(c);

  const coshC = (cos_c + cos_a * cos_b) / (sin_a * sin_b);
  const coshB = (cos_b + cos_c * cos_a) / (sin_c * sin_a);
  const C = Math.acosh(coshC);

  const halfPlaneC: Mat = id;
  const halfPlaneB: Mat = rotation(Math.PI + a);
  const halfPlaneA: Mat = mulMat(translation(C), rotation(Math.PI - b));

  const matC: Mat = refl(halfPlaneC);
  const matB: Mat = refl(halfPlaneB);
  const matA: Mat = refl(halfPlaneA);

  function r_pq(): r_pqData {
    const halfPlane: Mat = mulMat(
      translation(Math.atanh(cos_a * Math.sqrt(1 - Math.pow(coshB, -2)))),
      rotation(Math.PI / 2)
    );
    return {
      invHalfPl1: invDet1(halfPlane),
    };
  }

  function rp_q(): rp_qData {
    const euclDistToPoint = (
      realLineIntxns(mulMat(rotation(-a / 2), halfPlaneA)) as [number, number]
    )[0];

    const halfPlane1: Mat = mulMat(
      translation(
        Math.atanh(
          Math.cos(a / 2) * 2 * (euclDistToPoint / (1 + euclDistToPoint ** 2))
        )
      ),
      rotation(Math.PI / 2)
    );
    const halfPlane2: Mat = mulMat(refl(rotation(a / 2)), halfPlane1);

    return {
      invHalfPl1: invDet1(halfPlane1),
      invHalfPl2: invDet1(halfPlane2),
    };
  }

  function pqr_(): pqr_Data {
    const euclDistToPoint = (
      realLineIntxns(
        mulMat(rotation(-a / 2), mulMat(translation(C), rotation(-b / 2)))
      ) as [number, number]
    )[0];

    const halfPlane1: Mat = mulMat(
      translation(
        Math.atanh(
          Math.cos(a / 2) * 2 * (euclDistToPoint / (1 + euclDistToPoint ** 2))
        )
      ),
      rotation(Math.PI / 2)
    );
    const halfPlane2: Mat = mulMat(refl(rotation(a / 2)), conjMat(halfPlane1));
    const halfPlane3: Mat = mulMat(
      mulMat(translation(C), mulMat(refl(rotation(-b / 2)), translation(-C))),
      conjMat(halfPlane1)
    );

    return {
      invHalfPl1: invDet1(halfPlane1),
      invHalfPl2: invDet1(halfPlane2),
      invHalfPl3: invDet1(halfPlane3),
    };
  }

  let tilingData: undefined | r_pqData | rp_qData | pqr_Data;
  switch (mode) {
    case "triangles":
      tilingData = undefined;
      break;
    case "r|pq":
      tilingData = r_pq();
      break;
    case "rp|q":
      tilingData = rp_q();
      break;
    case "pqr|":
      tilingData = pqr_();
      break;
  }

  return {
    matA: matA,
    matB: matB,
    matC: matC,
    invHalfPlA: invDet1(halfPlaneA),
    invHalfPlB: invDet1(halfPlaneB),
    invHalfPlC: invDet1(halfPlaneC),
    t: tilingData,
  };
}

function mat4(m: Mat): number[] {
  const r: number[] = Array.from({ length: 16 }, () => 0);
  r[0] = m[0].re;
  r[1] = m[0].im;
  r[2] = m[2].re;
  r[3] = m[2].im;
  r[4] = -m[0].im;
  r[5] = m[0].re;
  r[6] = -m[2].im;
  r[7] = m[2].re;
  r[8] = m[1].re;
  r[9] = m[1].im;
  r[10] = m[3].re;
  r[11] = m[3].im;
  r[12] = -m[1].im;
  r[13] = m[1].re;
  r[14] = -m[3].im;
  r[15] = m[3].re;
  return r;
}

const vsSource = /* glsl */ `#version 300 es
#pragma vscode_glsllint_stage : vert

in vec4 a_position;

void main() {
  gl_Position = a_position;
}
`;

const fsSource = /* glsl */ `#version 300 es
#pragma vscode_glsllint_stage : frag

precision highp float;

uniform vec2 u_resolution;
uniform mat4 u_matA;
uniform mat4 u_matB;
uniform mat4 u_matC;
uniform mat4 u_invHalfPlA;
uniform mat4 u_invHalfPlB;
uniform mat4 u_invHalfPlC;
uniform mat4 u_invHalfPl1;
uniform mat4 u_invHalfPl2;
uniform mat4 u_invHalfPl3;
uniform int u_mode;
uniform int u_showTriangles;

out vec4 outputColor;

mat2 complex(float re, float im) {
  return mat2(re, im, -im, re);
}

mat2x4 homogenize(mat2 z) {
  return mat2x4(z[0], 1.0, 0.0, z[1], 0.0, 1.0);
}

mat2 dehomogenize(mat2x4 p) {
  return mat2(p[0].xy, p[1].xy) * inverse(mat2(p[0].zw, p[1].zw));
}

mat2x4 conj(mat2x4 p) {
  return mat2x4(
    p[0].x, p[1].x, p[0].z, p[1].z,
    p[0].y, p[1].y, p[0].w, p[1].w
  );
}

bool inHalfPlane(mat4 invPlane, mat2x4 p) {
  return dehomogenize(invPlane * p)[0].y >= 0.0;
}

int r_pqRegion(mat2x4 p) {
  if (inHalfPlane(u_invHalfPl1, p)) {
    return 0;
  }
  return 1;
}

int rp_qRegion(mat2x4 p) {
  if (!inHalfPlane(u_invHalfPl1, p)) {
    return 1;
  }
  if (!inHalfPlane(u_invHalfPl2, p)) {
    return 2;
  }
  return 0;
}

int pqr_Region(mat2x4 p) {
  bool ins1 = inHalfPlane(u_invHalfPl1, p);
  bool ins2 = inHalfPlane(u_invHalfPl2, p);
  bool ins3 = inHalfPlane(u_invHalfPl3, p);

  if (ins2 && !ins3) {
    return 1;
  }
  if (ins3 && !ins1) {
    return 2;
  }
  return 0;
}

void main() {
  mat2 z = complex(
    2.0 * gl_FragCoord.x / u_resolution.x - 1.0,
    1.0 - 2.0 * gl_FragCoord.y / u_resolution.y
  );

  if (determinant(z) > 1.0) {
    outputColor = vec4(1, 1, 1, 1);
    return;
  }

  mat2x4 p = homogenize(z);
  bool insA = inHalfPlane(u_invHalfPlA, p);
  bool insB = inHalfPlane(u_invHalfPlB, p);
  bool insC = inHalfPlane(u_invHalfPlC, p);
  int numRefls = 0;

  while ((!insA || !insB || !insC) && numRefls < 50) {
    if (!insA) {
      p = u_matA * conj(p);
    } else if (!insB) {
      p = u_matB * conj(p);
    } else {
      p = u_matC * conj(p);
    }
    insA = inHalfPlane(u_invHalfPlA, p);
    insB = inHalfPlane(u_invHalfPlB, p);
    insC = inHalfPlane(u_invHalfPlC, p);
    numRefls += 1;
  }

  if (!insA || !insB || !insC) {
    outputColor = vec4(.5, 0.5, 0.5, 1);
    return;
  }

  int triangleRegion = numRefls % 2;
  float val;
  if (u_mode == 0) {
    val = float(triangleRegion);
  } else {
    int region;
    if (u_mode == 1) {
      region = r_pqRegion(p);
    } else if (u_mode == 2) {
      region = rp_qRegion(p);
    } else if (u_mode == 3) {
      region = pqr_Region(p);
    }
    val = (float(region) + 1.0) / 4.0;
    if (u_showTriangles == 1) {
      val += (2.0 * float(triangleRegion) - 1.0) / 16.0;
    }
  }

  outputColor = vec4(val, val, val, 1);
}
`;

function initShaderProgram(
  gl: WebGL2RenderingContext,
  vsSource: string,
  fsSource: string
): WebGLProgram | null {
  const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
  if (vertexShader === null) {
    return null;
  }
  const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
  if (fragmentShader === null) {
    return null;
  }

  const shaderProgram = gl.createProgram();
  if (shaderProgram === null) {
    console.error("could not create program");
    return null;
  }

  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(shaderProgram));
    return null;
  }

  return shaderProgram;
}

function loadShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (shader === null) {
    console.error("could not create shader");
    return null;
  }

  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

function main(): void {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;

  const maybeGl = canvas.getContext("webgl2");
  if (!maybeGl) {
    console.error("could not initialize WebGL");
    return;
  }
  const gl: WebGL2RenderingContext = maybeGl;
  const program = initShaderProgram(gl, vsSource, fsSource);
  if (!program) {
    return;
  }

  const positionAttributeLocation = gl.getAttribLocation(program, "a_position");

  const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
  const matALocation = gl.getUniformLocation(program, "u_matA");
  const matBLocation = gl.getUniformLocation(program, "u_matB");
  const matCLocation = gl.getUniformLocation(program, "u_matC");
  const invHalfPlALocation = gl.getUniformLocation(program, "u_invHalfPlA");
  const invHalfPlBLocation = gl.getUniformLocation(program, "u_invHalfPlB");
  const invHalfPlCLocation = gl.getUniformLocation(program, "u_invHalfPlC");
  const invHalfPl1Location = gl.getUniformLocation(program, "u_invHalfPl1");
  const invHalfPl2Location = gl.getUniformLocation(program, "u_invHalfPl2");
  const invHalfPl3Location = gl.getUniformLocation(program, "u_invHalfPl3");
  const modeLocation = gl.getUniformLocation(program, "u_mode");
  const showTrianglesLocation = gl.getUniformLocation(
    program,
    "u_showTriangles"
  );

  const positionBuffer = gl.createBuffer();
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.enableVertexAttribArray(positionAttributeLocation);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW
  );
  gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

  function render(data: DrawData, mode: Mode, showTriangles: boolean): void {
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(program);
    gl.bindVertexArray(vao);

    gl.uniform2f(resolutionLocation, gl.canvas.width, gl.canvas.height);
    gl.uniformMatrix4fv(matALocation, false, mat4(data.matA));
    gl.uniformMatrix4fv(matBLocation, false, mat4(data.matB));
    gl.uniformMatrix4fv(matCLocation, false, mat4(data.matC));
    gl.uniformMatrix4fv(invHalfPlALocation, false, mat4(data.invHalfPlA));
    gl.uniformMatrix4fv(invHalfPlBLocation, false, mat4(data.invHalfPlB));
    gl.uniformMatrix4fv(invHalfPlCLocation, false, mat4(data.invHalfPlC));
    if (data.t && "invHalfPl1" in data.t) {
      gl.uniformMatrix4fv(invHalfPl1Location, false, mat4(data.t.invHalfPl1));
    }
    if (data.t && "invHalfPl2" in data.t) {
      gl.uniformMatrix4fv(invHalfPl2Location, false, mat4(data.t.invHalfPl2));
    }
    if (data.t && "invHalfPl3" in data.t) {
      gl.uniformMatrix4fv(invHalfPl3Location, false, mat4(data.t.invHalfPl3));
    }
    gl.uniform1i(modeLocation, modeIndex(mode));
    gl.uniform1i(showTrianglesLocation, showTriangles ? 1 : 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  const initialParams = { p: 4, q: 3, r: 3 };

  const pInput = document.getElementById("input-p") as HTMLInputElement;
  pInput.min = String(2);
  pInput.value = String(initialParams.p);

  const qInput = document.getElementById("input-q") as HTMLInputElement;
  qInput.min = String(2);
  qInput.value = String(initialParams.q);

  const rInput = document.getElementById("input-r") as HTMLInputElement;
  rInput.min = String(2);
  rInput.value = String(initialParams.r);

  const modeSelect = document.getElementById(
    "select-mode"
  ) as HTMLSelectElement;
  modeSelect.appendChild(
    new Option("fundamental triangles", "triangles", true)
  );
  modeSelect.appendChild(new Option("r | p q", "r|pq"));
  modeSelect.appendChild(new Option("r p | q", "rp|q"));
  modeSelect.appendChild(new Option("p q r |", "pqr|"));

  const showTrianglesCheck = document.getElementById(
    "check-show-triangles"
  ) as HTMLInputElement;
  showTrianglesCheck.checked = true;

  const status = document.getElementById("status") as HTMLParagraphElement;

  const drawButton = document.getElementById(
    "draw-button"
  ) as HTMLButtonElement;
  drawButton.addEventListener("click", () => {
    const p = parseInt(pInput.value);
    const q = parseInt(qInput.value);
    const r = parseInt(rInput.value);
    const mode = modeSelect.value as Mode;
    const showTriangles = showTrianglesCheck.checked;

    if (q * r + p * r + p * q >= p * q * r) {
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      status.textContent = `(${p} ${q} ${r}) does not satisfy 1/p + 1/q + 1/r < 1`;
      return;
    }
    status.textContent = "";
    const data = calculateData({ p: p, q: q, r: r }, mode);
    render(data, mode, showTriangles);
  });

  const data = calculateData(initialParams, "triangles");
  render(data, "triangles", true);
}

main();
