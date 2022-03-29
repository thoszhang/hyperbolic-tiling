import { mat2, mat4, ReadonlyMat4 } from "gl-matrix";

const RADIUS = 800;
const CANVAS_SIZE = 2 * RADIUS;

type C = mat2;
type C2 = mat4; // only use the first 2 columns
type Mat = mat4;

const zero = fromReal(0);
const id: Mat = mat4.create();

function toC(re: number, im: number): C {
  // prettier-ignore
  return mat2.fromValues(
    re, im,
    -im, re
  );
}

function fromCol(v: C, w: C): Mat {
  // prettier-ignore
  return mat4.fromValues(
    w[0], -w[1], v[0], -v[1],
    -w[2], w[3], -v[2], v[3],
    v[0], v[1], w[0], w[1],
    v[2], v[3], w[2], w[3]
  );
}

function rot(angle: number): Mat {
  return fromCol(zero, fromPolar(1, -angle / 2));
}

function transl(dist: number): Mat {
  return fromCol(fromReal(Math.sinh(dist / 2)), fromReal(Math.cosh(dist / 2)));
}

function refl(line: Mat): Mat {
  const out = mat4.create();
  conjMat(out, line);
  mat4.adjoint(out, out);
  mat4.mul(out, line, out);
  return out;
}

function reflRot(angle: number): Mat {
  return fromCol(zero, fromPolar(1, -angle));
}

function conjReflRot(angle: number): Mat {
  return fromCol(zero, fromPolar(1, angle));
}

function conjMat(out: mat4, m: ReadonlyMat4) {
  if (out !== m) {
    mat4.copy(out, m);
  }
  out[1] = -out[1];
  out[3] = -out[3];
  out[4] = -out[4];
  out[6] = -out[6];
  out[9] = -out[9];
  out[11] = -out[11];
  out[12] = -out[12];
  out[14] = -out[14];
}

function mulMat(...ms: Mat[]): Mat {
  const result = mat4.create();
  ms.forEach((m) => {
    mat4.mul(result, result, m);
  });
  return result;
}

function fromPolar(mod: number, arg: number): C {
  return toC(mod * Math.cos(arg), mod * Math.sin(arg));
}

function fromReal(x: number): C {
  return toC(x, 0);
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
  const vRe = line[8];
  const vIm = line[9];
  const wRe = line[10];
  const wIm = line[11];
  const roots = quadraticRoots(
    -vRe * wIm + vIm * wRe,
    2 * (vRe * vIm - wRe * wIm),
    -vRe * wIm + vIm * wRe
  );
  if (!roots) {
    return undefined;
  }
  const intersections = roots.map((r) =>
    Math.sqrt(
      mat2.determinant(dehomogenize(mulMat(line, homogenize(fromReal(r)))))
    )
  );
  if (intersections[1] > intersections[0]) {
    return [intersections[0], intersections[1]];
  } else {
    return [intersections[1], intersections[0]];
  }
}

function homogenize(z: C): C2 {
  // prettier-ignore
  return mat4.fromValues(
    z[0], z[1], 1, 0,
    z[2], z[3], 0, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
  );
}

function dehomogenize(p: C2): C {
  const z1 = mat2.fromValues(p[0], p[1], p[4], p[5]);
  const z2 = mat2.fromValues(p[2], p[3], p[6], p[7]);
  mat2.invert(z2, z2);
  mat2.mul(z2, z1, z2);
  return z2;
}

type Params = { p: number; q: number; r: number };
type Mode = "triangles" | "p|qr" | "q|pr" | "r|pq" | "rp|q" | "pqr|";

function modeIndex(m: Mode): number {
  switch (m) {
    case "triangles":
      return 0;
    case "p|qr":
    case "q|pr":
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
  t: undefined | x_yzData | xy_zData | pqr_Data;
};
type x_yzData = {
  invHalfPl1: Mat;
};
type xy_zData = {
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
  const coshA = (cos_a + cos_b * cos_c) / (sin_b * sin_c);
  const C = Math.acosh(coshC);
  const B = Math.acosh(coshB);
  const A = Math.acosh(coshA);

  const halfPlaneC: Mat = id;
  const halfPlaneB: Mat = rot(Math.PI + a);
  const halfPlaneA: Mat = mulMat(transl(C), rot(Math.PI - b));

  const matC: Mat = refl(halfPlaneC);
  const matB: Mat = refl(halfPlaneB);
  const matA: Mat = refl(halfPlaneA);

  function p_qr(): x_yzData {
    const halfPlane: Mat = mulMat(
      reflRot(a / 2),
      transl(B),
      conjReflRot(-c / 2),
      transl(-A + Math.atanh(cos_b * Math.sqrt(1 - Math.pow(coshC, -2)))),
      rot(Math.PI / 2)
    );

    mat4.adjoint(halfPlane, halfPlane);
    return {
      invHalfPl1: halfPlane,
    };
  }

  function q_pr(): x_yzData {
    const halfPlane: Mat = mulMat(
      transl(C),
      reflRot(-b / 2),
      transl(-A),
      conjReflRot(c / 2),

      transl(Math.atanh(cos_c * Math.sqrt(1 - Math.pow(coshA, -2)))),
      rot(Math.PI / 2)
    );

    mat4.adjoint(halfPlane, halfPlane);
    return {
      invHalfPl1: halfPlane,
    };
  }

  function r_pq(): x_yzData {
    const halfPlane: Mat = mulMat(
      transl(Math.atanh(cos_a * Math.sqrt(1 - Math.pow(coshB, -2)))),
      rot(Math.PI / 2)
    );

    mat4.adjoint(halfPlane, halfPlane);
    return {
      invHalfPl1: halfPlane,
    };
  }

  function rp_q(): xy_zData {
    const euclDistToPoint = (
      realLineIntxns(mulMat(rot(-a / 2), halfPlaneA)) as [number, number]
    )[0];

    const halfPlane1: Mat = mulMat(
      transl(
        Math.atanh(
          Math.cos(a / 2) * 2 * (euclDistToPoint / (1 + euclDistToPoint ** 2))
        )
      ),
      rot(Math.PI / 2)
    );
    const halfPlane2: Mat = mulMat(reflRot(a / 2), halfPlane1);

    mat4.adjoint(halfPlane1, halfPlane1);
    mat4.adjoint(halfPlane2, halfPlane2);
    return {
      invHalfPl1: halfPlane1,
      invHalfPl2: halfPlane2,
    };
  }

  function pqr_(): pqr_Data {
    const euclDistToPoint = (
      realLineIntxns(mulMat(rot(-a / 2), transl(C), rot(-b / 2))) as [
        number,
        number
      ]
    )[0];

    const halfPlane1: Mat = mulMat(
      transl(
        Math.atanh(
          Math.cos(a / 2) * 2 * (euclDistToPoint / (1 + euclDistToPoint ** 2))
        )
      ),
      rot(Math.PI / 2)
    );
    const halfPlane1Conj = mat4.clone(halfPlane1);
    conjMat(halfPlane1Conj, halfPlane1Conj);
    const halfPlane2: Mat = mulMat(reflRot(a / 2), halfPlane1Conj);
    const halfPlane3: Mat = mulMat(
      transl(C),
      reflRot(-b / 2),
      transl(-C),
      halfPlane1Conj
    );

    mat4.adjoint(halfPlane1, halfPlane1);
    mat4.adjoint(halfPlane2, halfPlane2);
    mat4.adjoint(halfPlane3, halfPlane3);
    return {
      invHalfPl1: halfPlane1,
      invHalfPl2: halfPlane2,
      invHalfPl3: halfPlane3,
    };
  }

  let tilingData: undefined | x_yzData | xy_zData | pqr_Data;
  switch (mode) {
    case "triangles":
      tilingData = undefined;
      break;
    case "p|qr":
      tilingData = p_qr();
      break;
    case "q|pr":
      tilingData = q_pr();
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

  mat4.adjoint(halfPlaneA, halfPlaneA);
  mat4.adjoint(halfPlaneB, halfPlaneB);
  mat4.adjoint(halfPlaneC, halfPlaneC);
  return {
    matA: matA,
    matB: matB,
    matC: matC,
    invHalfPlA: halfPlaneA,
    invHalfPlB: halfPlaneB,
    invHalfPlC: halfPlaneC,
    t: tilingData,
  };
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

int x_yzRegion(mat2x4 p) {
  if (inHalfPlane(u_invHalfPl1, p)) {
    return 0;
  }
  return 1;
}

int xy_zRegion(mat2x4 p) {
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
      region = x_yzRegion(p);
    } else if (u_mode == 2) {
      region = xy_zRegion(p);
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
    gl.uniformMatrix4fv(matALocation, false, data.matA);
    gl.uniformMatrix4fv(matBLocation, false, data.matB);
    gl.uniformMatrix4fv(matCLocation, false, data.matC);
    gl.uniformMatrix4fv(invHalfPlALocation, false, data.invHalfPlA);
    gl.uniformMatrix4fv(invHalfPlBLocation, false, data.invHalfPlB);
    gl.uniformMatrix4fv(invHalfPlCLocation, false, data.invHalfPlC);
    if (data.t && "invHalfPl1" in data.t) {
      gl.uniformMatrix4fv(invHalfPl1Location, false, data.t.invHalfPl1);
    }
    if (data.t && "invHalfPl2" in data.t) {
      gl.uniformMatrix4fv(invHalfPl2Location, false, data.t.invHalfPl2);
    }
    if (data.t && "invHalfPl3" in data.t) {
      gl.uniformMatrix4fv(invHalfPl3Location, false, data.t.invHalfPl3);
    }
    gl.uniform1i(modeLocation, modeIndex(mode));
    gl.uniform1i(showTrianglesLocation, showTriangles ? 1 : 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  const initialParams = { p: 4, q: 3, r: 3 };

  const angleInputs: { [key: string]: HTMLElement } = {};
  (["p", "q", "r"] as const).forEach((label) => {
    const input = document.getElementById(`input-${label}`) as HTMLInputElement;
    angleInputs[`input-${label}`] = input;

    const inc = document.getElementById(
      `increment-${label}`
    ) as HTMLButtonElement;
    angleInputs[`increment-${label}`] = inc;

    const dec = document.getElementById(
      `decrement-${label}`
    ) as HTMLButtonElement;
    angleInputs[`decrement-${label}`] = dec;

    inc.addEventListener("click", () => {
      input.value = String(parseInt(input.value) + 1);
      dec.disabled = parseInt(input.value) <= 2;
    });
    dec.addEventListener("click", () => {
      input.value = String(parseInt(input.value) - 1);
      dec.disabled = parseInt(input.value) <= 2;
    });
    input.addEventListener("change", () => {
      dec.disabled = parseInt(input.value) <= 2;
    });

    input.min = String(2);
    input.value = String(initialParams[label]);

    input.addEventListener("change", update);
    inc.addEventListener("click", update);
    dec.addEventListener("click", update);
  });

  const modeSelect = document.getElementById(
    "select-mode"
  ) as HTMLSelectElement;
  modeSelect.appendChild(
    new Option("fundamental triangles", "triangles", true)
  );
  modeSelect.appendChild(new Option("p | q r", "p|qr"));
  modeSelect.appendChild(new Option("q | p r", "q|pr"));
  modeSelect.appendChild(new Option("r | p q", "r|pq"));
  modeSelect.appendChild(new Option("r p | q", "rp|q"));
  modeSelect.appendChild(new Option("p q r |", "pqr|"));

  const showTrianglesCheck = document.getElementById(
    "check-show-triangles"
  ) as HTMLInputElement;
  showTrianglesCheck.checked = true;

  const status = document.getElementById("status") as HTMLParagraphElement;

  function update(): void {
    const p = parseInt((angleInputs["input-p"] as HTMLInputElement).value);
    const q = parseInt((angleInputs["input-q"] as HTMLInputElement).value);
    const r = parseInt((angleInputs["input-r"] as HTMLInputElement).value);
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
  }

  modeSelect.addEventListener("change", update);
  showTrianglesCheck.addEventListener("change", update);

  update();
}

main();
