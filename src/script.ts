const RADIUS = 400;
const CANVAS_SIZE = 2 * RADIUS;

type CanvasCoord = { cx: number; cy: number };

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

function translationEuclidean(distE: number): Mat {
  return fromCol(
    fromReal(distE * Math.pow((1 + distE) * (1 - distE), 0.5)),
    fromReal(Math.pow((1 + distE) * (1 - distE), 0.5))
  );
}

function invDet1(m: Mat): Mat {
  return [m[3], neg(m[1]), neg(m[2]), m[0]];
}

function refl(line: Mat): Mat {
  return mulMat(line, invDet1(conjMat(line)));
}

function inHalfPlaneFn(plane: Mat): (p: C2) => boolean {
  const invPlane = invDet1(plane);
  return function (p: C2): boolean {
    return dehomogenize(apply(invPlane, p)).im >= 0;
  };
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

function arg(z: C): number {
  return Math.atan2(z.im, z.re);
}

function add(z1: C, z2: C): C {
  return { re: z1.re + z2.re, im: z1.im + z2.im };
}

function sub(z1: C, z2: C): C {
  return { re: z1.re - z2.re, im: z1.im - z2.im };
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

function applyAntiMoebius(m: Mat, p: C2): C2 {
  const conjP0 = conj(p[0]);
  const conjP1 = conj(p[1]);
  return [
    add(mul(m[0], conjP0), mul(m[1], conjP1)),
    add(mul(m[2], conjP0), mul(m[3], conjP1)),
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

// Each pixel corresponds to its midpoint.
function complex(c: CanvasCoord): C {
  return { re: (c.cx + 0.5) / RADIUS - 1, im: 1 - (c.cy + 0.5) / RADIUS };
}

type Params = { p: number; q: number; r: number };
type Mode = "triangles" | "r|pq" | "rp|q" | "pqr|";

function draw(
  canvasCtx: CanvasRenderingContext2D,
  params: Params,
  mode: Mode,
  showTriangles: boolean
): boolean {
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

  const insideC = inHalfPlaneFn(halfPlaneC);
  const insideB = inHalfPlaneFn(halfPlaneB);
  const insideA = inHalfPlaneFn(halfPlaneA);

  function r_pq(): (p: C2) => 0 | 1 {
    const halfPlane: Mat = mulMat(
      translation(Math.atanh(cos_a * Math.sqrt(1 - Math.pow(coshB, -2)))),
      rotation(Math.PI / 2)
    );
    const insideHalfPlane = inHalfPlaneFn(halfPlane);

    return function (p: C2) {
      return insideHalfPlane(p) ? 0 : 1;
    };
  }

  function rp_q(): (p: C2) => 0 | 1 | 2 {
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

    const insideHalfPlane1 = inHalfPlaneFn(halfPlane1);
    const insideHalfPlane2 = inHalfPlaneFn(halfPlane2);

    return function (p: C2): 0 | 1 | 2 {
      if (!insideHalfPlane1(p)) {
        return 1;
      } else if (!insideHalfPlane2(p)) {
        return 2;
      } else {
        return 0;
      }
    };
  }

  function pqr_(): (p: C2) => 0 | 1 | 2 {
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

    const insideHalfPlane1 = inHalfPlaneFn(halfPlane1);
    const insideHalfPlane2 = inHalfPlaneFn(halfPlane2);
    const insideHalfPlane3 = inHalfPlaneFn(halfPlane3);

    return function (p: C2): 0 | 1 | 2 {
      if (insideHalfPlane2(p) && !insideHalfPlane3(p)) {
        return 1;
      } else if (insideHalfPlane3(p) && !insideHalfPlane1(p)) {
        return 2;
      } else {
        return 0;
      }
    };
  }

  let drawFn: (p: C2) => 0 | 1 | 2;
  switch (mode) {
    case "triangles":
      drawFn = () => 0;
      break;
    case "r|pq":
      drawFn = r_pq();
      break;
    case "rp|q":
      drawFn = rp_q();
      break;
    case "pqr|":
      drawFn = pqr_();
      break;
  }

  const imgData = new ImageData(CANVAS_SIZE, CANVAS_SIZE);
  const data = imgData.data;

  let reachedIterationLimit = false;

  for (let cy = 0; cy < CANVAS_SIZE; cy++) {
    for (let cx = 0; cx < CANVAS_SIZE; cx++) {
      const z = complex({ cx: cx, cy: cy });
      if (z.re ** 2 + z.im ** 2 > 1) {
        continue;
      }

      let p: C2 = homogenize(z);
      let insA = insideA(p);
      let insB = insideB(p);
      let insC = insideC(p);
      let numRefls = 0;

      while ((!insA || !insB || !insC) && numRefls < 50) {
        if (!insA) {
          p = applyAntiMoebius(matA, p);
        } else if (!insB) {
          p = applyAntiMoebius(matB, p);
        } else {
          p = applyAntiMoebius(matC, p);
        }
        insA = insideA(p);
        insB = insideB(p);
        insC = insideC(p);
        numRefls += 1;
      }

      if (!insA || !insB || !insC) {
        data[(cy * CANVAS_SIZE + cx) * 4 + 0] = 255;
        data[(cy * CANVAS_SIZE + cx) * 4 + 1] = 0;
        data[(cy * CANVAS_SIZE + cx) * 4 + 2] = 0;
        data[(cy * CANVAS_SIZE + cx) * 4 + 3] = 255;
        reachedIterationLimit = true;
        continue;
      }

      const triangleRegion = numRefls % 2;
      let val: number;
      if (mode === "triangles") {
        val = 255 * triangleRegion;
      } else {
        val = 64 * (1 + drawFn(p));
        if (showTriangles) {
          val += 16 * (2 * triangleRegion - 1);
        }
      }

      data[(cy * CANVAS_SIZE + cx) * 4 + 0] = val;
      data[(cy * CANVAS_SIZE + cx) * 4 + 1] = val;
      data[(cy * CANVAS_SIZE + cx) * 4 + 2] = val;
      data[(cy * CANVAS_SIZE + cx) * 4 + 3] = 255;
    }
  }

  canvasCtx.putImageData(imgData, 0, 0);
  return reachedIterationLimit;
}

function main(): void {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;
  const canvasCtx = canvas.getContext("2d") as CanvasRenderingContext2D;

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
      status.textContent = `(${p} ${q} ${r}) does not satisfy 1/p + 1/q + 1/r < 1`;
      canvasCtx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      return;
    }
    status.textContent = "";
    const reachedIterationLimit = draw(
      canvasCtx,
      { p: p, q: q, r: r },
      mode,
      showTriangles
    );
    if (reachedIterationLimit) {
      console.warn(
        "some pixels reached iteration limit due to numerical stability bugs"
      );
    }
  });

  draw(canvasCtx, initialParams, "triangles", true);
}

main();
