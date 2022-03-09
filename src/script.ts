const RADIUS = 200;
const CANVAS_SIZE = 2 * RADIUS;

type CanvasCoord = { cx: number; cy: number };

type C = { readonly re: number; readonly im: number };
type C2 = [C, C]; // Element of C^2, or CP^1.
type Mat = [C, C, C, C]; // Row-major 2x2 matrix.

function fromPolar(mod: number, arg: number): C {
  return { re: mod * Math.cos(arg), im: mod * Math.sin(arg) };
}

function fromReal(x: number): C {
  return { re: x, im: 0 };
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

function applyMoebius(m: Mat, p: C2): C2 {
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

function toC(p: C2): C {
  return div(p[0], p[1]);
}

// Each pixel corresponds to its midpoint.
function complex(c: CanvasCoord): C {
  return { re: (c.cx + 0.5) / RADIUS - 1, im: 1 - (c.cy + 0.5) / RADIUS };
}

type Params = { p: number; q: number; r: number };

function draw(canvasCtx: CanvasRenderingContext2D, params: Params): void {
  const matA: Mat = [fromReal(1), fromReal(0), fromReal(0), fromReal(1)];
  const matB: Mat = [
    fromPolar(1, Math.PI / params.p),
    fromReal(0),
    fromReal(0),
    fromPolar(1, -Math.PI / params.p),
  ];
  const alpha = Math.PI / params.q;
  const beta = Math.PI / params.p;
  const k =
    ((Math.sin(alpha) * Math.cos(alpha + beta)) / Math.sin(beta) + 1) /
    Math.sin(alpha + beta);
  const radiusSq = 1 / ((k + 1) * (k - 1));
  const center = k * Math.pow(radiusSq, 0.5);
  const matC: Mat = [
    fromReal(center),
    fromReal(-1),
    fromReal(1),
    fromReal(-center),
  ];

  function insideA(z: C): boolean {
    return z.im >= 0;
  }

  function insideB(z: C): boolean {
    return arg(z) <= beta && arg(z) >= -Math.PI + beta;
  }

  function insideC(z: C): boolean {
    return modSq(sub(z, { re: center, im: 0 })) >= radiusSq;
  }

  const imgData = new ImageData(CANVAS_SIZE, CANVAS_SIZE);
  const data = imgData.data;

  for (let cy = 0; cy < CANVAS_SIZE; cy++) {
    for (let cx = 0; cx < CANVAS_SIZE; cx++) {
      let z = complex({ cx: cx, cy: cy });
      if (z.re ** 2 + z.im ** 2 > 1) {
        continue;
      }

      let insA = insideA(z);
      let insB = insideB(z);
      let insC = insideC(z);
      let p: C2 = [z, { re: 1, im: 0 }];
      let numRefls = 0;

      while (!insA || !insB || !insC) {
        if (!insA) {
          p = applyAntiMoebius(matA, p);
        } else if (!insB) {
          p = applyAntiMoebius(matB, p);
        } else {
          p = applyAntiMoebius(matC, p);
        }
        z = toC(p);
        insA = insideA(z);
        insB = insideB(z);
        insC = insideC(z);
        numRefls += 1;
      }

      const odd = numRefls % 2 == 1;

      data[(cy * CANVAS_SIZE + cx) * 4 + 0] = odd ? 255 : 0;
      data[(cy * CANVAS_SIZE + cx) * 4 + 1] = odd ? 255 : 0;
      data[(cy * CANVAS_SIZE + cx) * 4 + 2] = odd ? 255 : 0;
      data[(cy * CANVAS_SIZE + cx) * 4 + 3] = 255;
    }
  }

  canvasCtx.putImageData(imgData, 0, 0);
  return;
}

function main(): void {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;
  canvas.style.width = "400px";
  canvas.style.height = "400px";
  const canvasCtx = canvas.getContext("2d") as CanvasRenderingContext2D;

  const initialParams = { p: 5, q: 5, r: 2 };

  const pInput = document.getElementById("input-p") as HTMLInputElement;
  pInput.min = String(2);
  pInput.value = String(initialParams.p);

  const qInput = document.getElementById("input-q") as HTMLInputElement;
  qInput.min = String(2);
  qInput.value = String(initialParams.q);

  const rInput = document.getElementById("input-r") as HTMLInputElement;
  rInput.min = String(2);
  rInput.value = String(initialParams.r);
  rInput.disabled = true;

  const status = document.getElementById("status") as HTMLParagraphElement;

  const drawButton = document.getElementById(
    "draw-button"
  ) as HTMLButtonElement;
  drawButton.addEventListener("click", () => {
    const p = parseInt(pInput.value);
    const q = parseInt(qInput.value);
    const r = parseInt(rInput.value);

    if (q * r + p * r + p * q >= p * q * r) {
      status.textContent = `(${p} ${q} ${r}) does not satisfy 1/p + 1/q + 1/r < 1`;
      canvasCtx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      return;
    }
    status.textContent = "";
    draw(canvasCtx, { p: p, q: q, r: r });
  });

  draw(canvasCtx, initialParams);
}

main();
