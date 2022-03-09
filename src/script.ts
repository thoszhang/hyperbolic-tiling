const RADIUS = 200;
const CANVAS_SIZE = 2 * RADIUS;

function draw(
  canvasCtx: CanvasRenderingContext2D,
  p: number,
  q: number,
  r: number
): void {
  const imgData = new ImageData(CANVAS_SIZE, CANVAS_SIZE);
  for (let i = 0; i < CANVAS_SIZE * CANVAS_SIZE * 4; i += 4) {
    imgData.data[i] = 255;
    imgData.data[i + 1] = 0;
    imgData.data[i + 2] = 0;
    imgData.data[i + 3] = 255;
  }

  canvasCtx.putImageData(imgData, 0, 0);
  return;
}

function main(): void {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;
  const canvasCtx = canvas.getContext("2d") as CanvasRenderingContext2D;

  const pInput = document.getElementById("input-p") as HTMLInputElement;
  pInput.value = String(5);
  pInput.min = String(3);

  const qInput = document.getElementById("input-q") as HTMLInputElement;
  qInput.value = String(5);
  qInput.min = String(3);

  const rInput = document.getElementById("input-r") as HTMLInputElement;
  rInput.value = String(2);
  rInput.disabled = true;

  const status = document.getElementById("status") as HTMLParagraphElement;

  const drawButton = document.getElementById(
    "draw-button"
  ) as HTMLButtonElement;
  drawButton.addEventListener("click", () => {
    const p = parseInt(pInput.value);
    const q = parseInt(qInput.value);
    const r = parseInt(rInput.value);

    if ((p - 2) * (q - 2) <= 4) {
      status.textContent = `{${p}, ${q}, ${r}} is not a valid hyperbolic tiling`;
      return;
    }
    status.textContent = "";
    draw(canvasCtx, p, q, r);
  });
}

main();
