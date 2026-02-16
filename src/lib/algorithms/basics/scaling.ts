// ── Seeded random ─────────────────────────────────────────────────────
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

function gaussianRandom(rng: () => number): number {
  const u1 = rng()
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// ── Data generators ──────────────────────────────────────────────────
export function makeMultiScaleData(
  n: number,
  seed = 42
): { squareFeet: number[]; bedrooms: number[]; price: number[] } {
  const rng = seededRandom(seed)
  const squareFeet: number[] = []
  const bedrooms: number[] = []
  const price: number[] = []

  for (let i = 0; i < n; i++) {
    const sf = 500 + rng() * 4500
    const br = Math.round(1 + rng() * 5)
    const p = 50000 + sf * 100 + br * 30000 + gaussianRandom(rng) * 20000
    squareFeet.push(sf)
    bedrooms.push(br)
    price.push(p)
  }

  return { squareFeet, bedrooms, price }
}

// ── Scaling functions ────────────────────────────────────────────────
export function minMaxScale(values: number[]): { scaled: number[]; min: number; max: number } {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  return {
    scaled: values.map((v) => (v - min) / range),
    min,
    max,
  }
}

export function standardize(values: number[]): { scaled: number[]; mean: number; std: number } {
  const n = values.length
  const mean = values.reduce((a, b) => a + b, 0) / n
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / n
  const std = Math.sqrt(variance) || 1
  return {
    scaled: values.map((v) => (v - mean) / std),
    mean,
    std,
  }
}

export function robustScale(values: number[]): { scaled: number[]; median: number; iqr: number } {
  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)]
  const q1 = sorted[Math.floor(n * 0.25)]
  const q3 = sorted[Math.floor(n * 0.75)]
  const iqr = q3 - q1 || 1
  return {
    scaled: values.map((v) => (v - median) / iqr),
    median,
    iqr,
  }
}

export type ScaleMethod = 'minmax' | 'standard' | 'robust'

export function scaleDataset2D(
  x1: number[],
  x2: number[],
  method: ScaleMethod
): { x1Scaled: number[]; x2Scaled: number[] } {
  const scaleFn =
    method === 'minmax' ? minMaxScale : method === 'standard' ? standardize : robustScale
  return {
    x1Scaled: scaleFn(x1).scaled,
    x2Scaled: scaleFn(x2).scaled,
  }
}

// ── Loss surface for 2D linear regression ────────────────────────────
// For demonstrating how contour shape changes with scaling
export function compute2DLossSurface(
  x1: number[],
  x2: number[],
  y: number[],
  w1Range: [number, number],
  w2Range: [number, number],
  resolution: number
): { w1s: number[]; w2s: number[]; losses: number[][] } {
  const w1s: number[] = []
  const w2s: number[] = []
  const losses: number[][] = []
  const n = x1.length

  for (let i = 0; i < resolution; i++) {
    w1s.push(w1Range[0] + (w1Range[1] - w1Range[0]) * (i / (resolution - 1)))
  }
  for (let j = 0; j < resolution; j++) {
    w2s.push(w2Range[0] + (w2Range[1] - w2Range[0]) * (j / (resolution - 1)))
  }

  for (let j = 0; j < resolution; j++) {
    const row: number[] = []
    for (let i = 0; i < resolution; i++) {
      let sum = 0
      for (let k = 0; k < n; k++) {
        const pred = w1s[i] * x1[k] + w2s[j] * x2[k]
        const err = y[k] - pred
        sum += err * err
      }
      row.push(sum / n)
    }
    losses.push(row)
  }

  return { w1s, w2s, losses }
}

// Simple 2D gradient descent for scaling demo
export function run2DGradientDescent(
  x1: number[],
  x2: number[],
  y: number[],
  startW1: number,
  startW2: number,
  lr: number,
  steps: number
): { w1: number; w2: number; loss: number }[] {
  const n = x1.length
  const history: { w1: number; w2: number; loss: number }[] = []
  let w1 = startW1
  let w2 = startW2

  for (let s = 0; s <= steps; s++) {
    let loss = 0
    let dw1 = 0
    let dw2 = 0
    for (let i = 0; i < n; i++) {
      const pred = w1 * x1[i] + w2 * x2[i]
      const err = y[i] - pred
      loss += err * err
      dw1 += -2 * err * x1[i]
      dw2 += -2 * err * x2[i]
    }
    loss /= n
    dw1 /= n
    dw2 /= n

    history.push({ w1, w2, loss })

    if (s < steps) {
      w1 -= lr * dw1
      w2 -= lr * dw2
    }
  }

  return history
}
