// ── Seeded random for reproducible data generation ────────────────────
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

// Box-Muller transform for normal distribution
function gaussianRandom(rng: () => number): number {
  const u1 = rng()
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// ── Data generators ───────────────────────────────────────────────────
export function makeLinearData(
  n: number,
  trueSlope: number,
  trueIntercept: number,
  noiseStd: number,
  seed = 42
): { x: number[]; y: number[] } {
  const rng = seededRandom(seed)
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const xi = rng() * 10 - 2
    const yi = trueSlope * xi + trueIntercept + gaussianRandom(rng) * noiseStd
    x.push(xi)
    y.push(yi)
  }
  return { x, y }
}

export function makeLinearDataWithOutlier(
  n: number,
  trueSlope: number,
  trueIntercept: number,
  noiseStd: number,
  outlierX: number,
  outlierY: number,
  seed = 42
): { x: number[]; y: number[] } {
  const { x, y } = makeLinearData(n, trueSlope, trueIntercept, noiseStd, seed)
  x.push(outlierX)
  y.push(outlierY)
  return { x, y }
}

// ── Loss types ────────────────────────────────────────────────────────
export type LossType = 'mse' | 'mae' | 'huber'

// ── Simple linear model: y = slope * x + intercept ────────────────────
export function predict(x: number[], slope: number, intercept: number): number[] {
  return x.map((xi) => slope * xi + intercept)
}

export function computeResiduals(
  x: number[],
  y: number[],
  slope: number,
  intercept: number
): number[] {
  return y.map((yi, i) => yi - (slope * x[i] + intercept))
}

export function computeLoss(
  x: number[],
  y: number[],
  slope: number,
  intercept: number,
  lossType: LossType,
  huberDelta = 1.0
): number {
  const n = x.length
  if (n === 0) return 0

  let sum = 0
  for (let i = 0; i < n; i++) {
    const residual = y[i] - (slope * x[i] + intercept)
    switch (lossType) {
      case 'mse':
        sum += residual * residual
        break
      case 'mae':
        sum += Math.abs(residual)
        break
      case 'huber': {
        const absR = Math.abs(residual)
        sum += absR <= huberDelta
          ? 0.5 * residual * residual
          : huberDelta * (absR - 0.5 * huberDelta)
        break
      }
    }
  }
  return sum / n
}

// ── Gradient of MSE w.r.t. slope and intercept ────────────────────────
export function computeGradient(
  x: number[],
  y: number[],
  slope: number,
  intercept: number
): { dSlope: number; dIntercept: number } {
  const n = x.length
  if (n === 0) return { dSlope: 0, dIntercept: 0 }

  let dSlope = 0
  let dIntercept = 0
  for (let i = 0; i < n; i++) {
    const residual = y[i] - (slope * x[i] + intercept)
    dSlope += -2 * residual * x[i]
    dIntercept += -2 * residual
  }
  return { dSlope: dSlope / n, dIntercept: dIntercept / n }
}

// ── Loss surface computation ──────────────────────────────────────────
export function computeLossSurface(
  x: number[],
  y: number[],
  slopeRange: [number, number],
  interceptRange: [number, number],
  resolution: number,
  lossType: LossType = 'mse'
): { slopes: number[]; intercepts: number[]; losses: number[][] } {
  const slopes: number[] = []
  const intercepts: number[] = []
  const losses: number[][] = []

  for (let i = 0; i < resolution; i++) {
    slopes.push(slopeRange[0] + (slopeRange[1] - slopeRange[0]) * (i / (resolution - 1)))
  }
  for (let j = 0; j < resolution; j++) {
    intercepts.push(
      interceptRange[0] + (interceptRange[1] - interceptRange[0]) * (j / (resolution - 1))
    )
  }

  for (let j = 0; j < resolution; j++) {
    const row: number[] = []
    for (let i = 0; i < resolution; i++) {
      row.push(computeLoss(x, y, slopes[i], intercepts[j], lossType))
    }
    losses.push(row)
  }

  return { slopes, intercepts, losses }
}

// ── OLS closed-form solution ──────────────────────────────────────────
export function olsSolution(
  x: number[],
  y: number[]
): { slope: number; intercept: number } {
  const n = x.length
  const meanX = x.reduce((a, b) => a + b, 0) / n
  const meanY = y.reduce((a, b) => a + b, 0) / n

  let num = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    num += (x[i] - meanX) * (y[i] - meanY)
    den += (x[i] - meanX) * (x[i] - meanX)
  }

  const slope = den === 0 ? 0 : num / den
  const intercept = meanY - slope * meanX
  return { slope, intercept }
}
