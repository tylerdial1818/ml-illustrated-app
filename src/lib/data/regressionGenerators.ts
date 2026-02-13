import { createRng, normalRandom } from '../math/random'

export interface RegressionPoint {
  x: number
  y: number
  features?: number[]
}

export interface ClassificationPoint {
  x: number
  y: number
  label: number
}

export function makeLinear(
  n: number,
  slope = 2,
  intercept = 1,
  noise = 1,
  seed = 42
): RegressionPoint[] {
  const rng = createRng(seed)
  const points: RegressionPoint[] = []

  for (let i = 0; i < n; i++) {
    const x = -3 + (6 * i) / (n - 1) + normalRandom(rng, 0, 0.1)
    const y = slope * x + intercept + normalRandom(rng, 0, noise)
    points.push({ x, y })
  }

  return points
}

export function makePolynomial(
  n: number,
  degree = 3,
  noise = 0.5,
  seed = 42
): RegressionPoint[] {
  const rng = createRng(seed)
  const points: RegressionPoint[] = []
  const coeffs = Array.from({ length: degree + 1 }, () => normalRandom(rng, 0, 1))

  for (let i = 0; i < n; i++) {
    const x = -2 + (4 * i) / (n - 1)
    let y = 0
    for (let d = 0; d <= degree; d++) {
      y += coeffs[d] * Math.pow(x, d)
    }
    y += normalRandom(rng, 0, noise)
    points.push({ x, y })
  }

  return points
}

export function makeMultiFeature(
  n: number,
  nFeatures: number,
  nRelevant: number,
  noise = 0.5,
  seed = 42
): { X: number[][]; y: number[]; trueCoeffs: number[] } {
  const rng = createRng(seed)

  // True coefficients: first nRelevant are non-zero
  const trueCoeffs = Array.from({ length: nFeatures }, (_, i) =>
    i < nRelevant ? normalRandom(rng, 0, 2) : 0
  )

  const X: number[][] = []
  const y: number[] = []

  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: nFeatures }, () => normalRandom(rng, 0, 1))
    X.push(row)

    let target = 0
    for (let j = 0; j < nFeatures; j++) {
      target += trueCoeffs[j] * row[j]
    }
    target += normalRandom(rng, 0, noise)
    y.push(target)
  }

  return { X, y, trueCoeffs }
}

export function makeClassification(
  n: number,
  separation = 2,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    points.push({
      x: normalRandom(rng, -separation / 2, 1),
      y: normalRandom(rng, -separation / 4, 1),
      label: 0,
    })
  }

  for (let i = 0; i < n - half; i++) {
    points.push({
      x: normalRandom(rng, separation / 2, 1),
      y: normalRandom(rng, separation / 4, 1),
      label: 1,
    })
  }

  return points
}
