import { createRng, normalRandom } from '../math/random'

export interface ClassificationPoint {
  x: number
  y: number
  label: number
}

export interface RegressionPoint1D {
  x: number
  y: number
}

/** Axis-aligned separable data ideal for tree demos */
export function makeTreeFriendly(
  n: number,
  nSplits = 3,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []

  for (let i = 0; i < n; i++) {
    const x = rng() * 10
    const y = rng() * 10
    // Create axis-aligned regions
    let label = 0
    if (nSplits >= 1 && x > 5) label ^= 1
    if (nSplits >= 2 && y > 5) label ^= 1
    if (nSplits >= 3 && x > 7.5) label ^= 1
    if (nSplits >= 4 && y > 2.5) label ^= 1
    if (nSplits >= 5 && x > 2.5 && y > 7.5) label ^= 1
    if (nSplits >= 6 && x < 1.5) label ^= 1
    if (nSplits >= 7 && y < 1.5 && x > 3) label ^= 1
    if (nSplits >= 8 && x > 6 && y < 3) label ^= 1
    points.push({ x, y, label })
  }

  return points
}

/** Non-linear classification data with adjustable complexity */
export function makeNonLinearClassification(
  n: number,
  complexity = 2,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []

  for (let i = 0; i < n; i++) {
    const x = normalRandom(rng, 0, 2)
    const y = normalRandom(rng, 0, 2)

    let label: number
    if (complexity <= 1) {
      // Simple linear-ish boundary
      label = x + y > 0 ? 1 : 0
    } else if (complexity <= 2) {
      // Quadratic boundary
      label = x * x + y > 1 ? 1 : 0
    } else if (complexity <= 3) {
      // Circular boundary
      label = x * x + y * y > 3 ? 1 : 0
    } else {
      // XOR-like
      label = (x > 0) !== (y > 0) ? 1 : 0
    }

    points.push({ x, y, label })
  }

  return points
}

/** Clean data with noise injection for overfitting demos */
export function makeNoisyClassification(
  n: number,
  noiseRatio = 0.1,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []

  for (let i = 0; i < n; i++) {
    const x = normalRandom(rng, 0, 2)
    const y = normalRandom(rng, 0, 2)
    let label = x * x + y * y > 2.5 ? 1 : 0

    // Flip label with noiseRatio probability
    if (rng() < noiseRatio) {
      label = 1 - label
    }

    points.push({ x, y, label })
  }

  return points
}

/** 1D regression curve for GBT residual visualization */
export function makeRegressionCurve(
  n: number,
  func: 'sine' | 'quadratic' | 'step' | 'complex' = 'sine',
  noise = 0.3,
  seed = 42
): RegressionPoint1D[] {
  const rng = createRng(seed)
  const points: RegressionPoint1D[] = []

  for (let i = 0; i < n; i++) {
    const x = -3 + (6 * i) / (n - 1) + normalRandom(rng, 0, 0.05)
    let y: number

    switch (func) {
      case 'sine':
        y = Math.sin(x * 1.5) + 0.5 * Math.sin(x * 3)
        break
      case 'quadratic':
        y = 0.5 * x * x - 1
        break
      case 'step':
        y = x < -1 ? -1 : x < 1 ? 0 : 1
        break
      case 'complex':
        y = Math.sin(x * 2) * Math.exp(-0.3 * Math.abs(x)) + 0.5 * x
        break
    }

    y += normalRandom(rng, 0, noise)
    points.push({ x, y })
  }

  return points
}

/** Moon-shaped data for tree comparison */
export function makeMoonsClassification(
  n: number,
  noise = 0.15,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    const angle = (Math.PI * i) / half
    points.push({
      x: Math.cos(angle) + normalRandom(rng, 0, noise),
      y: Math.sin(angle) + normalRandom(rng, 0, noise),
      label: 0,
    })
  }

  for (let i = 0; i < n - half; i++) {
    const angle = (Math.PI * i) / (n - half)
    points.push({
      x: 1 - Math.cos(angle) + normalRandom(rng, 0, noise),
      y: 0.5 - Math.sin(angle) + normalRandom(rng, 0, noise),
      label: 1,
    })
  }

  return points
}

/** Spiral data - stress test for all models */
export function makeSpiralsClassification(
  n: number,
  noise = 0.2,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    const r = (5 * i) / half
    const angle = (1.75 * i) / half * Math.PI + Math.PI
    points.push({
      x: r * Math.sin(angle) + normalRandom(rng, 0, noise),
      y: r * Math.cos(angle) + normalRandom(rng, 0, noise),
      label: 0,
    })
  }

  for (let i = 0; i < n - half; i++) {
    const r = (5 * i) / (n - half)
    const angle = (1.75 * i) / (n - half) * Math.PI
    points.push({
      x: r * Math.sin(angle) + normalRandom(rng, 0, noise),
      y: r * Math.cos(angle) + normalRandom(rng, 0, noise),
      label: 1,
    })
  }

  return points
}
