import { computeGradient, computeLoss } from './linearModel'

// ── Types ─────────────────────────────────────────────────────────────
export type GDVariant = 'batch' | 'sgd' | 'mini-batch'

export interface GDStep {
  slope: number
  intercept: number
  loss: number
  gradient: { dSlope: number; dIntercept: number }
}

// ── Seeded random for SGD ─────────────────────────────────────────────
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

// ── Gradient computation with variant support ─────────────────────────
function computeVariantGradient(
  x: number[],
  y: number[],
  slope: number,
  intercept: number,
  variant: GDVariant,
  rng: () => number,
  batchSize = 4
): { dSlope: number; dIntercept: number } {
  if (variant === 'batch') {
    return computeGradient(x, y, slope, intercept)
  }

  const n = x.length
  let indices: number[]

  if (variant === 'sgd') {
    // Single random sample
    const idx = Math.floor(rng() * n)
    indices = [idx]
  } else {
    // Mini-batch: random subset
    indices = []
    const used = new Set<number>()
    const size = Math.min(batchSize, n)
    while (indices.length < size) {
      const idx = Math.floor(rng() * n)
      if (!used.has(idx)) {
        used.add(idx)
        indices.push(idx)
      }
    }
  }

  const subX = indices.map((i) => x[i])
  const subY = indices.map((i) => y[i])
  return computeGradient(subX, subY, slope, intercept)
}

// ── Gradient Descent optimizer with full history ──────────────────────
export function runGradientDescent(
  x: number[],
  y: number[],
  startSlope: number,
  startIntercept: number,
  learningRate: number,
  maxSteps: number,
  variant: GDVariant = 'batch',
  batchSize = 4,
  seed = 123
): GDStep[] {
  const rng = seededRandom(seed)
  const history: GDStep[] = []

  let slope = startSlope
  let intercept = startIntercept

  // Record initial state
  const initGrad = computeVariantGradient(x, y, slope, intercept, variant, rng, batchSize)
  const initLoss = computeLoss(x, y, slope, intercept, 'mse')
  history.push({ slope, intercept, loss: initLoss, gradient: initGrad })

  for (let step = 0; step < maxSteps; step++) {
    const grad = computeVariantGradient(x, y, slope, intercept, variant, rng, batchSize)
    slope = slope - learningRate * grad.dSlope
    intercept = intercept - learningRate * grad.dIntercept
    const loss = computeLoss(x, y, slope, intercept, 'mse')

    history.push({ slope, intercept, loss, gradient: grad })

    // Check convergence
    if (Math.abs(grad.dSlope) < 1e-6 && Math.abs(grad.dIntercept) < 1e-6) {
      break
    }
  }

  return history
}
