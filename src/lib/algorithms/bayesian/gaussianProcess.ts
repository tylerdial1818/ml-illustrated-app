import { choleskyDecomposition } from './matrixUtils'
import type { KernelFunction } from './kernels'

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

// ── Solve L * x = b for lower triangular L (forward substitution) ───
function solveL(L: number[][], b: number[]): number[] {
  const n = L.length
  const x = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    let sum = b[i]
    for (let j = 0; j < i; j++) {
      sum -= L[i][j] * x[j]
    }
    x[i] = sum / L[i][i]
  }
  return x
}

// ── Solve Lᵀ * x = b for lower triangular L (back substitution) ────
function solveLT(L: number[][], b: number[]): number[] {
  const n = L.length
  const x = new Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i]
    for (let j = i + 1; j < n; j++) {
      sum -= L[j][i] * x[j]
    }
    x[i] = sum / L[i][i]
  }
  return x
}

// ── Gaussian Process Regression ─────────────────────────────────────

export class GaussianProcess {
  kernel: KernelFunction
  noiseVariance: number
  trainingX: number[]
  trainingY: number[]

  // Cached Cholesky factor of (K + σₙ²I)
  private L: number[][] | null = null
  private alpha: number[] | null = null

  constructor(kernel: KernelFunction, noiseVariance = 0.1) {
    this.kernel = kernel
    this.noiseVariance = noiseVariance
    this.trainingX = []
    this.trainingY = []
  }

  fit(X: number[], y: number[]): void {
    this.trainingX = [...X]
    this.trainingY = [...y]
    this.L = null
    this.alpha = null

    if (X.length === 0) return

    const n = X.length
    // K(X, X) + σₙ²I
    const K = this.kernel.computeMatrix(X, X)
    for (let i = 0; i < n; i++) {
      K[i][i] += this.noiseVariance + 1e-8 // jitter for numerical stability
    }

    // Cholesky: K = LLᵀ
    this.L = choleskyDecomposition(K)

    // α = L^T \ (L \ y)
    const Ly = solveL(this.L, y)
    this.alpha = solveLT(this.L, Ly)
  }

  predict(xNew: number[]): { mean: number[]; variance: number[] } {
    if (this.trainingX.length === 0 || !this.L || !this.alpha) {
      // No training data: return prior mean=0, variance=kernel(x,x)
      const mean = xNew.map(() => 0)
      const variance = xNew.map((x) => this.kernel.compute(x, x) + this.noiseVariance)
      return { mean, variance }
    }

    const Kstar = this.kernel.computeMatrix(xNew, this.trainingX) // n* × n
    const mean: number[] = []
    const variance: number[] = []

    for (let i = 0; i < xNew.length; i++) {
      // μ* = k*ᵀ α
      let mu = 0
      for (let j = 0; j < this.trainingX.length; j++) {
        mu += Kstar[i][j] * this.alpha![j]
      }
      mean.push(mu)

      // v = L \ k*
      const v = solveL(this.L!, Kstar[i])

      // σ²* = k(x*, x*) - v·v + σₙ²
      const kss = this.kernel.compute(xNew[i], xNew[i])
      let vdot = 0
      for (let j = 0; j < v.length; j++) {
        vdot += v[j] * v[j]
      }
      variance.push(Math.max(kss - vdot + this.noiseVariance, 1e-8))
    }

    return { mean, variance }
  }

  getCredibleBand(
    xRange: number[],
    level = 0.95
  ): { lower: number[]; upper: number[]; mean: number[] } {
    const z = level === 0.95 ? 1.96 : level === 0.99 ? 2.576 : 1.645
    const { mean, variance } = this.predict(xRange)
    const lower = mean.map((m, i) => m - z * Math.sqrt(variance[i]))
    const upper = mean.map((m, i) => m + z * Math.sqrt(variance[i]))
    return { lower, upper, mean }
  }

  // Sample functions from the prior (no data)
  samplePrior(xRange: number[], nSamples: number, seed = 42): number[][] {
    const n = xRange.length
    const K = this.kernel.computeMatrix(xRange, xRange)
    // Add jitter
    for (let i = 0; i < n; i++) K[i][i] += 1e-6

    const L = choleskyDecomposition(K)
    const rng = seededRandom(seed)
    const samples: number[][] = []

    for (let s = 0; s < nSamples; s++) {
      const z: number[] = []
      for (let i = 0; i < n; i++) z.push(gaussianRandom(rng))
      // f = L * z (mean is zero for prior)
      const f: number[] = []
      for (let i = 0; i < n; i++) {
        let val = 0
        for (let j = 0; j <= i; j++) {
          val += L[i][j] * z[j]
        }
        f.push(val)
      }
      samples.push(f)
    }

    return samples
  }

  // Sample functions from the posterior
  samplePosterior(xRange: number[], nSamples: number, seed = 42): number[][] {
    if (this.trainingX.length === 0) return this.samplePrior(xRange, nSamples, seed)

    const { mean, variance } = this.predict(xRange)
    const n = xRange.length

    // Build posterior covariance at xRange
    // Full cov: K** - K*n (Knn)^-1 Kn*
    const Kss = this.kernel.computeMatrix(xRange, xRange)
    const Kns = this.kernel.computeMatrix(xRange, this.trainingX)

    // V = L \ Kns^T (each column is L \ k*_i)
    // Cov = Kss - V^T V
    const posteriorCov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
    for (let i = 0; i < n; i++) {
      const vi = solveL(this.L!, Kns[i])
      for (let j = i; j < n; j++) {
        const vj = solveL(this.L!, Kns[j])
        let vdot = 0
        for (let k = 0; k < vi.length; k++) vdot += vi[k] * vj[k]
        posteriorCov[i][j] = Kss[i][j] - vdot
        posteriorCov[j][i] = posteriorCov[i][j]
      }
      // Ensure positive diagonal
      posteriorCov[i][i] = Math.max(posteriorCov[i][i], 1e-8)
    }

    // Cholesky of posterior covariance
    const Lpost = choleskyDecomposition(posteriorCov)
    const rng = seededRandom(seed)
    const samples: number[][] = []

    for (let s = 0; s < nSamples; s++) {
      const z: number[] = []
      for (let i = 0; i < n; i++) z.push(gaussianRandom(rng))
      // f = mean + Lpost * z
      const f: number[] = []
      for (let i = 0; i < n; i++) {
        let val = mean[i]
        for (let j = 0; j <= i; j++) {
          val += Lpost[i][j] * z[j]
        }
        f.push(val)
      }
      samples.push(f)
    }

    // Clamp variance for stability
    for (let s = 0; s < nSamples; s++) {
      for (let i = 0; i < n; i++) {
        if (!isFinite(samples[s][i])) samples[s][i] = mean[i]
      }
    }

    return samples
  }

  setKernel(kernel: KernelFunction): void {
    this.kernel = kernel
    // Refit if we have data
    if (this.trainingX.length > 0) {
      this.fit(this.trainingX, this.trainingY)
    }
  }

  setNoise(noiseVariance: number): void {
    this.noiseVariance = noiseVariance
    if (this.trainingX.length > 0) {
      this.fit(this.trainingX, this.trainingY)
    }
  }

  reset(): void {
    this.trainingX = []
    this.trainingY = []
    this.L = null
    this.alpha = null
  }
}

// ── Data generators ─────────────────────────────────────────────────

export function makeSineData(
  n: number,
  noise: number,
  seed = 42
): { x: number[]; y: number[] } {
  const rng = seededRandom(seed)
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const xi = rng() * 10
    x.push(xi)
    y.push(Math.sin(xi) + gaussianRandom(rng) * noise)
  }
  return { x, y }
}

export function makeGapData(
  n: number,
  gapStart = 3,
  gapEnd = 7,
  seed = 42
): { x: number[]; y: number[] } {
  const rng = seededRandom(seed)
  const x: number[] = []
  const y: number[] = []
  let placed = 0
  while (placed < n) {
    const xi = rng() * 10
    if (xi >= gapStart && xi <= gapEnd) continue
    x.push(xi)
    y.push(Math.sin(xi) + gaussianRandom(rng) * 0.2)
    placed++
  }
  return { x, y }
}

export function makeLinearTrendData(
  n: number,
  noise: number,
  seed = 42
): { x: number[]; y: number[] } {
  const rng = seededRandom(seed)
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const xi = rng() * 10
    x.push(xi)
    y.push(0.5 * xi - 1 + gaussianRandom(rng) * noise)
  }
  return { x, y }
}
