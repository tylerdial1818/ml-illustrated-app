// ── Kernel interface ─────────────────────────────────────────────────

export interface KernelFunction {
  name: string
  parameters: Record<string, number>
  compute(x1: number, x2: number): number
  computeMatrix(X1: number[], X2: number[]): number[][]
}

// ── RBF (Squared Exponential) ───────────────────────────────────────
// k(x, x') = σ² exp(-|x - x'|² / (2ℓ²))
// Infinitely differentiable, very smooth functions.

export class RBFKernel implements KernelFunction {
  name = 'RBF'
  lengthScale: number
  signalVariance: number

  constructor(lengthScale = 1.0, signalVariance = 1.0) {
    this.lengthScale = lengthScale
    this.signalVariance = signalVariance
  }

  get parameters() {
    return { lengthScale: this.lengthScale, signalVariance: this.signalVariance }
  }

  compute(x1: number, x2: number): number {
    const d = x1 - x2
    return this.signalVariance * Math.exp(-(d * d) / (2 * this.lengthScale * this.lengthScale))
  }

  computeMatrix(X1: number[], X2: number[]): number[][] {
    return X1.map((x1) => X2.map((x2) => this.compute(x1, x2)))
  }
}

// ── Matern 3/2 ──────────────────────────────────────────────────────
// k(x, x') = σ² (1 + √3 |x-x'|/ℓ) exp(-√3 |x-x'|/ℓ)
// Once differentiable, rougher than RBF.

export class MaternKernel implements KernelFunction {
  name = 'Matérn 3/2'
  lengthScale: number
  signalVariance: number

  constructor(lengthScale = 1.0, signalVariance = 1.0) {
    this.lengthScale = lengthScale
    this.signalVariance = signalVariance
  }

  get parameters() {
    return { lengthScale: this.lengthScale, signalVariance: this.signalVariance }
  }

  compute(x1: number, x2: number): number {
    const r = Math.abs(x1 - x2)
    const s3 = Math.sqrt(3) * r / this.lengthScale
    return this.signalVariance * (1 + s3) * Math.exp(-s3)
  }

  computeMatrix(X1: number[], X2: number[]): number[][] {
    return X1.map((x1) => X2.map((x2) => this.compute(x1, x2)))
  }
}

// ── Periodic ────────────────────────────────────────────────────────
// k(x, x') = σ² exp(-2 sin²(π|x-x'|/p) / ℓ²)
// Repeating patterns with period p.

export class PeriodicKernel implements KernelFunction {
  name = 'Periodic'
  lengthScale: number
  signalVariance: number
  period: number

  constructor(lengthScale = 1.0, signalVariance = 1.0, period = 3.0) {
    this.lengthScale = lengthScale
    this.signalVariance = signalVariance
    this.period = period
  }

  get parameters() {
    return { lengthScale: this.lengthScale, signalVariance: this.signalVariance, period: this.period }
  }

  compute(x1: number, x2: number): number {
    const d = Math.abs(x1 - x2)
    const sinTerm = Math.sin(Math.PI * d / this.period)
    return this.signalVariance * Math.exp(-2 * sinTerm * sinTerm / (this.lengthScale * this.lengthScale))
  }

  computeMatrix(X1: number[], X2: number[]): number[][] {
    return X1.map((x1) => X2.map((x2) => this.compute(x1, x2)))
  }
}

// ── Linear ──────────────────────────────────────────────────────────
// k(x, x') = σ² + σ_b² (x - c)(x' - c)
// Produces straight-line functions. Equivalent to Bayesian linear regression.

export class LinearKernel implements KernelFunction {
  name = 'Linear'
  signalVariance: number
  biasVariance: number
  center: number

  constructor(signalVariance = 0.5, biasVariance = 1.0, center = 5.0) {
    this.signalVariance = signalVariance
    this.biasVariance = biasVariance
    this.center = center
  }

  get parameters() {
    return { signalVariance: this.signalVariance, biasVariance: this.biasVariance, center: this.center }
  }

  compute(x1: number, x2: number): number {
    return this.biasVariance + this.signalVariance * (x1 - this.center) * (x2 - this.center)
  }

  computeMatrix(X1: number[], X2: number[]): number[][] {
    return X1.map((x1) => X2.map((x2) => this.compute(x1, x2)))
  }
}

// ── Kernel factory ──────────────────────────────────────────────────

export type KernelType = 'rbf' | 'matern' | 'periodic' | 'linear'

export function createKernel(type: KernelType, lengthScale = 1.0, signalVariance = 1.0): KernelFunction {
  switch (type) {
    case 'rbf':
      return new RBFKernel(lengthScale, signalVariance)
    case 'matern':
      return new MaternKernel(lengthScale, signalVariance)
    case 'periodic':
      return new PeriodicKernel(lengthScale, signalVariance)
    case 'linear':
      return new LinearKernel(signalVariance)
    default:
      return new RBFKernel(lengthScale, signalVariance)
  }
}
