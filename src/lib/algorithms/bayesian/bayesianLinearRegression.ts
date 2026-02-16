import {
  matrixMultiply,
  matrixTranspose,
  matrixInverse,
  matrixAdd,
  matrixScale,
  matrixVectorMultiply,
  identityMatrix,
  sampleMultivariateNormal,
  gaussian2DPDF,
} from './matrixUtils'

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

// ── Data generator ───────────────────────────────────────────────────
export function makeBayesianLinearData(
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
    const xi = rng() * 10 - 3
    const yi = trueSlope * xi + trueIntercept + gaussianRandom(rng) * noiseStd
    x.push(xi)
    y.push(yi)
  }
  return { x, y }
}

// ── Contour data type ────────────────────────────────────────────────
export interface ContourData {
  xGrid: number[]
  yGrid: number[]
  density: number[][]
}

// ── Bayesian Linear Regression (analytic posterior) ───────────────────
// Model: y = β₀ + β₁x + ε, ε ~ N(0, σ²)
// Parameters: β = [β₀, β₁]ᵀ  (intercept, slope)
// Prior: β ~ N(μ₀, Σ₀)
// Posterior: β|D ~ N(μₙ, Σₙ)

export class BayesianLinearRegression {
  priorMean: number[]
  priorCov: number[][]
  posteriorMean: number[]
  posteriorCov: number[][]
  noiseVariance: number
  hasData: boolean

  constructor(
    priorMean: number[] = [0, 0],
    priorCov: number[][] = [
      [10, 0],
      [0, 10],
    ],
    noiseVariance = 1.0
  ) {
    this.priorMean = priorMean
    this.priorCov = priorCov
    this.posteriorMean = [...priorMean]
    this.posteriorCov = priorCov.map((row) => [...row])
    this.noiseVariance = noiseVariance
    this.hasData = false
  }

  // Build design matrix X from raw x values: each row is [1, xᵢ]
  private buildDesignMatrix(x: number[]): number[][] {
    return x.map((xi) => [1, xi])
  }

  // Fit the analytic posterior given data
  fit(x: number[], y: number[]): void {
    if (x.length === 0) {
      this.posteriorMean = [...this.priorMean]
      this.posteriorCov = this.priorCov.map((row) => [...row])
      this.hasData = false
      return
    }

    this.hasData = true
    const X = this.buildDesignMatrix(x)
    const Xt = matrixTranspose(X)

    // Σ₀⁻¹
    const priorCovInv = matrixInverse(this.priorCov)

    // σ⁻² XᵀX
    const XtX = matrixMultiply(Xt, X)
    const scaledXtX = matrixScale(XtX, 1 / this.noiseVariance)

    // Σₙ = (Σ₀⁻¹ + σ⁻² XᵀX)⁻¹
    const precisionSum = matrixAdd(priorCovInv, scaledXtX)
    this.posteriorCov = matrixInverse(precisionSum)

    // Xᵀy
    const Xty: number[] = [0, 0]
    for (let i = 0; i < x.length; i++) {
      Xty[0] += y[i]
      Xty[1] += x[i] * y[i]
    }

    // μₙ = Σₙ(Σ₀⁻¹μ₀ + σ⁻² Xᵀy)
    const term1 = matrixVectorMultiply(priorCovInv, this.priorMean)
    const term2 = Xty.map((v) => v / this.noiseVariance)
    const combined = term1.map((v, i) => v + term2[i])
    this.posteriorMean = matrixVectorMultiply(this.posteriorCov, combined)
  }

  // Predict at a new x value
  predict(xNew: number): { mean: number; variance: number } {
    const phi = [1, xNew] // design vector
    const mean =
      this.posteriorMean[0] * phi[0] + this.posteriorMean[1] * phi[1]

    // Predictive variance: φᵀΣₙφ + σ²
    let paramVar = 0
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        paramVar += phi[i] * this.posteriorCov[i][j] * phi[j]
      }
    }
    const variance = paramVar + this.noiseVariance

    return { mean, variance }
  }

  // Sample weight vectors from the posterior
  samplePosteriorWeights(nSamples: number, seed = 42): number[][] {
    return sampleMultivariateNormal(
      this.posteriorMean,
      this.posteriorCov,
      nSamples,
      seed
    )
  }

  // Credible band over a range of x values
  getCredibleBand(
    xRange: number[],
    level = 0.95
  ): { lower: number[]; upper: number[]; mean: number[] } {
    const z = level === 0.95 ? 1.96 : level === 0.99 ? 2.576 : 1.645
    const lower: number[] = []
    const upper: number[] = []
    const meanLine: number[] = []

    for (const xv of xRange) {
      const pred = this.predict(xv)
      const halfWidth = z * Math.sqrt(pred.variance)
      meanLine.push(pred.mean)
      lower.push(pred.mean - halfWidth)
      upper.push(pred.mean + halfWidth)
    }

    return { lower, upper, mean: meanLine }
  }

  // 2D density grid for parameter space contour plot
  getPriorContours(
    slopeRange: [number, number],
    interceptRange: [number, number],
    resolution: number
  ): ContourData {
    return this.computeContours(
      this.priorMean,
      this.priorCov,
      slopeRange,
      interceptRange,
      resolution
    )
  }

  getPosteriorContours(
    slopeRange: [number, number],
    interceptRange: [number, number],
    resolution: number
  ): ContourData {
    return this.computeContours(
      this.posteriorMean,
      this.posteriorCov,
      slopeRange,
      interceptRange,
      resolution
    )
  }

  private computeContours(
    mean: number[],
    cov: number[][],
    slopeRange: [number, number],
    interceptRange: [number, number],
    resolution: number
  ): ContourData {
    const xGrid: number[] = []
    const yGrid: number[] = []
    const density: number[][] = []

    for (let i = 0; i < resolution; i++) {
      xGrid.push(
        slopeRange[0] +
          (slopeRange[1] - slopeRange[0]) * (i / (resolution - 1))
      )
    }
    for (let j = 0; j < resolution; j++) {
      yGrid.push(
        interceptRange[0] +
          (interceptRange[1] - interceptRange[0]) * (j / (resolution - 1))
      )
    }

    // Note: mean is [intercept, slope] but we display slope on x, intercept on y
    const mean2d: [number, number] = [mean[1], mean[0]]
    const cov2d: number[][] = [
      [cov[1][1], cov[1][0]],
      [cov[0][1], cov[0][0]],
    ]

    for (let j = 0; j < resolution; j++) {
      const row: number[] = []
      for (let i = 0; i < resolution; i++) {
        row.push(gaussian2DPDF(xGrid[i], yGrid[j], mean2d, cov2d))
      }
      density.push(row)
    }

    return { xGrid, yGrid, density }
  }

  // OLS estimate for comparison (MAP with flat prior)
  static olsEstimate(x: number[], y: number[]): { intercept: number; slope: number } {
    const n = x.length
    if (n === 0) return { intercept: 0, slope: 0 }
    const mx = x.reduce((a, b) => a + b, 0) / n
    const my = y.reduce((a, b) => a + b, 0) / n
    let num = 0
    let den = 0
    for (let i = 0; i < n; i++) {
      num += (x[i] - mx) * (y[i] - my)
      den += (x[i] - mx) * (x[i] - mx)
    }
    const slope = den === 0 ? 0 : num / den
    const intercept = my - slope * mx
    return { intercept, slope }
  }

  // Set a new prior and recompute posterior
  setPrior(mean: number[], cov: number[][]): void {
    this.priorMean = mean
    this.priorCov = cov
  }

  reset(): void {
    this.posteriorMean = [...this.priorMean]
    this.posteriorCov = this.priorCov.map((row) => [...row])
    this.hasData = false
  }
}
