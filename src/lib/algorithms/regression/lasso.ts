import { mean } from '../../math/linalg'

export interface LassoResult {
  coefficients: number[]
  predictions: number[]
  residuals: number[]
  rSquared: number
  sse: number
  activeFeatures: number // count of non-zero coefficients (excluding intercept)
}

/**
 * Soft-thresholding operator: S(z, gamma) = sign(z) * max(|z| - gamma, 0)
 */
function softThreshold(z: number, gamma: number): number {
  if (z > gamma) return z - gamma
  if (z < -gamma) return z + gamma
  return 0
}

/**
 * Solve Lasso regression using coordinate descent.
 *
 * Minimizes: (1/2N) * ||y - Xβ||² + α * ||β||₁
 *
 * X should include an intercept column (first column all 1s).
 * The L1 penalty is NOT applied to the intercept.
 * Features are standardized internally for numerical stability.
 *
 * The coordinate descent update for feature j (j > 0) is:
 *   β_j = S(ρ_j, α) / (x_j^T x_j / N)
 * where ρ_j = (1/N) * x_j^T * (y - X_{-j} β_{-j}) is the partial residual correlation.
 */
export function solveLasso(
  X: number[][],
  y: number[],
  alpha: number,
  maxIter: number = 1000
): LassoResult {
  const n = X.length
  const p = X[0].length
  const tol = 1e-6

  // Standardize features (not intercept)
  const featureMeans: number[] = new Array(p).fill(0)
  const featureStds: number[] = new Array(p).fill(1)
  const Xs = X.map((row) => [...row])

  for (let j = 1; j < p; j++) {
    let sum = 0
    for (let i = 0; i < n; i++) sum += X[i][j]
    const mu = sum / n
    featureMeans[j] = mu

    let sumSq = 0
    for (let i = 0; i < n; i++) sumSq += (X[i][j] - mu) * (X[i][j] - mu)
    const std = Math.sqrt(sumSq / n)
    featureStds[j] = std < 1e-15 ? 1 : std

    for (let i = 0; i < n; i++) {
      Xs[i][j] = (X[i][j] - mu) / featureStds[j]
    }
  }

  // Precompute column norms squared (divided by N)
  const colNormSq: number[] = new Array(p).fill(0)
  for (let j = 0; j < p; j++) {
    let sum = 0
    for (let i = 0; i < n; i++) sum += Xs[i][j] * Xs[i][j]
    colNormSq[j] = sum / n
  }

  // Initialize coefficients
  const beta = new Array(p).fill(0)
  // Initialize intercept to mean of y
  beta[0] = mean(y)

  // Current residuals: r = y - X * beta
  const residual = new Array(n)
  for (let i = 0; i < n; i++) {
    let pred = 0
    for (let j = 0; j < p; j++) pred += Xs[i][j] * beta[j]
    residual[i] = y[i] - pred
  }

  // Coordinate descent
  for (let iter = 0; iter < maxIter; iter++) {
    let maxChange = 0

    for (let j = 0; j < p; j++) {
      const oldBeta = beta[j]

      // Compute partial residual correlation: rho_j = (1/N) * x_j^T * (r + x_j * old_beta)
      let rho = 0
      for (let i = 0; i < n; i++) {
        rho += Xs[i][j] * (residual[i] + Xs[i][j] * oldBeta)
      }
      rho /= n

      if (j === 0) {
        // No regularization on intercept
        beta[j] = rho / (colNormSq[j] + 1e-15)
      } else {
        // Apply soft-thresholding
        beta[j] = softThreshold(rho, alpha) / (colNormSq[j] + 1e-15)
      }

      // Update residuals if beta changed
      const diff = beta[j] - oldBeta
      if (Math.abs(diff) > 1e-15) {
        for (let i = 0; i < n; i++) {
          residual[i] -= Xs[i][j] * diff
        }
      }

      maxChange = Math.max(maxChange, Math.abs(diff))
    }

    // Check convergence
    if (maxChange < tol) break
  }

  // Un-standardize coefficients
  const coefficients = new Array(p)
  let interceptAdj = 0
  for (let j = 1; j < p; j++) {
    coefficients[j] = beta[j] / featureStds[j]
    interceptAdj += beta[j] * featureMeans[j] / featureStds[j]
  }
  coefficients[0] = beta[0] - interceptAdj

  // Predictions using original X
  const predictions = X.map((row) => {
    let sum = 0
    for (let j = 0; j < p; j++) sum += row[j] * coefficients[j]
    return sum
  })

  // Residuals
  const residuals = y.map((yi, i) => yi - predictions[i])

  // SSE
  const sse = residuals.reduce((acc, r) => acc + r * r, 0)

  // R-squared
  const yMean = mean(y)
  const sst = y.reduce((acc, yi) => acc + (yi - yMean) * (yi - yMean), 0)
  const rSquared = sst < 1e-15 ? 1 : 1 - sse / sst

  // Count active features (non-zero coefficients, excluding intercept)
  let activeFeatures = 0
  for (let j = 1; j < p; j++) {
    if (Math.abs(coefficients[j]) > 1e-10) activeFeatures++
  }

  return { coefficients, predictions, residuals, rSquared, sse, activeFeatures }
}

/**
 * Compute the coefficient path for Lasso regression over a range of alpha values.
 * Useful for visualizing feature selection as regularization increases.
 */
export function lassoPath(
  X: number[][],
  y: number[],
  alphas: number[]
): {
  alphas: number[]
  coeffPaths: number[][]
  trainErrors: number[]
  activeFeatureCounts: number[]
} {
  const coeffPaths: number[][] = []
  const trainErrors: number[] = []
  const activeFeatureCounts: number[] = []

  for (const alpha of alphas) {
    const result = solveLasso(X, y, alpha)
    coeffPaths.push(result.coefficients)
    trainErrors.push(result.sse / X.length) // MSE
    activeFeatureCounts.push(result.activeFeatures)
  }

  return { alphas: [...alphas], coeffPaths, trainErrors, activeFeatureCounts }
}
