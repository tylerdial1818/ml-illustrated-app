import { matMul, transpose, solve, mean } from '../../math/linalg'

export interface RidgeResult {
  coefficients: number[]
  predictions: number[]
  residuals: number[]
  rSquared: number
  sse: number
}

/**
 * Standardize the feature columns of X (excluding the intercept column at index 0).
 * Returns the standardized matrix along with the means and standard deviations
 * used, so coefficients can be un-standardized afterwards.
 */
function standardizeFeatures(X: number[][]): {
  Xs: number[][]
  featureMeans: number[]
  featureStds: number[]
} {
  const n = X.length
  const p = X[0].length // includes intercept at col 0
  const featureMeans: number[] = new Array(p).fill(0)
  const featureStds: number[] = new Array(p).fill(1)

  // Copy X
  const Xs = X.map((row) => [...row])

  // Standardize columns 1..p-1 (skip intercept column 0)
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

  return { Xs, featureMeans, featureStds }
}

/**
 * Convert coefficients from standardized space back to the original feature space.
 * In standardized space: y = b0s + b1s * ((x1 - mu1) / s1) + ...
 * In original space:      y = b0 + b1 * x1 + ...
 *   where b_j = b_js / s_j, and b0 = b0s - Σ(b_js * mu_j / s_j)
 */
function unstandardizeCoefficients(
  coeffs: number[],
  featureMeans: number[],
  featureStds: number[]
): number[] {
  const p = coeffs.length
  const result = new Array(p)

  // Intercept adjustment
  let interceptAdj = 0
  for (let j = 1; j < p; j++) {
    result[j] = coeffs[j] / featureStds[j]
    interceptAdj += coeffs[j] * featureMeans[j] / featureStds[j]
  }
  result[0] = coeffs[0] - interceptAdj

  return result
}

/**
 * Solve Ridge regression: β = (X^T X + αI)^(-1) X^T y
 *
 * X should include an intercept column (first column all 1s).
 * The regularization penalty is NOT applied to the intercept (column 0).
 * Features are standardized internally for numerical stability, and
 * coefficients are converted back to the original scale.
 */
export function solveRidge(X: number[][], y: number[], alpha: number): RidgeResult {
  const p = X[0].length

  // Standardize features (not the intercept)
  const { Xs, featureMeans, featureStds } = standardizeFeatures(X)

  // X^T X
  const Xt = transpose(Xs)
  const XtX = matMul(Xt, Xs)

  // Add αI to X^T X, but skip the intercept (index 0)
  for (let j = 1; j < p; j++) {
    XtX[j][j] += alpha
  }

  // X^T y
  const yCol = y.map((v) => [v])
  const XtYMat = matMul(Xt, yCol)
  const XtY = XtYMat.map((row) => row[0])

  // Solve the system
  const coeffsStd = solve(XtX, XtY)

  // Un-standardize coefficients
  const coefficients = unstandardizeCoefficients(coeffsStd, featureMeans, featureStds)

  // Predictions using original X and un-standardized coefficients
  const predictions = X.map((row) => {
    let sum = 0
    for (let j = 0; j < p; j++) {
      sum += row[j] * coefficients[j]
    }
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

  return { coefficients, predictions, residuals, rSquared, sse }
}

/**
 * Compute the coefficient path for Ridge regression over a range of alpha values.
 * Useful for visualizing how coefficients shrink as regularization increases.
 */
export function ridgePath(
  X: number[][],
  y: number[],
  alphas: number[]
): {
  alphas: number[]
  coeffPaths: number[][] // each entry is coefficients at that alpha
  trainErrors: number[]
} {
  const coeffPaths: number[][] = []
  const trainErrors: number[] = []

  for (const alpha of alphas) {
    const result = solveRidge(X, y, alpha)
    coeffPaths.push(result.coefficients)
    trainErrors.push(result.sse / X.length) // MSE
  }

  return { alphas: [...alphas], coeffPaths, trainErrors }
}
