import { matMul, transpose, solve, mean } from '../../math/linalg'

export interface OLSResult {
  coefficients: number[] // [intercept, slope] for simple, or [b0, b1, ...bn] for multi
  predictions: number[]
  residuals: number[]
  rSquared: number
  sse: number // sum of squared errors
}

/**
 * Solve ordinary least squares regression using the normal equations.
 *
 * X should be the design matrix WITH an intercept column already included
 * (i.e., the first column should be all 1s).
 *
 * Returns coefficients β = (X^T X)^(-1) X^T y via Gaussian elimination.
 */
export function solveOLS(X: number[][], y: number[]): OLSResult {
  const p = X[0].length

  // X^T
  const Xt = transpose(X)

  // X^T X  (p x p)
  const XtX = matMul(Xt, X)

  // X^T y  (p x 1 -> flat array)
  const yCol = y.map((v) => [v])
  const XtYMat = matMul(Xt, yCol)
  const XtY = XtYMat.map((row) => row[0])

  // Solve (X^T X) β = X^T y
  const coefficients = solve(XtX, XtY)

  // Predictions
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

  // R-squared = 1 - SSE / SST
  const yMean = mean(y)
  const sst = y.reduce((acc, yi) => acc + (yi - yMean) * (yi - yMean), 0)
  const rSquared = sst < 1e-15 ? 1 : 1 - sse / sst

  return { coefficients, predictions, residuals, rSquared, sse }
}

/**
 * Convenience wrapper for simple linear regression from {x, y} points.
 * Automatically constructs the design matrix with an intercept column.
 *
 * Returns coefficients as [intercept, slope].
 */
export function solveSimpleOLS(points: { x: number; y: number }[]): OLSResult {
  // Build design matrix [1, x_i]
  const X = points.map((p) => [1, p.x])
  const y = points.map((p) => p.y)

  return solveOLS(X, y)
}
