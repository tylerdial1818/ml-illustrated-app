import type { Point2D } from '../../types'

export function euclideanDistance(a: Point2D, b: Point2D): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return Math.sqrt(dx * dx + dy * dy)
}

export function squaredDistance(a: Point2D, b: Point2D): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return dx * dx + dy * dy
}

export function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length
}

export function centroid(points: Point2D[]): Point2D {
  if (points.length === 0) return { x: 0, y: 0 }
  return {
    x: mean(points.map((p) => p.x)),
    y: mean(points.map((p) => p.y)),
  }
}

// 2x2 matrix operations for GMM
export interface Matrix2x2 {
  a: number; b: number
  c: number; d: number
}

export function identity2x2(): Matrix2x2 {
  return { a: 1, b: 0, c: 0, d: 1 }
}

export function det2x2(m: Matrix2x2): number {
  return m.a * m.d - m.b * m.c
}

export function inv2x2(m: Matrix2x2): Matrix2x2 {
  const d = det2x2(m)
  if (Math.abs(d) < 1e-10) return identity2x2()
  return {
    a: m.d / d, b: -m.b / d,
    c: -m.c / d, d: m.a / d,
  }
}

// Compute covariance matrix for a set of 2D points with weights
export function weightedCovariance(
  points: Point2D[],
  weights: number[],
  meanPt: Point2D
): Matrix2x2 {
  let totalWeight = 0
  let sxx = 0, sxy = 0, syy = 0

  for (let i = 0; i < points.length; i++) {
    const w = weights[i]
    const dx = points[i].x - meanPt.x
    const dy = points[i].y - meanPt.y
    sxx += w * dx * dx
    sxy += w * dx * dy
    syy += w * dy * dy
    totalWeight += w
  }

  if (totalWeight < 1e-10) return identity2x2()

  return {
    a: sxx / totalWeight + 1e-6,
    b: sxy / totalWeight,
    c: sxy / totalWeight,
    d: syy / totalWeight + 1e-6,
  }
}

// Eigenvalues and eigenvectors of 2x2 symmetric matrix (for ellipse rendering)
export function eigen2x2Symmetric(m: Matrix2x2): {
  values: [number, number]
  vectors: [[number, number], [number, number]]
  angle: number
} {
  const trace = m.a + m.d
  const det = det2x2(m)
  const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det))
  const lambda1 = trace / 2 + disc
  const lambda2 = trace / 2 - disc

  let angle = 0
  if (Math.abs(m.b) > 1e-10) {
    angle = Math.atan2(lambda1 - m.a, m.b)
  } else if (m.a < m.d) {
    angle = Math.PI / 2
  }

  return {
    values: [Math.max(lambda1, 1e-6), Math.max(lambda2, 1e-6)],
    vectors: [
      [Math.cos(angle), Math.sin(angle)],
      [-Math.sin(angle), Math.cos(angle)],
    ],
    angle: (angle * 180) / Math.PI,
  }
}

// Gaussian PDF for 2D point given mean and covariance
export function gaussian2DPDF(
  point: Point2D,
  mu: Point2D,
  covInv: Matrix2x2,
  covDet: number
): number {
  const dx = point.x - mu.x
  const dy = point.y - mu.y
  const exponent = -0.5 * (covInv.a * dx * dx + (covInv.b + covInv.c) * dx * dy + covInv.d * dy * dy)
  const norm = 1 / (2 * Math.PI * Math.sqrt(Math.max(covDet, 1e-10)))
  return norm * Math.exp(exponent)
}

// Matrix multiply for regression: X^T * X, X^T * y, etc.
export function matMul(A: number[][], B: number[][]): number[][] {
  const rows = A.length
  const cols = B[0].length
  const inner = B.length
  const result: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0))
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < inner; k++) {
        result[i][j] += A[i][k] * B[k][j]
      }
    }
  }
  return result
}

export function transpose(A: number[][]): number[][] {
  const rows = A.length
  const cols = A[0].length
  const result: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0))
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = A[i][j]
    }
  }
  return result
}

// Solve Ax = b for small matrices using Gaussian elimination with partial pivoting
export function solve(A: number[][], b: number[]): number[] {
  const n = A.length
  const aug: number[][] = A.map((row, i) => [...row, b[i]])

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]]

    if (Math.abs(aug[col][col]) < 1e-12) continue

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col]
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j]
      }
    }
  }

  // Back substitution
  const x = new Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i][n]
    for (let j = i + 1; j < n; j++) {
      sum -= aug[i][j] * x[j]
    }
    x[i] = Math.abs(aug[i][i]) < 1e-12 ? 0 : sum / aug[i][i]
  }

  return x
}
