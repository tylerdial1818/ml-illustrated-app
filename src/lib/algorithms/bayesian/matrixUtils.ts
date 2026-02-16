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

// ── Matrix operations ────────────────────────────────────────────────

export function matrixMultiply(A: number[][], B: number[][]): number[][] {
  const m = A.length
  const n = B[0].length
  const p = B.length
  const C: number[][] = Array.from({ length: m }, () => new Array(n).fill(0))
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < p; k++) {
        C[i][j] += A[i][k] * B[k][j]
      }
    }
  }
  return C
}

export function matrixTranspose(A: number[][]): number[][] {
  const m = A.length
  const n = A[0].length
  const T: number[][] = Array.from({ length: n }, () => new Array(m).fill(0))
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      T[j][i] = A[i][j]
    }
  }
  return T
}

export function matrixVectorMultiply(A: number[][], v: number[]): number[] {
  return A.map((row) => row.reduce((sum, a, j) => sum + a * v[j], 0))
}

export function matrixAdd(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((a, j) => a + B[i][j]))
}

export function matrixScale(A: number[][], s: number): number[][] {
  return A.map((row) => row.map((a) => a * s))
}

export function identityMatrix(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => (i === j ? 1 : 0))
  )
}

// ── Matrix inverse (Gauss-Jordan) ────────────────────────────────────
export function matrixInverse(A: number[][]): number[][] {
  const n = A.length
  // Augment [A | I]
  const aug: number[][] = A.map((row, i) => [
    ...row.map((v) => v),
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  ])

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxVal = Math.abs(aug[col][col])
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col])
        maxRow = row
      }
    }
    if (maxRow !== col) {
      const tmp = aug[col]
      aug[col] = aug[maxRow]
      aug[maxRow] = tmp
    }

    const pivot = aug[col][col]
    if (Math.abs(pivot) < 1e-12) {
      // Singular: return identity as fallback
      return identityMatrix(n)
    }

    // Scale pivot row
    for (let j = 0; j < 2 * n; j++) {
      aug[col][j] /= pivot
    }

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue
      const factor = aug[row][col]
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j]
      }
    }
  }

  return aug.map((row) => row.slice(n))
}

// ── Cholesky decomposition (lower triangular L such that A = LLᵀ) ───
export function choleskyDecomposition(A: number[][]): number[][] {
  const n = A.length
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k]
      }
      if (i === j) {
        const val = A[i][i] - sum
        L[i][j] = Math.sqrt(Math.max(val, 1e-10))
      } else {
        L[i][j] = L[j][j] === 0 ? 0 : (A[i][j] - sum) / L[j][j]
      }
    }
  }

  return L
}

// ── Sample from multivariate normal ──────────────────────────────────
export function sampleMultivariateNormal(
  mean: number[],
  cov: number[][],
  nSamples: number,
  seed = 42
): number[][] {
  const rng = seededRandom(seed)
  const d = mean.length
  const L = choleskyDecomposition(cov)

  const samples: number[][] = []
  for (let s = 0; s < nSamples; s++) {
    // Generate d independent standard normals
    const z: number[] = []
    for (let i = 0; i < d; i++) {
      z.push(gaussianRandom(rng))
    }
    // Transform: x = mean + L * z
    const x: number[] = []
    for (let i = 0; i < d; i++) {
      let val = mean[i]
      for (let j = 0; j <= i; j++) {
        val += L[i][j] * z[j]
      }
      x.push(val)
    }
    samples.push(x)
  }

  return samples
}

// ── 2D Gaussian PDF (for contour plots) ──────────────────────────────
export function gaussian2DPDF(
  x: number,
  y: number,
  mean: [number, number],
  cov: number[][]
): number {
  const det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0]
  if (Math.abs(det) < 1e-12) return 0

  const invDet = 1 / det
  const dx = x - mean[0]
  const dy = y - mean[1]

  const exponent =
    -0.5 *
    (invDet *
      (cov[1][1] * dx * dx - 2 * cov[0][1] * dx * dy + cov[0][0] * dy * dy))

  return (1 / (2 * Math.PI * Math.sqrt(Math.abs(det)))) * Math.exp(exponent)
}
