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

// ── Data generators ──────────────────────────────────────────────────
export function makeTiltedCloud(
  n: number,
  angle: number,
  spreadMajor: number,
  spreadMinor: number,
  seed = 42
): { x: number[]; y: number[] } {
  const rng = seededRandom(seed)
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  const x: number[] = []
  const y: number[] = []

  for (let i = 0; i < n; i++) {
    const major = gaussianRandom(rng) * spreadMajor
    const minor = gaussianRandom(rng) * spreadMinor
    x.push(cos * major - sin * minor)
    y.push(sin * major + cos * minor)
  }

  return { x, y }
}

export function makeCircularCloud(
  n: number,
  spread: number,
  seed = 42
): { x: number[]; y: number[] } {
  const rng = seededRandom(seed)
  const x: number[] = []
  const y: number[] = []

  for (let i = 0; i < n; i++) {
    x.push(gaussianRandom(rng) * spread)
    y.push(gaussianRandom(rng) * spread)
  }

  return { x, y }
}

export function makeThreeClusters(
  n: number,
  seed = 42
): { x: number[]; y: number[] } {
  const rng = seededRandom(seed)
  const centers = [
    { cx: -3, cy: 2 },
    { cx: 2, cy: -1 },
    { cx: 4, cy: 3 },
  ]
  const x: number[] = []
  const y: number[] = []

  for (let i = 0; i < n; i++) {
    const c = centers[i % 3]
    x.push(c.cx + gaussianRandom(rng) * 0.8)
    y.push(c.cy + gaussianRandom(rng) * 0.8)
  }

  return { x, y }
}

// ── PCA 2D ───────────────────────────────────────────────────────────
export interface PCAResult {
  mean: [number, number]
  components: [
    { direction: [number, number]; eigenvalue: number; varianceExplained: number },
    { direction: [number, number]; eigenvalue: number; varianceExplained: number },
  ]
  totalVariance: number
}

export function fitPCA(x: number[], y: number[]): PCAResult {
  const n = x.length

  // Compute mean
  const meanX = x.reduce((a, b) => a + b, 0) / n
  const meanY = y.reduce((a, b) => a + b, 0) / n

  // Centered data
  const cx = x.map((v) => v - meanX)
  const cy = y.map((v) => v - meanY)

  // Covariance matrix [cxx cxy; cxy cyy]
  let cxx = 0, cxy = 0, cyy = 0
  for (let i = 0; i < n; i++) {
    cxx += cx[i] * cx[i]
    cxy += cx[i] * cy[i]
    cyy += cy[i] * cy[i]
  }
  cxx /= n
  cxy /= n
  cyy /= n

  // Eigenvalues of 2x2 symmetric matrix via quadratic formula
  // λ² - (cxx + cyy)λ + (cxx*cyy - cxy²) = 0
  const trace = cxx + cyy
  const det = cxx * cyy - cxy * cxy
  const disc = Math.sqrt(Math.max(trace * trace - 4 * det, 0))
  const lambda1 = (trace + disc) / 2
  const lambda2 = (trace - disc) / 2

  // Eigenvectors
  function eigenvector(lambda: number): [number, number] {
    // (cxx - λ)v1 + cxy*v2 = 0
    if (Math.abs(cxy) > 1e-10) {
      const v: [number, number] = [cxy, lambda - cxx]
      const norm = Math.sqrt(v[0] * v[0] + v[1] * v[1])
      return [v[0] / norm, v[1] / norm]
    }
    // If cxy ≈ 0, eigenvectors are axis-aligned
    if (cxx >= cyy) {
      return lambda === lambda1 ? [1, 0] : [0, 1]
    }
    return lambda === lambda1 ? [0, 1] : [1, 0]
  }

  const ev1 = eigenvector(lambda1)
  const ev2 = eigenvector(lambda2)
  const totalVariance = lambda1 + lambda2 || 1

  return {
    mean: [meanX, meanY],
    components: [
      { direction: ev1, eigenvalue: lambda1, varianceExplained: lambda1 / totalVariance },
      { direction: ev2, eigenvalue: lambda2, varianceExplained: lambda2 / totalVariance },
    ],
    totalVariance,
  }
}

// Project data onto principal components
export function projectPCA(
  x: number[],
  y: number[],
  pca: PCAResult,
  nComponents: 1 | 2 = 1
): number[][] {
  const [mx, my] = pca.mean
  return x.map((xi, i) => {
    const cx = xi - mx
    const cy = y[i] - my
    const scores: number[] = []
    for (let c = 0; c < nComponents; c++) {
      const [dx, dy] = pca.components[c].direction
      scores.push(cx * dx + cy * dy)
    }
    return scores
  })
}

// Reconstruct 2D points from projections
export function reconstructPCA(
  projected: number[][],
  pca: PCAResult
): { x: number[]; y: number[] } {
  const [mx, my] = pca.mean
  const rx: number[] = []
  const ry: number[] = []

  for (const scores of projected) {
    let x = mx
    let y = my
    for (let c = 0; c < scores.length; c++) {
      const [dx, dy] = pca.components[c].direction
      x += scores[c] * dx
      y += scores[c] * dy
    }
    rx.push(x)
    ry.push(y)
  }

  return { x: rx, y: ry }
}

// Get projection lines (from original to projected position on PC1)
export function getProjectionLines(
  x: number[],
  y: number[],
  pca: PCAResult
): { fromX: number; fromY: number; toX: number; toY: number }[] {
  const [mx, my] = pca.mean
  const [dx, dy] = pca.components[0].direction

  return x.map((xi, i) => {
    const cx = xi - mx
    const cy = y[i] - my
    const score = cx * dx + cy * dy
    return {
      fromX: xi,
      fromY: y[i],
      toX: mx + score * dx,
      toY: my + score * dy,
    }
  })
}

// Compute variance captured at a given rotation angle (for the rotation demo)
export function varianceAtAngle(
  x: number[],
  y: number[],
  angle: number
): number {
  const n = x.length
  const meanX = x.reduce((a, b) => a + b, 0) / n
  const meanY = y.reduce((a, b) => a + b, 0) / n

  const dx = Math.cos(angle)
  const dy = Math.sin(angle)

  let variance = 0
  for (let i = 0; i < n; i++) {
    const score = (x[i] - meanX) * dx + (y[i] - meanY) * dy
    variance += score * score
  }

  return variance / n
}
