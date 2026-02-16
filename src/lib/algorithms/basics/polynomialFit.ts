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
export function makeNoisyCurve(
  n: number,
  trueFn: (x: number) => number,
  noiseStd: number,
  xRange: [number, number],
  seed = 42
): { x: number[]; y: number[] } {
  const rng = seededRandom(seed)
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const xi = xRange[0] + rng() * (xRange[1] - xRange[0])
    const yi = trueFn(xi) + gaussianRandom(rng) * noiseStd
    x.push(xi)
    y.push(yi)
  }
  return { x, y }
}

// ── Polynomial fit via normal equations ──────────────────────────────
// Solves X^T X β = X^T y where X is the Vandermonde matrix

export function fitPolynomial(
  x: number[],
  y: number[],
  degree: number
): number[] {
  const n = x.length
  const p = degree + 1

  // Build Vandermonde matrix X (n × p)
  const X: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < p; j++) {
      row.push(Math.pow(x[i], j))
    }
    X.push(row)
  }

  // Compute X^T X (p × p)
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0))
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let sum = 0
      for (let k = 0; k < n; k++) {
        sum += X[k][i] * X[k][j]
      }
      XtX[i][j] = sum
    }
  }

  // Compute X^T y (p × 1)
  const XtY: number[] = new Array(p).fill(0)
  for (let i = 0; i < p; i++) {
    let sum = 0
    for (let k = 0; k < n; k++) {
      sum += X[k][i] * y[k]
    }
    XtY[i] = sum
  }

  // Add small ridge regularization for numerical stability
  const lambda = 1e-8
  for (let i = 0; i < p; i++) {
    XtX[i][i] += lambda
  }

  // Solve via Gaussian elimination with partial pivoting
  return solveLinearSystem(XtX, XtY)
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length
  // Augmented matrix
  const aug: number[][] = A.map((row, i) => [...row, b[i]])

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(aug[col][col])
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col])
        maxRow = row
      }
    }

    // Swap rows
    if (maxRow !== col) {
      const tmp = aug[col]
      aug[col] = aug[maxRow]
      aug[maxRow] = tmp
    }

    // Eliminate below
    const pivot = aug[col][col]
    if (Math.abs(pivot) < 1e-12) continue

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / pivot
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

export function predictPolynomial(x: number[], coefficients: number[]): number[] {
  return x.map((xi) => {
    let y = 0
    for (let j = 0; j < coefficients.length; j++) {
      y += coefficients[j] * Math.pow(xi, j)
    }
    return y
  })
}

export function computeMSE(yTrue: number[], yPred: number[]): number {
  const n = yTrue.length
  if (n === 0) return 0
  let sum = 0
  for (let i = 0; i < n; i++) {
    const d = yTrue[i] - yPred[i]
    sum += d * d
  }
  return sum / n
}

// Compute train and test loss for all degrees 1..maxDegree
export function computeLossCurves(
  trainX: number[],
  trainY: number[],
  testX: number[],
  testY: number[],
  maxDegree: number
): { degree: number; trainLoss: number; testLoss: number }[] {
  const results: { degree: number; trainLoss: number; testLoss: number }[] = []
  for (let d = 1; d <= maxDegree; d++) {
    const coeffs = fitPolynomial(trainX, trainY, d)
    const trainPred = predictPolynomial(trainX, coeffs)
    const testPred = predictPolynomial(testX, coeffs)
    results.push({
      degree: d,
      trainLoss: computeMSE(trainY, trainPred),
      testLoss: computeMSE(testY, testPred),
    })
  }
  return results
}

// Split data into train/test
export function trainTestSplit(
  x: number[],
  y: number[],
  trainRatio: number,
  seed = 42
): { trainX: number[]; trainY: number[]; testX: number[]; testY: number[] } {
  const rng = seededRandom(seed)
  const n = x.length
  const indices = Array.from({ length: n }, (_, i) => i)

  // Fisher-Yates shuffle
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp
  }

  const nTrain = Math.round(n * trainRatio)
  const trainX: number[] = []
  const trainY: number[] = []
  const testX: number[] = []
  const testY: number[] = []

  for (let i = 0; i < n; i++) {
    const idx = indices[i]
    if (i < nTrain) {
      trainX.push(x[idx])
      trainY.push(y[idx])
    } else {
      testX.push(x[idx])
      testY.push(y[idx])
    }
  }

  return { trainX, trainY, testX, testY }
}
