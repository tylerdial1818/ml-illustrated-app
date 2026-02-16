// ── Seeded random ─────────────────────────────────────────────────────
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

// ── Beta function utilities ──────────────────────────────────────────
// Log-gamma via Stirling's approximation (sufficient for visualization)
function logGamma(z: number): number {
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z)
  }
  z -= 1
  const g = 7
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  let x = c[0]
  for (let i = 1; i < g + 2; i++) {
    x += c[i] / (z + i)
  }
  const t = z + g + 0.5
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
}

function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b)
}

// ── Beta PDF ─────────────────────────────────────────────────────────
export function betaPDF(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1) return 0
  const logPdf = (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logBeta(alpha, beta)
  return Math.exp(logPdf)
}

// ── Beta mean ────────────────────────────────────────────────────────
export function betaMean(alpha: number, beta: number): number {
  return alpha / (alpha + beta)
}

// ── Beta mode ────────────────────────────────────────────────────────
export function betaMode(alpha: number, beta: number): number {
  if (alpha <= 1 && beta <= 1) return 0.5
  if (alpha <= 1) return 0
  if (beta <= 1) return 1
  return (alpha - 1) / (alpha + beta - 2)
}

// ── Beta credible interval (percentile-based) ────────────────────────
// Uses bisection on the incomplete beta function approximation
export function betaCredibleInterval(
  alpha: number,
  beta: number,
  level: number
): [number, number] {
  const tail = (1 - level) / 2
  return [betaQuantile(alpha, beta, tail), betaQuantile(alpha, beta, 1 - tail)]
}

// Approximate quantile via bisection on the CDF
function betaQuantile(alpha: number, beta: number, p: number): number {
  let lo = 0
  let hi = 1
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2
    if (betaCDF(mid, alpha, beta) < p) {
      lo = mid
    } else {
      hi = mid
    }
  }
  return (lo + hi) / 2
}

// Approximate CDF via numerical integration (trapezoidal)
function betaCDF(x: number, alpha: number, beta: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  const n = 200
  const dx = x / n
  let sum = 0
  for (let i = 0; i <= n; i++) {
    const xi = i * dx
    const fi = betaPDF(xi, alpha, beta)
    sum += fi * (i === 0 || i === n ? 0.5 : 1)
  }
  return sum * dx
}

// ── Beta-Binomial model ──────────────────────────────────────────────
export class BetaBinomial {
  priorAlpha: number
  priorBeta: number
  alpha: number
  beta: number
  totalHeads: number
  totalTails: number

  constructor(priorAlpha = 1, priorBeta = 1) {
    this.priorAlpha = priorAlpha
    this.priorBeta = priorBeta
    this.alpha = priorAlpha
    this.beta = priorBeta
    this.totalHeads = 0
    this.totalTails = 0
  }

  observe(heads: number, tails: number): void {
    this.totalHeads += heads
    this.totalTails += tails
    this.alpha = this.priorAlpha + this.totalHeads
    this.beta = this.priorBeta + this.totalTails
  }

  getPosteriorPDF(resolution = 200): { x: number[]; y: number[] } {
    const x: number[] = []
    const y: number[] = []
    for (let i = 0; i <= resolution; i++) {
      const xi = i / resolution
      x.push(xi)
      y.push(betaPDF(xi, this.alpha, this.beta))
    }
    return { x, y }
  }

  getPriorPDF(resolution = 200): { x: number[]; y: number[] } {
    const x: number[] = []
    const y: number[] = []
    for (let i = 0; i <= resolution; i++) {
      const xi = i / resolution
      x.push(xi)
      y.push(betaPDF(xi, this.priorAlpha, this.priorBeta))
    }
    return { x, y }
  }

  getLikelihoodPDF(resolution = 200): { x: number[]; y: number[] } {
    const x: number[] = []
    const y: number[] = []
    const h = this.totalHeads
    const t = this.totalTails

    if (h === 0 && t === 0) {
      // No data: flat likelihood
      for (let i = 0; i <= resolution; i++) {
        x.push(i / resolution)
        y.push(1)
      }
      return { x, y }
    }

    // Likelihood: θ^h * (1-θ)^t (unnormalized)
    // Normalize for display
    let maxVal = 0
    const rawY: number[] = []
    for (let i = 0; i <= resolution; i++) {
      const xi = i / resolution
      x.push(xi)
      let val = 0
      if (xi > 0 && xi < 1) {
        val = Math.exp(h * Math.log(xi) + t * Math.log(1 - xi))
      } else if (xi === 0 && h === 0) {
        val = 1
      } else if (xi === 1 && t === 0) {
        val = 1
      }
      rawY.push(val)
      if (val > maxVal) maxVal = val
    }

    // Normalize so max = peak of posterior (for visual comparison)
    const posteriorPeak = Math.max(...this.getPosteriorPDF(resolution).y)
    const scale = maxVal > 0 ? posteriorPeak / maxVal : 1
    for (const v of rawY) {
      y.push(v * scale)
    }

    return { x, y }
  }

  getCredibleInterval(level = 0.95): [number, number] {
    return betaCredibleInterval(this.alpha, this.beta, level)
  }

  getPosteriorMean(): number {
    return betaMean(this.alpha, this.beta)
  }

  reset(): void {
    this.alpha = this.priorAlpha
    this.beta = this.priorBeta
    this.totalHeads = 0
    this.totalTails = 0
  }

  setPrior(alpha: number, beta: number): void {
    this.priorAlpha = alpha
    this.priorBeta = beta
    this.alpha = alpha + this.totalHeads
    this.beta = beta + this.totalTails
  }
}

// ── Coin flip simulator ──────────────────────────────────────────────
export function simulateCoinFlips(
  n: number,
  trueBias: number,
  seed = 42
): { outcomes: ('H' | 'T')[]; cumulativeHeads: number[]; cumulativeTails: number[] } {
  const rng = seededRandom(seed)
  const outcomes: ('H' | 'T')[] = []
  const cumulativeHeads: number[] = []
  const cumulativeTails: number[] = []
  let h = 0
  let t = 0

  for (let i = 0; i < n; i++) {
    if (rng() < trueBias) {
      outcomes.push('H')
      h++
    } else {
      outcomes.push('T')
      t++
    }
    cumulativeHeads.push(h)
    cumulativeTails.push(t)
  }

  return { outcomes, cumulativeHeads, cumulativeTails }
}
