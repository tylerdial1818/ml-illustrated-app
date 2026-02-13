import type { Point2D } from '../../types'
import {
  identity2x2,
  det2x2,
  inv2x2,
  weightedCovariance,
  gaussian2DPDF,
  type Matrix2x2,
} from '../../math/linalg'
import { createRng } from '../../math/random'

export interface GMMSnapshot {
  iteration: number
  means: Point2D[]
  covariances: Matrix2x2[]
  weights: number[]            // mixing coefficients (pi_k)
  responsibilities: number[][] // N x K matrix of soft assignments
  logLikelihood: number
  converged: boolean
}

/**
 * Run Expectation-Maximisation for a Gaussian Mixture Model.
 *
 * Produces one snapshot per EM iteration (plus the initial state) so
 * the UI can animate the Gaussians evolving over time.
 *
 * @param data     Array of 2-D points
 * @param k        Number of Gaussian components
 * @param seed     Optional PRNG seed (default 42)
 * @param maxIter  Maximum EM iterations (default 100)
 * @returns        Array of snapshots, one per iteration
 */
export function runGMM(
  data: Point2D[],
  k: number,
  seed: number = 42,
  maxIter: number = 100
): GMMSnapshot[] {
  const n = data.length
  if (n === 0 || k <= 0) return []

  const effectiveK = Math.min(k, n)
  const rng = createRng(seed)
  const snapshots: GMMSnapshot[] = []
  const CONVERGENCE_THRESHOLD = 1e-6

  // --- Initialise parameters ---

  // Means: pick k distinct data points at random
  const chosenIndices: number[] = []
  const used = new Set<number>()
  while (chosenIndices.length < effectiveK) {
    const idx = Math.floor(rng() * n)
    if (!used.has(idx)) {
      used.add(idx)
      chosenIndices.push(idx)
    }
  }

  const means: Point2D[] = chosenIndices.map((i) => ({ ...data[i] }))
  const covariances: Matrix2x2[] = Array.from({ length: effectiveK }, () => identity2x2())
  const weights: number[] = new Array(effectiveK).fill(1 / effectiveK)

  // Responsibilities: N x K
  let responsibilities: number[][] = Array.from({ length: n }, () =>
    new Array(effectiveK).fill(0)
  )

  // Compute initial log-likelihood and responsibility from the initial params
  const initLL = computeLogLikelihoodAndResponsibilities(
    data, means, covariances, weights, responsibilities
  )

  snapshots.push(makeSnapshot(0, means, covariances, weights, responsibilities, initLL, false))

  let prevLL = initLL

  // --- EM loop ---
  for (let iter = 1; iter <= maxIter; iter++) {
    // ── E-step ──────────────────────────────────────────────────────
    const ll = computeLogLikelihoodAndResponsibilities(
      data, means, covariances, weights, responsibilities
    )

    // ── M-step ──────────────────────────────────────────────────────
    for (let c = 0; c < effectiveK; c++) {
      // Effective number of points assigned to component c
      let Nc = 0
      for (let i = 0; i < n; i++) {
        Nc += responsibilities[i][c]
      }

      if (Nc < 1e-10) {
        // Component has effectively died — reinitialise it to a random point
        const ri = Math.floor(rng() * n)
        means[c] = { ...data[ri] }
        covariances[c] = identity2x2()
        weights[c] = 1 / effectiveK
        continue
      }

      // Update mean
      let mx = 0
      let my = 0
      for (let i = 0; i < n; i++) {
        mx += responsibilities[i][c] * data[i].x
        my += responsibilities[i][c] * data[i].y
      }
      means[c] = { x: mx / Nc, y: my / Nc }

      // Update covariance using the weighted covariance helper
      const w = data.map((_, i) => responsibilities[i][c])
      covariances[c] = weightedCovariance(data, w, means[c])

      // Update mixing weight
      weights[c] = Nc / n
    }

    // Normalise weights (they should already sum to ~1, but ensure it)
    const wSum = weights.reduce((a, b) => a + b, 0)
    for (let c = 0; c < effectiveK; c++) {
      weights[c] /= wSum
    }

    // Recompute log-likelihood after the M-step for the snapshot
    const newLL = computeLogLikelihoodAndResponsibilities(
      data, means, covariances, weights, responsibilities
    )

    const converged = Math.abs(newLL - prevLL) < CONVERGENCE_THRESHOLD

    snapshots.push(makeSnapshot(iter, means, covariances, weights, responsibilities, newLL, converged))

    if (converged) break

    prevLL = newLL
  }

  return snapshots
}

// ── helpers ──────────────────────────────────────────────────────────

/**
 * Compute the log-likelihood of the data under the current model
 * and update the responsibility matrix in place (the E-step).
 *
 * Returns the log-likelihood.
 */
function computeLogLikelihoodAndResponsibilities(
  data: Point2D[],
  means: Point2D[],
  covariances: Matrix2x2[],
  weights: number[],
  responsibilities: number[][]
): number {
  const n = data.length
  const k = means.length
  let logLikelihood = 0

  // Pre-compute inverse and determinant for each component
  const covInvs: Matrix2x2[] = covariances.map((c) => inv2x2(c))
  const covDets: number[] = covariances.map((c) => det2x2(c))

  for (let i = 0; i < n; i++) {
    // Weighted probability of point i under each component
    let totalProb = 0
    for (let c = 0; c < k; c++) {
      const prob = weights[c] * gaussian2DPDF(data[i], means[c], covInvs[c], covDets[c])
      responsibilities[i][c] = prob
      totalProb += prob
    }

    // Normalise to get posterior responsibilities
    if (totalProb > 1e-300) {
      for (let c = 0; c < k; c++) {
        responsibilities[i][c] /= totalProb
      }
      logLikelihood += Math.log(totalProb)
    } else {
      // Fallback: assign equal responsibility to prevent NaN
      for (let c = 0; c < k; c++) {
        responsibilities[i][c] = 1 / k
      }
    }
  }

  return logLikelihood
}

/** Create a deep-copied snapshot. */
function makeSnapshot(
  iteration: number,
  means: Point2D[],
  covariances: Matrix2x2[],
  weights: number[],
  responsibilities: number[][],
  logLikelihood: number,
  converged: boolean
): GMMSnapshot {
  return {
    iteration,
    means: means.map((m) => ({ ...m })),
    covariances: covariances.map((c) => ({ ...c })),
    weights: [...weights],
    responsibilities: responsibilities.map((row) => [...row]),
    logLikelihood,
    converged,
  }
}
