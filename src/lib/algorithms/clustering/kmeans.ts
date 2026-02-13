import type { Point2D } from '../../types'
import { squaredDistance, centroid } from '../../math/linalg'
import { createRng } from '../../math/random'

export interface KMeansSnapshot {
  iteration: number
  centroids: Point2D[]
  assignments: number[]       // cluster index per point
  centroidHistory: Point2D[][] // all centroid positions so far (for trails)
  cost: number                // total within-cluster sum of squares
  converged: boolean
}

/**
 * Run K-Means clustering using Lloyd's algorithm.
 *
 * Produces one snapshot per iteration (including the initial state before
 * the first assignment pass) so that the UI can animate each step.
 *
 * @param data       Array of 2-D points to cluster
 * @param k          Number of clusters
 * @param seed       Optional PRNG seed for reproducibility (default 42)
 * @param maxIter    Maximum iterations (default 100)
 * @returns          Array of snapshots, one per iteration
 */
export function runKMeans(
  data: Point2D[],
  k: number,
  seed: number = 42,
  maxIter: number = 100
): KMeansSnapshot[] {
  const n = data.length
  if (n === 0 || k <= 0) return []

  // Clamp k to the number of data points
  const effectiveK = Math.min(k, n)

  const rng = createRng(seed)
  const snapshots: KMeansSnapshot[] = []

  // --- Initialise centroids by sampling k distinct data points ---
  const indices: number[] = []
  const usedIndices = new Set<number>()
  while (indices.length < effectiveK) {
    const idx = Math.floor(rng() * n)
    if (!usedIndices.has(idx)) {
      usedIndices.add(idx)
      indices.push(idx)
    }
  }
  let centroids: Point2D[] = indices.map((i) => ({ ...data[i] }))

  // History of centroid positions across iterations (for drawing trails)
  const centroidHistory: Point2D[][] = [centroids.map((c) => ({ ...c }))]

  // --- Initial snapshot (iteration 0): centroids placed, no assignments yet ---
  const initialAssignments = assignPoints(data, centroids)
  const initialCost = computeCost(data, centroids, initialAssignments)
  snapshots.push({
    iteration: 0,
    centroids: centroids.map((c) => ({ ...c })),
    assignments: [...initialAssignments],
    centroidHistory: centroidHistory.map((h) => h.map((p) => ({ ...p }))),
    cost: initialCost,
    converged: false,
  })

  let prevAssignments = initialAssignments

  // --- Main loop ---
  for (let iter = 1; iter <= maxIter; iter++) {
    // 1. Assign each point to its nearest centroid
    const assignments = assignPoints(data, centroids)

    // 2. Recompute centroids as the mean of assigned points
    const newCentroids: Point2D[] = []
    for (let c = 0; c < effectiveK; c++) {
      const members = data.filter((_, i) => assignments[i] === c)
      if (members.length > 0) {
        newCentroids.push(centroid(members))
      } else {
        // Keep the old centroid if a cluster becomes empty
        newCentroids.push({ ...centroids[c] })
      }
    }

    centroids = newCentroids
    centroidHistory.push(centroids.map((c) => ({ ...c })))

    // 3. Compute cost (within-cluster sum of squares)
    const cost = computeCost(data, centroids, assignments)

    // 4. Check for convergence (assignments unchanged)
    const converged = assignments.every((a, i) => a === prevAssignments[i])

    snapshots.push({
      iteration: iter,
      centroids: centroids.map((c) => ({ ...c })),
      assignments: [...assignments],
      centroidHistory: centroidHistory.map((h) => h.map((p) => ({ ...p }))),
      cost,
      converged,
    })

    if (converged) break

    prevAssignments = assignments
  }

  return snapshots
}

// ── helpers ──────────────────────────────────────────────────────────

/** Assign every point to the index of its nearest centroid. */
function assignPoints(data: Point2D[], centroids: Point2D[]): number[] {
  return data.map((p) => {
    let bestIdx = 0
    let bestDist = Infinity
    for (let c = 0; c < centroids.length; c++) {
      const d = squaredDistance(p, centroids[c])
      if (d < bestDist) {
        bestDist = d
        bestIdx = c
      }
    }
    return bestIdx
  })
}

/** Total within-cluster sum of squared distances. */
function computeCost(
  data: Point2D[],
  centroids: Point2D[],
  assignments: number[]
): number {
  let cost = 0
  for (let i = 0; i < data.length; i++) {
    cost += squaredDistance(data[i], centroids[assignments[i]])
  }
  return cost
}
