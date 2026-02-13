import type { Point2D } from '../../../types'
import { euclideanDistance } from '../../math/linalg'

export interface MergeStep {
  cluster1: number
  cluster2: number
  distance: number
  newClusterId: number
}

export interface AgglomerativeSnapshot {
  stepIndex: number
  assignments: number[]      // current cluster assignment per point
  numClusters: number
  mergeHistory: MergeStep[]  // all merges so far
  distances: number[]        // merge distances (for dendrogram heights)
}

export type LinkageMethod = 'single' | 'complete' | 'average' | 'ward'

/**
 * Run agglomerative (bottom-up) hierarchical clustering.
 *
 * Starts with every point as its own cluster and repeatedly merges the
 * two closest clusters until a single cluster remains.  One snapshot is
 * produced after each merge so the UI can animate the dendrogram being
 * built.
 *
 * @param data     Array of 2-D points
 * @param linkage  Inter-cluster distance criterion
 * @returns        Array of snapshots (length = n, from n clusters down to 1)
 */
export function runAgglomerative(
  data: Point2D[],
  linkage: LinkageMethod
): AgglomerativeSnapshot[] {
  const n = data.length
  if (n === 0) return []

  const snapshots: AgglomerativeSnapshot[] = []
  const mergeHistory: MergeStep[] = []
  const mergeDistances: number[] = []

  // Each point starts in its own cluster, identified by its index.
  // When two clusters merge, the new cluster gets the next available id.
  let assignments = data.map((_, i) => i)
  let nextClusterId = n

  // Set of currently active cluster ids.
  const activeClusters = new Set<number>(assignments)

  // Map from cluster id -> set of original point indices belonging to it.
  const clusterMembers = new Map<number, number[]>()
  for (let i = 0; i < n; i++) {
    clusterMembers.set(i, [i])
  }

  // For Ward's method we also track cluster centroids and sizes.
  const clusterCentroid = new Map<number, Point2D>()
  const clusterSize = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    clusterCentroid.set(i, { x: data[i].x, y: data[i].y })
    clusterSize.set(i, 1)
  }

  // --- Initial snapshot: n clusters ---
  snapshots.push({
    stepIndex: 0,
    assignments: [...assignments],
    numClusters: activeClusters.size,
    mergeHistory: [],
    distances: [],
  })

  // --- Main loop: merge until one cluster remains ---
  while (activeClusters.size > 1) {
    let bestDist = Infinity
    let bestA = -1
    let bestB = -1

    const activeArr = Array.from(activeClusters)

    // Find the pair of active clusters with the smallest linkage distance.
    for (let i = 0; i < activeArr.length; i++) {
      for (let j = i + 1; j < activeArr.length; j++) {
        const cA = activeArr[i]
        const cB = activeArr[j]
        const d = clusterDistance(
          clusterMembers.get(cA)!,
          clusterMembers.get(cB)!,
          data,
          linkage,
          clusterCentroid.get(cA)!,
          clusterCentroid.get(cB)!,
          clusterSize.get(cA)!,
          clusterSize.get(cB)!
        )
        if (d < bestDist) {
          bestDist = d
          bestA = cA
          bestB = cB
        }
      }
    }

    // Merge bestA and bestB into a new cluster.
    const newId = nextClusterId++
    const membersA = clusterMembers.get(bestA)!
    const membersB = clusterMembers.get(bestB)!
    const mergedMembers = [...membersA, ...membersB]

    clusterMembers.set(newId, mergedMembers)
    clusterMembers.delete(bestA)
    clusterMembers.delete(bestB)

    // Update centroid and size for Ward's method
    const sizeA = clusterSize.get(bestA)!
    const sizeB = clusterSize.get(bestB)!
    const centA = clusterCentroid.get(bestA)!
    const centB = clusterCentroid.get(bestB)!
    const totalSize = sizeA + sizeB
    clusterCentroid.set(newId, {
      x: (centA.x * sizeA + centB.x * sizeB) / totalSize,
      y: (centA.y * sizeA + centB.y * sizeB) / totalSize,
    })
    clusterSize.set(newId, totalSize)
    clusterCentroid.delete(bestA)
    clusterCentroid.delete(bestB)
    clusterSize.delete(bestA)
    clusterSize.delete(bestB)

    activeClusters.delete(bestA)
    activeClusters.delete(bestB)
    activeClusters.add(newId)

    // Update point-level assignments so they refer to the current active ids.
    for (const idx of mergedMembers) {
      assignments[idx] = newId
    }

    const step: MergeStep = {
      cluster1: bestA,
      cluster2: bestB,
      distance: bestDist,
      newClusterId: newId,
    }
    mergeHistory.push(step)
    mergeDistances.push(bestDist)

    snapshots.push({
      stepIndex: snapshots.length,
      assignments: [...assignments],
      numClusters: activeClusters.size,
      mergeHistory: mergeHistory.map((m) => ({ ...m })),
      distances: [...mergeDistances],
    })
  }

  return snapshots
}

// ── linkage distance functions ───────────────────────────────────────

function clusterDistance(
  membersA: number[],
  membersB: number[],
  data: Point2D[],
  linkage: LinkageMethod,
  centroidA: Point2D,
  centroidB: Point2D,
  sizeA: number,
  sizeB: number
): number {
  switch (linkage) {
    case 'single':
      return singleLinkage(membersA, membersB, data)
    case 'complete':
      return completeLinkage(membersA, membersB, data)
    case 'average':
      return averageLinkage(membersA, membersB, data)
    case 'ward':
      return wardDistance(centroidA, centroidB, sizeA, sizeB)
  }
}

/** Single linkage: minimum distance between any two points in the clusters. */
function singleLinkage(
  membersA: number[],
  membersB: number[],
  data: Point2D[]
): number {
  let min = Infinity
  for (const a of membersA) {
    for (const b of membersB) {
      const d = euclideanDistance(data[a], data[b])
      if (d < min) min = d
    }
  }
  return min
}

/** Complete linkage: maximum distance between any two points in the clusters. */
function completeLinkage(
  membersA: number[],
  membersB: number[],
  data: Point2D[]
): number {
  let max = 0
  for (const a of membersA) {
    for (const b of membersB) {
      const d = euclideanDistance(data[a], data[b])
      if (d > max) max = d
    }
  }
  return max
}

/** Average linkage: mean pairwise distance. */
function averageLinkage(
  membersA: number[],
  membersB: number[],
  data: Point2D[]
): number {
  let total = 0
  for (const a of membersA) {
    for (const b of membersB) {
      total += euclideanDistance(data[a], data[b])
    }
  }
  return total / (membersA.length * membersB.length)
}

/**
 * Ward's method: increase in total within-cluster variance caused by merging.
 *
 * The distance is defined as:
 *   d(A,B) = sqrt( (2 * nA * nB) / (nA + nB) ) * ||centroid_A - centroid_B||
 *
 * This is equivalent to the increase in total sum-of-squares when A and B
 * are merged, and produces the same dendrogram as minimising the full
 * variance increase formula.
 */
function wardDistance(
  centroidA: Point2D,
  centroidB: Point2D,
  sizeA: number,
  sizeB: number
): number {
  const dist = euclideanDistance(centroidA, centroidB)
  return Math.sqrt((2 * sizeA * sizeB) / (sizeA + sizeB)) * dist
}
