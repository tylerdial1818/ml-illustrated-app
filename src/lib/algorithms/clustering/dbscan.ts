import type { Point2D } from '../../../types'
import { euclideanDistance } from '../../math/linalg'

export type PointClassification = 'unvisited' | 'core' | 'border' | 'noise'

export interface DBSCANSnapshot {
  stepIndex: number
  currentPointIndex: number
  classifications: PointClassification[]
  clusterAssignments: number[]  // -1 for noise/unvisited
  neighborhoodHighlight: { center: number; neighbors: number[] } | null
  currentClusterId: number
  phase: 'classifying' | 'expanding' | 'done'
}

/**
 * Run the DBSCAN density-based clustering algorithm.
 *
 * Produces fine-grained snapshots at each interesting step so the UI can
 * animate point classification and cluster expansion.
 *
 * @param data     Array of 2-D points
 * @param epsilon  Neighbourhood radius (ε)
 * @param minPts   Minimum points to form a dense region
 * @returns        Array of snapshots
 */
export function runDBSCAN(
  data: Point2D[],
  epsilon: number,
  minPts: number
): DBSCANSnapshot[] {
  const n = data.length
  if (n === 0) return []

  const classifications: PointClassification[] = new Array(n).fill('unvisited')
  const clusterAssignments: number[] = new Array(n).fill(-1)
  const snapshots: DBSCANSnapshot[] = []
  let stepIndex = 0
  let currentClusterId = -1

  // Pre-compute the ε-neighbourhood for every point (avoids redundant work)
  const neighbourhoods: number[][] = data.map((p, _i) => {
    const neighbours: number[] = []
    for (let j = 0; j < n; j++) {
      if (euclideanDistance(p, data[j]) <= epsilon) {
        neighbours.push(j)
      }
    }
    return neighbours
  })

  /** Capture a snapshot of the current algorithm state. */
  function snap(
    pointIdx: number,
    highlight: { center: number; neighbors: number[] } | null,
    phase: 'classifying' | 'expanding' | 'done'
  ): void {
    snapshots.push({
      stepIndex: stepIndex++,
      currentPointIndex: pointIdx,
      classifications: [...classifications],
      clusterAssignments: [...clusterAssignments],
      neighborhoodHighlight: highlight,
      currentClusterId,
      phase,
    })
  }

  // --- Main loop: iterate over every point ---
  for (let i = 0; i < n; i++) {
    if (classifications[i] !== 'unvisited') continue

    const neighbours = neighbourhoods[i]

    // Snapshot: we are examining this point's neighbourhood
    snap(i, { center: i, neighbors: neighbours }, 'classifying')

    if (neighbours.length < minPts) {
      // Not enough neighbours — tentatively mark as noise.
      // It may later be reclassified as a border point.
      classifications[i] = 'noise'
      snap(i, null, 'classifying')
      continue
    }

    // This is a core point — start a new cluster
    currentClusterId++
    classifications[i] = 'core'
    clusterAssignments[i] = currentClusterId

    snap(i, { center: i, neighbors: neighbours }, 'expanding')

    // Seed set: all neighbours to process (use a queue for BFS expansion)
    const queue: number[] = [...neighbours]
    const inQueue = new Set<number>(neighbours)

    let qIdx = 0
    while (qIdx < queue.length) {
      const j = queue[qIdx++]

      // If j was previously labelled noise, it becomes a border point
      if (classifications[j] === 'noise') {
        classifications[j] = 'border'
        clusterAssignments[j] = currentClusterId
        snap(j, { center: j, neighbors: neighbourhoods[j] }, 'expanding')
        continue
      }

      // Skip already-processed points (core or border of another cluster)
      if (classifications[j] !== 'unvisited') continue

      // Assign j to the current cluster
      clusterAssignments[j] = currentClusterId

      const jNeighbours = neighbourhoods[j]

      snap(j, { center: j, neighbors: jNeighbours }, 'expanding')

      if (jNeighbours.length >= minPts) {
        // j is also a core point — add its neighbours to the queue
        classifications[j] = 'core'
        for (const nb of jNeighbours) {
          if (!inQueue.has(nb)) {
            inQueue.add(nb)
            queue.push(nb)
          }
        }
      } else {
        // j is a border point
        classifications[j] = 'border'
      }

      snap(j, { center: j, neighbors: jNeighbours }, 'expanding')
    }
  }

  // Final snapshot
  snap(-1, null, 'done')

  return snapshots
}
