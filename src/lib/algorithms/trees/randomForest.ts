import { createRng } from '../../math/random'

// ── Types re-exported for consumers ─────────────────────────────────

export interface DecisionBoundaryRegion {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  prediction: number
  confidence: number
}

export interface RandomForestSnapshot {
  treeIndex: number
  totalTrees: number
  /** Boundary regions produced by each individual tree built so far */
  treeBoundaries: DecisionBoundaryRegion[][]
  /** Aggregated majority-vote grid (25 x 25) */
  aggregatedBoundary: { x: number; y: number; prediction: number; confidence: number }[]
  /** [x_importance, y_importance] accumulated across the ensemble */
  featureImportances: number[]
  /** Out-of-bag classification error (fraction of OOB samples misclassified) */
  oobError: number
  /** Training accuracy of the full ensemble on all data */
  trainAccuracy: number
  /** Indices of data points in this tree's bootstrap sample */
  bootstrapIndices: number[]
}

export interface RandomForestConfig {
  nEstimators?: number // default 10
  maxDepth?: number // default 5
  maxFeatures?: number // default sqrt(p) – for 2 features this is 1
  bootstrap?: boolean // default true
  seed?: number
}

// ── Inline simplified decision tree ─────────────────────────────────

interface InternalNode {
  feature: 0 | 1 // 0 = x, 1 = y
  threshold: number
  left: InternalNode
  right: InternalNode
  isLeaf: boolean
  prediction: number
  impurityDecrease: number // weighted impurity decrease at this split
}

interface DataPoint {
  x: number
  y: number
  label: number
}

/** Gini impurity for a set of binary labels */
function giniImpurity(labels: number[]): number {
  if (labels.length === 0) return 0
  const counts = new Map<number, number>()
  for (const l of labels) {
    counts.set(l, (counts.get(l) ?? 0) + 1)
  }
  let gini = 1
  for (const c of counts.values()) {
    const p = c / labels.length
    gini -= p * p
  }
  return gini
}

/** Majority class in a set of labels (ties broken by smaller label) */
function majorityClass(labels: number[]): number {
  const counts = new Map<number, number>()
  for (const l of labels) {
    counts.set(l, (counts.get(l) ?? 0) + 1)
  }
  let best = labels[0]
  let bestCount = 0
  for (const [label, count] of counts) {
    if (count > bestCount || (count === bestCount && label < best)) {
      best = label
      bestCount = count
    }
  }
  return best
}

/**
 * Build a simplified classification tree (Gini criterion).
 *
 * `allowedFeatures` restricts which features can be considered at each split,
 * implementing the random-subspace aspect of Random Forests.
 */
function buildTree(
  data: DataPoint[],
  maxDepth: number,
  minSamplesSplit: number,
  allowedFeatures: (0 | 1)[],
  rng: () => number,
  depth = 0
): InternalNode {
  const labels = data.map((d) => d.label)
  const prediction = majorityClass(labels)
  const currentImpurity = giniImpurity(labels)

  // Leaf conditions
  if (
    depth >= maxDepth ||
    data.length < minSamplesSplit ||
    currentImpurity === 0
  ) {
    return {
      feature: 0,
      threshold: 0,
      left: null!,
      right: null!,
      isLeaf: true,
      prediction,
      impurityDecrease: 0,
    }
  }

  let bestFeature: 0 | 1 = allowedFeatures[0]
  let bestThreshold = 0
  let bestGain = -Infinity
  let bestLeftData: DataPoint[] = []
  let bestRightData: DataPoint[] = []

  for (const feat of allowedFeatures) {
    // Extract feature values and find candidate thresholds
    const values = data.map((d) => (feat === 0 ? d.x : d.y))
    const sorted = [...new Set(values)].sort((a, b) => a - b)

    // Use midpoints between unique sorted values as candidate thresholds
    for (let i = 0; i < sorted.length - 1; i++) {
      const threshold = (sorted[i] + sorted[i + 1]) / 2

      const leftData: DataPoint[] = []
      const rightData: DataPoint[] = []
      for (const d of data) {
        if ((feat === 0 ? d.x : d.y) <= threshold) {
          leftData.push(d)
        } else {
          rightData.push(d)
        }
      }

      if (leftData.length === 0 || rightData.length === 0) continue

      const leftImpurity = giniImpurity(leftData.map((d) => d.label))
      const rightImpurity = giniImpurity(rightData.map((d) => d.label))
      const weightedImpurity =
        (leftData.length / data.length) * leftImpurity +
        (rightData.length / data.length) * rightImpurity
      const gain = currentImpurity - weightedImpurity

      if (gain > bestGain) {
        bestGain = gain
        bestFeature = feat
        bestThreshold = threshold
        bestLeftData = leftData
        bestRightData = rightData
      }
    }
  }

  // If no beneficial split was found, make a leaf
  if (bestGain <= 0 || bestLeftData.length === 0 || bestRightData.length === 0) {
    return {
      feature: 0,
      threshold: 0,
      left: null!,
      right: null!,
      isLeaf: true,
      prediction,
      impurityDecrease: 0,
    }
  }

  const weightedDecrease = bestGain * data.length

  return {
    feature: bestFeature,
    threshold: bestThreshold,
    left: buildTree(bestLeftData, maxDepth, minSamplesSplit, allowedFeatures, rng, depth + 1),
    right: buildTree(bestRightData, maxDepth, minSamplesSplit, allowedFeatures, rng, depth + 1),
    isLeaf: false,
    prediction,
    impurityDecrease: weightedDecrease,
  }
}

/** Predict a single point using an internal tree */
function predictPoint(tree: InternalNode, x: number, y: number): number {
  if (tree.isLeaf) return tree.prediction
  const val = tree.feature === 0 ? x : y
  return val <= tree.threshold
    ? predictPoint(tree.left, x, y)
    : predictPoint(tree.right, x, y)
}

/** Extract axis-aligned boundary regions from a tree by recursive descent */
function extractBoundaries(
  tree: InternalNode,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  data: DataPoint[]
): DecisionBoundaryRegion[] {
  if (tree.isLeaf) {
    // Confidence = fraction of data in this region that matches the prediction
    const pointsInRegion = data.filter(
      (d) => d.x >= xMin && d.x <= xMax && d.y >= yMin && d.y <= yMax
    )
    const matching = pointsInRegion.filter((d) => d.label === tree.prediction).length
    const confidence = pointsInRegion.length > 0 ? matching / pointsInRegion.length : 1

    return [{ xMin, xMax, yMin, yMax, prediction: tree.prediction, confidence }]
  }

  const regions: DecisionBoundaryRegion[] = []

  if (tree.feature === 0) {
    // Split on x
    regions.push(
      ...extractBoundaries(tree.left, xMin, Math.min(tree.threshold, xMax), yMin, yMax, data)
    )
    regions.push(
      ...extractBoundaries(tree.right, Math.max(tree.threshold, xMin), xMax, yMin, yMax, data)
    )
  } else {
    // Split on y
    regions.push(
      ...extractBoundaries(tree.left, xMin, xMax, yMin, Math.min(tree.threshold, yMax), data)
    )
    regions.push(
      ...extractBoundaries(tree.right, xMin, xMax, Math.max(tree.threshold, yMin), yMax, data)
    )
  }

  return regions
}

/** Accumulate feature importance from every internal node in a tree */
function accumulateImportance(tree: InternalNode, importance: number[]): void {
  if (tree.isLeaf) return
  importance[tree.feature] += tree.impurityDecrease
  accumulateImportance(tree.left, importance)
  accumulateImportance(tree.right, importance)
}

// ── Main entry point ────────────────────────────────────────────────

/**
 * Build a Random Forest one tree at a time, returning a snapshot after
 * each tree is added to the ensemble.
 *
 * Each snapshot contains the state of the full ensemble up to that point,
 * allowing the UI to animate the forest growing.
 */
export function runRandomForest(
  data: { x: number; y: number; label: number }[],
  config: RandomForestConfig
): RandomForestSnapshot[] {
  const n = data.length
  if (n === 0) return []

  const nEstimators = config.nEstimators ?? 10
  const maxDepth = config.maxDepth ?? 5
  const bootstrap = config.bootstrap ?? true
  const seed = config.seed ?? 42
  // For 2 features, sqrt(2) ≈ 1.41 → default 1
  const maxFeatures = config.maxFeatures ?? Math.max(1, Math.round(Math.sqrt(2)))

  const rng = createRng(seed)

  // Determine data bounds for boundary extraction
  let xMin = Infinity
  let xMax = -Infinity
  let yMin = Infinity
  let yMax = -Infinity
  for (const d of data) {
    if (d.x < xMin) xMin = d.x
    if (d.x > xMax) xMax = d.x
    if (d.y < yMin) yMin = d.y
    if (d.y > yMax) yMax = d.y
  }
  // Add a small margin
  const xMargin = (xMax - xMin) * 0.05 || 0.5
  const yMargin = (yMax - yMin) * 0.05 || 0.5
  xMin -= xMargin
  xMax += xMargin
  yMin -= yMargin
  yMax += yMargin

  // Prepare the aggregation grid (25 x 25)
  const gridSize = 25
  const gridXs: number[] = []
  const gridYs: number[] = []
  for (let i = 0; i < gridSize; i++) {
    gridXs.push(xMin + ((xMax - xMin) * i) / (gridSize - 1))
    gridYs.push(yMin + ((yMax - yMin) * i) / (gridSize - 1))
  }

  // Accumulated votes per grid cell: votes[classLabel] counts
  const gridVotes: Map<number, number>[][] = []
  for (let i = 0; i < gridSize; i++) {
    gridVotes.push([])
    for (let j = 0; j < gridSize; j++) {
      gridVotes[i].push(new Map())
    }
  }

  // Collect unique class labels
  const classLabels = [...new Set(data.map((d) => d.label))].sort((a, b) => a - b)

  // OOB tracking: for each data point, collect votes from trees where it was OOB
  const oobVotes: Map<number, number>[] = data.map(() => new Map())

  // All features available
  const allFeatures: (0 | 1)[] = [0, 1]

  // Running feature importances
  const totalImportance = [0, 0]

  const snapshots: RandomForestSnapshot[] = []
  const allTreeBoundaries: DecisionBoundaryRegion[][] = []
  const allTrees: InternalNode[] = []

  for (let t = 0; t < nEstimators; t++) {
    // 1. Bootstrap sample
    const bootstrapIndices: number[] = []
    let sampleData: DataPoint[]
    if (bootstrap) {
      for (let i = 0; i < n; i++) {
        bootstrapIndices.push(Math.floor(rng() * n))
      }
      sampleData = bootstrapIndices.map((idx) => data[idx])
    } else {
      for (let i = 0; i < n; i++) {
        bootstrapIndices.push(i)
      }
      sampleData = [...data]
    }

    // 2. Feature subsampling: select maxFeatures random features
    let selectedFeatures: (0 | 1)[]
    if (maxFeatures >= allFeatures.length) {
      selectedFeatures = [...allFeatures]
    } else {
      // Shuffle and pick first maxFeatures
      const shuffled = [...allFeatures]
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1))
        ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
      }
      selectedFeatures = shuffled.slice(0, maxFeatures)
    }

    // 3. Build tree
    const tree = buildTree(sampleData, maxDepth, 2, selectedFeatures, rng)
    allTrees.push(tree)

    // 4. Extract boundary regions for this tree
    const boundaries = extractBoundaries(tree, xMin, xMax, yMin, yMax, sampleData)
    allTreeBoundaries.push(boundaries)

    // 5. Update grid votes with this new tree
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const pred = predictPoint(tree, gridXs[i], gridYs[j])
        gridVotes[i][j].set(pred, (gridVotes[i][j].get(pred) ?? 0) + 1)
      }
    }

    // 6. Build aggregated boundary from grid
    const aggregatedBoundary: { x: number; y: number; prediction: number; confidence: number }[] =
      []
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const votes = gridVotes[i][j]
        let bestLabel = classLabels[0]
        let bestCount = 0
        let totalVotes = 0
        for (const [label, count] of votes) {
          totalVotes += count
          if (count > bestCount || (count === bestCount && label < bestLabel)) {
            bestLabel = label
            bestCount = count
          }
        }
        aggregatedBoundary.push({
          x: gridXs[i],
          y: gridYs[j],
          prediction: bestLabel,
          confidence: totalVotes > 0 ? bestCount / totalVotes : 0,
        })
      }
    }

    // 7. Accumulate feature importances
    const treeImportance = [0, 0]
    accumulateImportance(tree, treeImportance)
    totalImportance[0] += treeImportance[0]
    totalImportance[1] += treeImportance[1]

    // Normalise so they sum to 1 (if non-zero)
    const impSum = totalImportance[0] + totalImportance[1]
    const featureImportances =
      impSum > 0
        ? [totalImportance[0] / impSum, totalImportance[1] / impSum]
        : [0.5, 0.5]

    // 8. OOB error
    if (bootstrap) {
      const oobSet = new Set(bootstrapIndices)
      for (let i = 0; i < n; i++) {
        if (!oobSet.has(i)) {
          // This point was NOT in the bootstrap sample → OOB
          const pred = predictPoint(tree, data[i].x, data[i].y)
          oobVotes[i].set(pred, (oobVotes[i].get(pred) ?? 0) + 1)
        }
      }
    }

    // Compute OOB error from accumulated OOB votes
    let oobCorrect = 0
    let oobTotal = 0
    for (let i = 0; i < n; i++) {
      if (oobVotes[i].size > 0) {
        oobTotal++
        let bestLabel = classLabels[0]
        let bestCount = 0
        for (const [label, count] of oobVotes[i]) {
          if (count > bestCount || (count === bestCount && label < bestLabel)) {
            bestLabel = label
            bestCount = count
          }
        }
        if (bestLabel === data[i].label) oobCorrect++
      }
    }
    const oobError = oobTotal > 0 ? 1 - oobCorrect / oobTotal : 0

    // 9. Training accuracy of the full ensemble
    let trainCorrect = 0
    for (let i = 0; i < n; i++) {
      // Majority vote across all trees built so far
      const votes = new Map<number, number>()
      for (const ensembleTree of allTrees) {
        const pred = predictPoint(ensembleTree, data[i].x, data[i].y)
        votes.set(pred, (votes.get(pred) ?? 0) + 1)
      }
      let bestLabel = classLabels[0]
      let bestCount = 0
      for (const [label, count] of votes) {
        if (count > bestCount || (count === bestCount && label < bestLabel)) {
          bestLabel = label
          bestCount = count
        }
      }
      if (bestLabel === data[i].label) trainCorrect++
    }
    const trainAccuracy = trainCorrect / n

    snapshots.push({
      treeIndex: t,
      totalTrees: nEstimators,
      treeBoundaries: allTreeBoundaries.map((b) => [...b]),
      aggregatedBoundary,
      featureImportances,
      oobError,
      trainAccuracy,
      bootstrapIndices: [...bootstrapIndices],
    })
  }

  return snapshots
}
