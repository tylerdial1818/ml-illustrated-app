// ── Public types ─────────────────────────────────────────────────────

/** A single data point with 2-D features and a class label. */
export interface TreeDataPoint {
  x: number
  y: number
  label: number
}

/** Supported impurity criteria. */
export type SplitCriterion = 'gini' | 'entropy'

/** Which feature (axis) a split is made on. */
export type FeatureIndex = 0 | 1

/** Configuration for building the decision tree. */
export interface DecisionTreeConfig {
  criterion?: SplitCriterion
  maxDepth?: number
  minSamplesSplit?: number
  seed?: number
}

/** An axis-aligned rectangular region produced by the tree's splits. */
export interface Region {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  /** The majority class label for this region. */
  label: number
  /** Class distribution as counts: Map<label, count>. */
  classCounts: Record<number, number>
  /** The leaf node id that owns this region. */
  nodeId: number
}

/** A node in the decision tree (used for tree diagram rendering). */
export interface TreeNode {
  id: number
  depth: number
  /** Indices into the original data array that reach this node. */
  sampleIndices: number[]
  /** Class distribution at this node: label -> count. */
  classCounts: Record<number, number>
  /** Impurity value (Gini or Entropy) at this node. */
  impurity: number
  /** The majority class at this node. */
  prediction: number
  /** Number of samples at this node. */
  numSamples: number
  /** True if this is a leaf (no children). */
  isLeaf: boolean
  // Split info (only present for internal/split nodes)
  /** Feature index: 0 = x, 1 = y. */
  splitFeature?: FeatureIndex
  /** Split threshold value. */
  splitThreshold?: number
  /** Weighted impurity decrease from this split. */
  impurityDecrease?: number
  /** Left child (feature <= threshold). */
  left?: TreeNode
  /** Right child (feature > threshold). */
  right?: TreeNode
  /** Bounding box for this node's region in feature space. */
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number }
}

/** Information about a single split (returned by growOneStep). */
export interface SplitResult {
  /** The node that was split. */
  nodeId: number
  feature: FeatureIndex
  featureName: string
  threshold: number
  impurityBefore: number
  impurityAfter: number
  impurityDecrease: number
  leftSamples: number
  rightSamples: number
  depth: number
  /** True if no more splits are possible after this one. */
  treeComplete: boolean
}

/** Result from predictWithPath: prediction plus the full decision path. */
export interface PathResult {
  prediction: number
  /** Probability distribution over class labels. */
  classProbabilities: Record<number, number>
  /** Ordered list of node IDs from root to the leaf. */
  path: number[]
  /** The leaf node that produced the prediction. */
  leafNodeId: number
}

/** A snapshot capturing the tree state after each split (for animation). */
export interface DecisionTreeSnapshot {
  /** 0-based index of this snapshot (i.e. which split number). */
  step: number
  /** The full tree structure at this point. */
  tree: TreeNode
  /** Current axis-aligned decision boundary regions. */
  regions: Region[]
  /** The node that was just split (null for the initial pre-split snapshot). */
  splitNodeId: number | null
  /** Info about the split just performed (null for initial snapshot). */
  splitInfo: SplitResult | null
  /** Training accuracy at this point. */
  trainAccuracy: number
  /** Number of leaves in the current tree. */
  numLeaves: number
}

// ── Impurity functions ───────────────────────────────────────────────

function giniImpurity(counts: Record<number, number>, total: number): number {
  if (total === 0) return 0
  let sumSq = 0
  for (const label in counts) {
    const p = counts[label] / total
    sumSq += p * p
  }
  return 1 - sumSq
}

function entropyImpurity(counts: Record<number, number>, total: number): number {
  if (total === 0) return 0
  let ent = 0
  for (const label in counts) {
    const p = counts[label] / total
    if (p > 0) {
      ent -= p * Math.log2(p)
    }
  }
  return ent
}

type ImpurityFn = (counts: Record<number, number>, total: number) => number

// ── Helper utilities ─────────────────────────────────────────────────

function classCounts(data: TreeDataPoint[], indices: number[]): Record<number, number> {
  const counts: Record<number, number> = {}
  for (const i of indices) {
    const label = data[i].label
    counts[label] = (counts[label] || 0) + 1
  }
  return counts
}

function majorityLabel(counts: Record<number, number>): number {
  let bestLabel = -1
  let bestCount = -1
  for (const label in counts) {
    if (counts[label] > bestCount) {
      bestCount = counts[label]
      bestLabel = Number(label)
    }
  }
  return bestLabel
}

function classProbabilities(counts: Record<number, number>, total: number): Record<number, number> {
  const probs: Record<number, number> = {}
  for (const label in counts) {
    probs[label] = counts[label] / total
  }
  return probs
}

function computeAccuracy(data: TreeDataPoint[], root: TreeNode): number {
  let correct = 0
  for (let i = 0; i < data.length; i++) {
    const pred = predictFromNode(root, data[i])
    if (pred === data[i].label) correct++
  }
  return data.length > 0 ? correct / data.length : 0
}

function predictFromNode(node: TreeNode, point: { x: number; y: number }): number {
  if (node.isLeaf) return node.prediction
  const val = node.splitFeature === 0 ? point.x : point.y
  if (val <= node.splitThreshold!) {
    return predictFromNode(node.left!, point)
  }
  return predictFromNode(node.right!, point)
}

function countLeaves(node: TreeNode): number {
  if (node.isLeaf) return 1
  return countLeaves(node.left!) + countLeaves(node.right!)
}

function featureName(f: FeatureIndex): string {
  return f === 0 ? 'x' : 'y'
}

function getFeatureValue(point: TreeDataPoint, f: FeatureIndex): number {
  return f === 0 ? point.x : point.y
}

// ── Deep-clone helpers ───────────────────────────────────────────────

function cloneTreeNode(node: TreeNode): TreeNode {
  const copy: TreeNode = {
    id: node.id,
    depth: node.depth,
    sampleIndices: [...node.sampleIndices],
    classCounts: { ...node.classCounts },
    impurity: node.impurity,
    prediction: node.prediction,
    numSamples: node.numSamples,
    isLeaf: node.isLeaf,
    bounds: { ...node.bounds },
  }
  if (node.splitFeature !== undefined) copy.splitFeature = node.splitFeature
  if (node.splitThreshold !== undefined) copy.splitThreshold = node.splitThreshold
  if (node.impurityDecrease !== undefined) copy.impurityDecrease = node.impurityDecrease
  if (node.left) copy.left = cloneTreeNode(node.left)
  if (node.right) copy.right = cloneTreeNode(node.right)
  return copy
}

// ── Best split search ────────────────────────────────────────────────

interface CandidateSplit {
  feature: FeatureIndex
  threshold: number
  leftIndices: number[]
  rightIndices: number[]
  impurityDecrease: number
  leftCounts: Record<number, number>
  rightCounts: Record<number, number>
}

/**
 * Find the best binary split for a given set of sample indices.
 * Iterates over both features (x=0, y=1), sorts by feature value,
 * and evaluates every midpoint between consecutive distinct values.
 */
function findBestSplit(
  data: TreeDataPoint[],
  indices: number[],
  impurityFn: ImpurityFn,
  parentImpurity: number
): CandidateSplit | null {
  const n = indices.length
  if (n <= 1) return null

  let bestSplit: CandidateSplit | null = null
  let bestDecrease = -Infinity

  for (const feature of [0, 1] as FeatureIndex[]) {
    // Sort indices by this feature's value
    const sorted = [...indices].sort(
      (a, b) => getFeatureValue(data[a], feature) - getFeatureValue(data[b], feature)
    )

    // Build running class counts for left partition
    const leftCounts: Record<number, number> = {}
    const rightCounts: Record<number, number> = {}

    // Initialize right to contain everything
    for (const idx of sorted) {
      const label = data[idx].label
      rightCounts[label] = (rightCounts[label] || 0) + 1
    }

    // Sweep from left to right, moving one sample at a time
    for (let i = 0; i < n - 1; i++) {
      const idx = sorted[i]
      const label = data[idx].label

      // Move sample from right to left
      leftCounts[label] = (leftCounts[label] || 0) + 1
      rightCounts[label]--
      if (rightCounts[label] === 0) delete rightCounts[label]

      const leftSize = i + 1
      const rightSize = n - leftSize

      // Only consider split points between distinct feature values
      const currentVal = getFeatureValue(data[sorted[i]], feature)
      const nextVal = getFeatureValue(data[sorted[i + 1]], feature)
      if (currentVal === nextVal) continue

      const threshold = (currentVal + nextVal) / 2

      // Compute weighted impurity of the children
      const leftImpurity = impurityFn(leftCounts, leftSize)
      const rightImpurity = impurityFn(rightCounts, rightSize)
      const weightedChildImpurity =
        (leftSize / n) * leftImpurity + (rightSize / n) * rightImpurity
      const decrease = parentImpurity - weightedChildImpurity

      if (decrease > bestDecrease) {
        bestDecrease = decrease
        bestSplit = {
          feature,
          threshold,
          leftIndices: sorted.slice(0, i + 1),
          rightIndices: sorted.slice(i + 1),
          impurityDecrease: decrease,
          leftCounts: { ...leftCounts },
          rightCounts: { ...rightCounts },
        }
      }
    }
  }

  // Only accept splits that actually reduce impurity
  if (bestSplit && bestSplit.impurityDecrease <= 0) return null

  return bestSplit
}

// ── DecisionTree class ───────────────────────────────────────────────

export class DecisionTree {
  private data: TreeDataPoint[]
  private criterion: SplitCriterion
  private maxDepth: number
  private minSamplesSplit: number
  private impurityFn: ImpurityFn
  private root: TreeNode
  private nextNodeId: number
  /** Leaf nodes eligible for splitting, ordered by impurity decrease potential. */
  private splittableLeaves: TreeNode[]
  private splitCount: number

  constructor(data: TreeDataPoint[], config: DecisionTreeConfig = {}) {
    this.data = data
    this.criterion = config.criterion ?? 'gini'
    this.maxDepth = config.maxDepth ?? 10
    this.minSamplesSplit = config.minSamplesSplit ?? 2
    this.impurityFn = this.criterion === 'gini' ? giniImpurity : entropyImpurity
    this.nextNodeId = 0
    this.splitCount = 0

    // Determine feature-space bounds from the data
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
    for (const p of data) {
      if (p.x < xMin) xMin = p.x
      if (p.x > xMax) xMax = p.x
      if (p.y < yMin) yMin = p.y
      if (p.y > yMax) yMax = p.y
    }
    // Add a small margin so boundary points are fully inside a region
    const xMargin = (xMax - xMin) * 0.05 || 0.5
    const yMargin = (yMax - yMin) * 0.05 || 0.5
    xMin -= xMargin; xMax += xMargin
    yMin -= yMargin; yMax += yMargin

    const allIndices = data.map((_, i) => i)
    const counts = classCounts(data, allIndices)
    const impurity = this.impurityFn(counts, allIndices.length)

    // Create root node (initially a leaf containing all data)
    this.root = {
      id: this.nextNodeId++,
      depth: 0,
      sampleIndices: allIndices,
      classCounts: counts,
      impurity,
      prediction: majorityLabel(counts),
      numSamples: allIndices.length,
      isLeaf: true,
      bounds: { xMin, xMax, yMin, yMax },
    }

    // Initialise the splittable leaves queue
    this.splittableLeaves = []
    this.enqueueSplittableLeaf(this.root)
  }

  // ── Public API ─────────────────────────────────────────────────────

  /**
   * Grow the tree by exactly one split (the best available).
   * Returns information about the split that was made.
   * Returns null if the tree is fully grown.
   */
  growOneStep(): SplitResult | null {
    // Find the best leaf to split (greatest impurity decrease)
    while (this.splittableLeaves.length > 0) {
      const leaf = this.splittableLeaves.shift()!

      // Re-check constraints: the leaf might have become ineligible
      if (!leaf.isLeaf) continue
      if (leaf.depth >= this.maxDepth) continue
      if (leaf.numSamples < this.minSamplesSplit) continue

      const split = findBestSplit(
        this.data,
        leaf.sampleIndices,
        this.impurityFn,
        leaf.impurity
      )
      if (!split) continue

      // Perform the split
      this.applySplit(leaf, split)
      this.splitCount++

      return {
        nodeId: leaf.id,
        feature: split.feature,
        featureName: featureName(split.feature),
        threshold: split.threshold,
        impurityBefore: leaf.impurity,
        impurityAfter:
          (split.leftIndices.length / leaf.numSamples) *
            this.impurityFn(split.leftCounts, split.leftIndices.length) +
          (split.rightIndices.length / leaf.numSamples) *
            this.impurityFn(split.rightCounts, split.rightIndices.length),
        impurityDecrease: split.impurityDecrease,
        leftSamples: split.leftIndices.length,
        rightSamples: split.rightIndices.length,
        depth: leaf.depth,
        treeComplete: this.splittableLeaves.length === 0,
      }
    }

    return null
  }

  /**
   * Check whether any further splits are possible.
   */
  isComplete(): boolean {
    return this.splittableLeaves.length === 0
  }

  /**
   * Return the current axis-aligned decision boundary regions.
   * Each leaf node produces one rectangular region.
   */
  getDecisionBoundary(): Region[] {
    const regions: Region[] = []
    this.collectRegions(this.root, regions)
    return regions
  }

  /**
   * Return the full tree structure (deep copy) for tree diagram rendering.
   */
  getTreeStructure(): TreeNode {
    return cloneTreeNode(this.root)
  }

  /**
   * Predict the class for a given point, and return the full decision path
   * from root to leaf (for hover-to-highlight-path feature).
   */
  predictWithPath(point: { x: number; y: number }): PathResult {
    const path: number[] = []
    let node = this.root

    while (!node.isLeaf) {
      path.push(node.id)
      const val = node.splitFeature === 0 ? point.x : point.y
      if (val <= node.splitThreshold!) {
        node = node.left!
      } else {
        node = node.right!
      }
    }
    path.push(node.id)

    return {
      prediction: node.prediction,
      classProbabilities: classProbabilities(node.classCounts, node.numSamples),
      path,
      leafNodeId: node.id,
    }
  }

  /**
   * Get the training accuracy of the current tree.
   */
  getTrainAccuracy(): number {
    return computeAccuracy(this.data, this.root)
  }

  /**
   * Get the number of leaves in the current tree.
   */
  getNumLeaves(): number {
    return countLeaves(this.root)
  }

  // ── Private helpers ────────────────────────────────────────────────

  /**
   * Apply a candidate split to a leaf node, turning it into an internal
   * node with two children.
   */
  private applySplit(leaf: TreeNode, split: CandidateSplit): void {
    leaf.isLeaf = false
    leaf.splitFeature = split.feature
    leaf.splitThreshold = split.threshold
    leaf.impurityDecrease = split.impurityDecrease

    const leftImpurity = this.impurityFn(split.leftCounts, split.leftIndices.length)
    const rightImpurity = this.impurityFn(split.rightCounts, split.rightIndices.length)

    // Compute child bounding boxes by bisecting the parent along the split axis
    const leftBounds = { ...leaf.bounds }
    const rightBounds = { ...leaf.bounds }

    if (split.feature === 0) {
      // Split on x: left gets x <= threshold, right gets x > threshold
      leftBounds.xMax = split.threshold
      rightBounds.xMin = split.threshold
    } else {
      // Split on y: left gets y <= threshold, right gets y > threshold
      leftBounds.yMax = split.threshold
      rightBounds.yMin = split.threshold
    }

    leaf.left = {
      id: this.nextNodeId++,
      depth: leaf.depth + 1,
      sampleIndices: split.leftIndices,
      classCounts: split.leftCounts,
      impurity: leftImpurity,
      prediction: majorityLabel(split.leftCounts),
      numSamples: split.leftIndices.length,
      isLeaf: true,
      bounds: leftBounds,
    }

    leaf.right = {
      id: this.nextNodeId++,
      depth: leaf.depth + 1,
      sampleIndices: split.rightIndices,
      classCounts: split.rightCounts,
      impurity: rightImpurity,
      prediction: majorityLabel(split.rightCounts),
      numSamples: split.rightIndices.length,
      isLeaf: true,
      bounds: rightBounds,
    }

    // Add new children to the splittable queue if they meet the criteria
    this.enqueueSplittableLeaf(leaf.left)
    this.enqueueSplittableLeaf(leaf.right)
  }

  /**
   * Add a leaf to the splittable queue if it can potentially be split.
   * We pre-check basic constraints here to avoid unnecessary work later.
   */
  private enqueueSplittableLeaf(leaf: TreeNode): void {
    // Cannot split if already at max depth
    if (leaf.depth >= this.maxDepth) return
    // Cannot split if too few samples
    if (leaf.numSamples < this.minSamplesSplit) return
    // Cannot split if pure (impurity is zero)
    if (leaf.impurity <= 0) return

    // Check if there is at least one valid split
    const split = findBestSplit(
      this.data,
      leaf.sampleIndices,
      this.impurityFn,
      leaf.impurity
    )
    if (!split) return

    // Insert into the queue sorted by impurity decrease (descending)
    // so we always pick the best split first.
    let inserted = false
    for (let i = 0; i < this.splittableLeaves.length; i++) {
      if (split.impurityDecrease * leaf.numSamples >
          this.getBestDecrease(this.splittableLeaves[i]) * this.splittableLeaves[i].numSamples) {
        this.splittableLeaves.splice(i, 0, leaf)
        inserted = true
        break
      }
    }
    if (!inserted) {
      this.splittableLeaves.push(leaf)
    }
  }

  /**
   * Get the best possible impurity decrease for a leaf (used for queue ordering).
   * This recomputes the best split, which is acceptable for small/medium datasets.
   */
  private getBestDecrease(leaf: TreeNode): number {
    const split = findBestSplit(
      this.data,
      leaf.sampleIndices,
      this.impurityFn,
      leaf.impurity
    )
    return split ? split.impurityDecrease : 0
  }

  /**
   * Recursively collect leaf regions from the tree.
   */
  private collectRegions(node: TreeNode, regions: Region[]): void {
    if (node.isLeaf) {
      regions.push({
        xMin: node.bounds.xMin,
        xMax: node.bounds.xMax,
        yMin: node.bounds.yMin,
        yMax: node.bounds.yMax,
        label: node.prediction,
        classCounts: { ...node.classCounts },
        nodeId: node.id,
      })
      return
    }
    this.collectRegions(node.left!, regions)
    this.collectRegions(node.right!, regions)
  }
}

// ── Convenience function (snapshot pattern) ──────────────────────────

/**
 * Build a decision tree step-by-step, producing an array of snapshots
 * (one per split) suitable for animation playback.
 *
 * The first snapshot (step 0) captures the initial state before any splits.
 * Each subsequent snapshot captures the state after one split.
 *
 * @param data   Array of labelled 2-D points
 * @param config Tree hyper-parameters
 * @returns      Array of DecisionTreeSnapshot, one per split + initial state
 */
export function runDecisionTree(
  data: TreeDataPoint[],
  config: DecisionTreeConfig = {}
): DecisionTreeSnapshot[] {
  if (data.length === 0) return []

  const tree = new DecisionTree(data, config)
  const snapshots: DecisionTreeSnapshot[] = []

  // Snapshot 0: initial state (single root leaf, no splits)
  snapshots.push({
    step: 0,
    tree: tree.getTreeStructure(),
    regions: tree.getDecisionBoundary(),
    splitNodeId: null,
    splitInfo: null,
    trainAccuracy: tree.getTrainAccuracy(),
    numLeaves: tree.getNumLeaves(),
  })

  // Grow the tree one split at a time, capturing a snapshot after each
  let step = 1
  while (!tree.isComplete()) {
    const result = tree.growOneStep()
    if (!result) break

    snapshots.push({
      step,
      tree: tree.getTreeStructure(),
      regions: tree.getDecisionBoundary(),
      splitNodeId: result.nodeId,
      splitInfo: result,
      trainAccuracy: tree.getTrainAccuracy(),
      numLeaves: tree.getNumLeaves(),
    })

    step++
  }

  return snapshots
}
