import { createRng } from '../../math/random'

// ── Types ───────────────────────────────────────────────────────────

export interface RegressionTreeNode {
  feature: number // 0 for x (1D input)
  threshold: number | null
  value: number // predicted value at this node (mean of y values)
  left: RegressionTreeNode | null
  right: RegressionTreeNode | null
  isLeaf: boolean
}

export interface GBTSnapshot {
  round: number
  totalRounds: number
  /** Current ensemble prediction for each training point */
  predictions: number[]
  /** Current residuals (y - predictions) */
  residuals: number[]
  /** Contribution of the tree just added this round (scaled by learning rate) */
  newTreePrediction: number[]
  /** Structure of the tree just fitted */
  newTreeStructure: RegressionTreeNode
  /** All component trees so far: each with its predictions and learning-rate weight */
  componentTrees: { prediction: number[]; weight: number }[]
  /** Training MSE loss */
  trainLoss: number
  /** Fine-grained prediction curve for smooth rendering (100 points) */
  predictionCurve: { x: number; y: number }[]
}

export interface GBTConfig {
  nEstimators?: number // default 20
  learningRate?: number // default 0.1
  maxDepth?: number // default 2 (weak learners)
  seed?: number
}

// ── Inline 1D regression tree ───────────────────────────────────────

interface RegressionSample {
  x: number
  y: number
}

/** Mean of an array of numbers */
function mean(values: number[]): number {
  if (values.length === 0) return 0
  let sum = 0
  for (const v of values) sum += v
  return sum / values.length
}

/** MSE of a set of values around their mean */
function mse(values: number[]): number {
  if (values.length === 0) return 0
  const m = mean(values)
  let sum = 0
  for (const v of values) {
    const d = v - m
    sum += d * d
  }
  return sum / values.length
}

/**
 * Build a regression tree on 1D input (splits on x, predicts y).
 *
 * Uses MSE reduction as the split criterion. The tree produces axis-aligned
 * splits on x, with each leaf predicting the mean y of its samples.
 */
function buildRegressionTree(
  data: RegressionSample[],
  maxDepth: number,
  minSamplesSplit: number,
  depth: number = 0
): RegressionTreeNode {
  const yValues = data.map((d) => d.y)
  const nodeValue = mean(yValues)

  // Leaf conditions
  if (depth >= maxDepth || data.length < minSamplesSplit || data.length <= 1) {
    return {
      feature: 0,
      threshold: null,
      value: nodeValue,
      left: null,
      right: null,
      isLeaf: true,
    }
  }

  const currentMSE = mse(yValues)

  // If all y values are the same, nothing to split
  if (currentMSE < 1e-15) {
    return {
      feature: 0,
      threshold: null,
      value: nodeValue,
      left: null,
      right: null,
      isLeaf: true,
    }
  }

  // Sort data by x to find the best split
  const sorted = [...data].sort((a, b) => a.x - b.x)
  const uniqueXs = [...new Set(sorted.map((d) => d.x))].sort((a, b) => a - b)

  let bestThreshold = 0
  let bestReduction = -Infinity
  let bestLeftData: RegressionSample[] = []
  let bestRightData: RegressionSample[] = []

  // Try midpoints between unique x values
  for (let i = 0; i < uniqueXs.length - 1; i++) {
    const threshold = (uniqueXs[i] + uniqueXs[i + 1]) / 2

    const leftData: RegressionSample[] = []
    const rightData: RegressionSample[] = []
    for (const d of sorted) {
      if (d.x <= threshold) {
        leftData.push(d)
      } else {
        rightData.push(d)
      }
    }

    if (leftData.length === 0 || rightData.length === 0) continue

    const leftMSE = mse(leftData.map((d) => d.y))
    const rightMSE = mse(rightData.map((d) => d.y))
    const weightedMSE =
      (leftData.length / data.length) * leftMSE +
      (rightData.length / data.length) * rightMSE
    const reduction = currentMSE - weightedMSE

    if (reduction > bestReduction) {
      bestReduction = reduction
      bestThreshold = threshold
      bestLeftData = leftData
      bestRightData = rightData
    }
  }

  // If no beneficial split, make a leaf
  if (bestReduction <= 0 || bestLeftData.length === 0 || bestRightData.length === 0) {
    return {
      feature: 0,
      threshold: null,
      value: nodeValue,
      left: null,
      right: null,
      isLeaf: true,
    }
  }

  return {
    feature: 0,
    threshold: bestThreshold,
    value: nodeValue,
    left: buildRegressionTree(bestLeftData, maxDepth, minSamplesSplit, depth + 1),
    right: buildRegressionTree(bestRightData, maxDepth, minSamplesSplit, depth + 1),
    isLeaf: false,
  }
}

/** Predict a single x value using a regression tree */
function predictRegressionTree(tree: RegressionTreeNode, x: number): number {
  if (tree.isLeaf || tree.threshold === null) return tree.value
  return x <= tree.threshold
    ? predictRegressionTree(tree.left!, x)
    : predictRegressionTree(tree.right!, x)
}

// ── Main entry point ────────────────────────────────────────────────

/**
 * Run Gradient Boosted Trees for 1D regression.
 *
 * The algorithm:
 * 1. Initialise predictions to the mean of y.
 * 2. Each round: compute residuals, fit a small regression tree to the
 *    residuals, scale by learning rate, and add to the ensemble.
 * 3. Produce a snapshot after each boosting round so the UI can animate
 *    the ensemble being built stage by stage.
 *
 * The `predictionCurve` in each snapshot provides 100 evenly spaced points
 * across the x range for smooth line rendering.
 */
export function runGradientBoostedTrees(
  data: { x: number; y: number }[],
  config: GBTConfig
): GBTSnapshot[] {
  const n = data.length
  if (n === 0) return []

  const nEstimators = config.nEstimators ?? 20
  const learningRate = config.learningRate ?? 0.1
  const maxDepth = config.maxDepth ?? 2
  // The seed is accepted in config for API consistency but the regression
  // tree builder is deterministic (greedy MSE splits). We consume the
  // import to keep the interface uniform with Random Forest.
  void createRng(config.seed ?? 42)

  const yValues = data.map((d) => d.y)
  const xValues = data.map((d) => d.x)

  // Determine x range for the prediction curve
  let xMin = Infinity
  let xMax = -Infinity
  for (const x of xValues) {
    if (x < xMin) xMin = x
    if (x > xMax) xMax = x
  }
  const xMargin = (xMax - xMin) * 0.05 || 0.5
  xMin -= xMargin
  xMax += xMargin

  // Fine-grained x values for prediction curve
  const curvePoints = 100
  const curveXs: number[] = []
  for (let i = 0; i < curvePoints; i++) {
    curveXs.push(xMin + ((xMax - xMin) * i) / (curvePoints - 1))
  }

  // Step 1: Initialise predictions to the overall mean
  const yMean = mean(yValues)
  const predictions = new Array<number>(n).fill(yMean)
  const curvePredictions = new Array<number>(curvePoints).fill(yMean)

  const snapshots: GBTSnapshot[] = []
  const componentTrees: { prediction: number[]; weight: number }[] = []

  for (let round = 0; round < nEstimators; round++) {
    // Step 2: Compute residuals = y - current predictions
    const residuals = yValues.map((y, i) => y - predictions[i])

    // Step 3: Fit a regression tree to the residuals
    const residualData: RegressionSample[] = data.map((d, i) => ({
      x: d.x,
      y: residuals[i],
    }))
    const tree = buildRegressionTree(residualData, maxDepth, 2)

    // Step 4: Compute this tree's predictions for training data and curve
    const treePredictions = data.map((d) => predictRegressionTree(tree, d.x))
    const scaledTreePredictions = treePredictions.map((p) => learningRate * p)

    const curveTreePredictions = curveXs.map((x) => predictRegressionTree(tree, x))

    // Step 5: Update ensemble predictions
    for (let i = 0; i < n; i++) {
      predictions[i] += scaledTreePredictions[i]
    }
    for (let i = 0; i < curvePoints; i++) {
      curvePredictions[i] += learningRate * curveTreePredictions[i]
    }

    // Record component tree
    componentTrees.push({
      prediction: [...scaledTreePredictions],
      weight: learningRate,
    })

    // Step 6: Compute training MSE
    let trainLoss = 0
    for (let i = 0; i < n; i++) {
      const diff = yValues[i] - predictions[i]
      trainLoss += diff * diff
    }
    trainLoss /= n

    // Step 7: Build prediction curve
    const predictionCurve = curveXs.map((x, i) => ({
      x,
      y: curvePredictions[i],
    }))

    snapshots.push({
      round,
      totalRounds: nEstimators,
      predictions: [...predictions],
      residuals: [...residuals],
      newTreePrediction: scaledTreePredictions,
      newTreeStructure: tree,
      componentTrees: componentTrees.map((ct) => ({
        prediction: [...ct.prediction],
        weight: ct.weight,
      })),
      trainLoss,
      predictionCurve,
    })
  }

  return snapshots
}
