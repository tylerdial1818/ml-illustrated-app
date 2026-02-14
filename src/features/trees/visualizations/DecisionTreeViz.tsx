import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { ConvergenceChart } from '../../../components/viz/ConvergenceChart'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'
import { COLORS } from '../../../types'
import { makeNoisyClassification, type ClassificationPoint } from '../../../lib/data/treeDataGenerators'

/* ------------------------------------------------------------------ */
/*  Self-contained decision tree algorithm                            */
/* ------------------------------------------------------------------ */

interface TreeNode {
  id: number
  depth: number
  featureIndex: number | null   // 0 = x, 1 = y
  threshold: number | null
  classCounts: number[]         // [countClass0, countClass1]
  prediction: number
  left: TreeNode | null
  right: TreeNode | null
  samples: number[]             // indices into data
  impurity: number
  isLeaf: boolean
  xMin: number; xMax: number; yMin: number; yMax: number
}

interface BoundaryRegion {
  xMin: number; xMax: number; yMin: number; yMax: number
  prediction: number
  depth: number
}

interface SplitLine {
  featureIndex: number
  threshold: number
  xMin: number; xMax: number; yMin: number; yMax: number
  depth: number
}

interface DTSnapshot {
  tree: TreeNode | null
  nodeCount: number
  regions: BoundaryRegion[]
  splits: SplitLine[]
  accuracy: number
  growingNodeId: number | null
  description: string
}

type CriterionFn = (counts: number[], total: number) => number

function giniImpurity(counts: number[], total: number): number {
  if (total === 0) return 0
  let sum = 0
  for (const c of counts) {
    const p = c / total
    sum += p * p
  }
  return 1 - sum
}

function entropyImpurity(counts: number[], total: number): number {
  if (total === 0) return 0
  let sum = 0
  for (const c of counts) {
    if (c === 0) continue
    const p = c / total
    sum -= p * Math.log2(p)
  }
  return sum
}

function countClasses(data: ClassificationPoint[], indices: number[], nClasses: number): number[] {
  const counts = new Array(nClasses).fill(0)
  for (const i of indices) counts[data[i].label]++
  return counts
}

function findBestSplit(
  data: ClassificationPoint[],
  indices: number[],
  nClasses: number,
  criterion: CriterionFn,
): { featureIndex: number; threshold: number; leftIndices: number[]; rightIndices: number[]; gain: number } | null {
  if (indices.length < 2) return null

  const total = indices.length
  const parentCounts = countClasses(data, indices, nClasses)
  const parentImpurity = criterion(parentCounts, total)

  let bestGain = 0
  let bestSplit: { featureIndex: number; threshold: number; leftIndices: number[]; rightIndices: number[] } | null = null

  for (let feat = 0; feat < 2; feat++) {
    const values = indices.map(i => feat === 0 ? data[i].x : data[i].y).sort((a, b) => a - b)
    const uniqueThresholds: number[] = []
    for (let i = 0; i < values.length - 1; i++) {
      if (values[i] !== values[i + 1]) {
        uniqueThresholds.push((values[i] + values[i + 1]) / 2)
      }
    }
    // subsample thresholds if too many
    const thresholds = uniqueThresholds.length > 20
      ? uniqueThresholds.filter((_, i) => i % Math.ceil(uniqueThresholds.length / 20) === 0)
      : uniqueThresholds

    for (const thresh of thresholds) {
      const leftIdx: number[] = []
      const rightIdx: number[] = []
      for (const i of indices) {
        const val = feat === 0 ? data[i].x : data[i].y
        if (val <= thresh) leftIdx.push(i)
        else rightIdx.push(i)
      }
      if (leftIdx.length === 0 || rightIdx.length === 0) continue

      const leftCounts = countClasses(data, leftIdx, nClasses)
      const rightCounts = countClasses(data, rightIdx, nClasses)
      const leftImp = criterion(leftCounts, leftIdx.length)
      const rightImp = criterion(rightCounts, rightIdx.length)
      const gain = parentImpurity
        - (leftIdx.length / total) * leftImp
        - (rightIdx.length / total) * rightImp

      if (gain > bestGain) {
        bestGain = gain
        bestSplit = { featureIndex: feat, threshold: thresh, leftIndices: leftIdx, rightIndices: rightIdx }
      }
    }
  }

  return bestSplit ? { ...bestSplit, gain: bestGain } : null
}

function buildDecisionTreeSnapshots(
  data: ClassificationPoint[],
  maxDepth: number,
  criterionType: 'gini' | 'entropy',
): DTSnapshot[] {
  const nClasses = 2
  const criterion: CriterionFn = criterionType === 'gini' ? giniImpurity : entropyImpurity
  const allIndices = data.map((_, i) => i)
  const xExtent = d3.extent(data, d => d.x) as [number, number]
  const yExtent = d3.extent(data, d => d.y) as [number, number]
  const snapshots: DTSnapshot[] = []
  let nodeIdCounter = 0

  function computeAccuracy(tree: TreeNode | null): number {
    if (!tree) return 0
    let correct = 0
    for (const pt of data) {
      let node = tree
      while (!node.isLeaf) {
        const val = node.featureIndex === 0 ? pt.x : pt.y
        node = val <= node.threshold! ? node.left! : node.right!
      }
      if (node.prediction === pt.label) correct++
    }
    return correct / data.length
  }

  function collectRegions(node: TreeNode | null): BoundaryRegion[] {
    if (!node) return []
    if (node.isLeaf) {
      return [{ xMin: node.xMin, xMax: node.xMax, yMin: node.yMin, yMax: node.yMax, prediction: node.prediction, depth: node.depth }]
    }
    return [...collectRegions(node.left), ...collectRegions(node.right)]
  }

  function collectSplits(node: TreeNode | null): SplitLine[] {
    if (!node || node.isLeaf) return []
    const splits: SplitLine[] = [{
      featureIndex: node.featureIndex!,
      threshold: node.threshold!,
      xMin: node.xMin, xMax: node.xMax, yMin: node.yMin, yMax: node.yMax,
      depth: node.depth,
    }]
    return [...splits, ...collectSplits(node.left), ...collectSplits(node.right)]
  }

  function deepClone(node: TreeNode | null): TreeNode | null {
    if (!node) return null
    return { ...node, left: deepClone(node.left), right: deepClone(node.right) }
  }

  // Initial snapshot: root leaf
  const rootCounts = countClasses(data, allIndices, nClasses)
  const rootPrediction = rootCounts[0] >= rootCounts[1] ? 0 : 1
  const rootNode: TreeNode = {
    id: nodeIdCounter++,
    depth: 0,
    featureIndex: null,
    threshold: null,
    classCounts: rootCounts,
    prediction: rootPrediction,
    left: null,
    right: null,
    samples: allIndices,
    impurity: criterion(rootCounts, allIndices.length),
    isLeaf: true,
    xMin: xExtent[0], xMax: xExtent[1], yMin: yExtent[0], yMax: yExtent[1],
  }

  snapshots.push({
    tree: deepClone(rootNode),
    nodeCount: 1,
    regions: collectRegions(rootNode),
    splits: [],
    accuracy: computeAccuracy(rootNode),
    growingNodeId: rootNode.id,
    description: 'Initial root node',
  })

  // BFS to split nodes
  const queue: TreeNode[] = [rootNode]
  while (queue.length > 0) {
    const node = queue.shift()!
    if (node.depth >= maxDepth) continue
    if (node.samples.length < 4) continue
    if (node.impurity <= 0.001) continue

    const split = findBestSplit(data, node.samples, nClasses, criterion)
    if (!split) continue

    node.featureIndex = split.featureIndex
    node.threshold = split.threshold
    node.isLeaf = false

    const leftCounts = countClasses(data, split.leftIndices, nClasses)
    const rightCounts = countClasses(data, split.rightIndices, nClasses)

    let leftBounds: { xMin: number; xMax: number; yMin: number; yMax: number }
    let rightBounds: { xMin: number; xMax: number; yMin: number; yMax: number }

    if (split.featureIndex === 0) {
      leftBounds = { xMin: node.xMin, xMax: split.threshold, yMin: node.yMin, yMax: node.yMax }
      rightBounds = { xMin: split.threshold, xMax: node.xMax, yMin: node.yMin, yMax: node.yMax }
    } else {
      leftBounds = { xMin: node.xMin, xMax: node.xMax, yMin: node.yMin, yMax: split.threshold }
      rightBounds = { xMin: node.xMin, xMax: node.xMax, yMin: split.threshold, yMax: node.yMax }
    }

    const leftNode: TreeNode = {
      id: nodeIdCounter++,
      depth: node.depth + 1,
      featureIndex: null,
      threshold: null,
      classCounts: leftCounts,
      prediction: leftCounts[0] >= leftCounts[1] ? 0 : 1,
      left: null,
      right: null,
      samples: split.leftIndices,
      impurity: criterion(leftCounts, split.leftIndices.length),
      isLeaf: true,
      ...leftBounds,
    }

    const rightNode: TreeNode = {
      id: nodeIdCounter++,
      depth: node.depth + 1,
      featureIndex: null,
      threshold: null,
      classCounts: rightCounts,
      prediction: rightCounts[0] >= rightCounts[1] ? 0 : 1,
      left: null,
      right: null,
      samples: split.rightIndices,
      impurity: criterion(rightCounts, split.rightIndices.length),
      isLeaf: true,
      ...rightBounds,
    }

    node.left = leftNode
    node.right = rightNode

    const featureLabel = split.featureIndex === 0 ? 'x' : 'y'
    snapshots.push({
      tree: deepClone(rootNode),
      nodeCount: nodeIdCounter,
      regions: collectRegions(rootNode),
      splits: collectSplits(rootNode),
      accuracy: computeAccuracy(rootNode),
      growingNodeId: node.id,
      description: `Split on ${featureLabel} <= ${split.threshold.toFixed(2)} at depth ${node.depth}`,
    })

    queue.push(leftNode)
    queue.push(rightNode)
  }

  // Final snapshot
  snapshots.push({
    tree: deepClone(rootNode),
    nodeCount: nodeIdCounter,
    regions: collectRegions(rootNode),
    splits: collectSplits(rootNode),
    accuracy: computeAccuracy(rootNode),
    growingNodeId: null,
    description: `Complete tree with ${nodeIdCounter} nodes`,
  })

  return snapshots
}

/* ------------------------------------------------------------------ */
/*  Tree diagram layout helper                                        */
/* ------------------------------------------------------------------ */

interface LayoutNode {
  id: number
  x: number
  y: number
  width: number
  height: number
  classCounts: number[]
  prediction: number
  isLeaf: boolean
  featureIndex: number | null
  threshold: number | null
  impurity: number
  samples: number
  parentX?: number
  parentY?: number
  depth: number
}

function layoutTree(tree: TreeNode | null, width: number, height: number): LayoutNode[] {
  if (!tree) return []
  const nodes: LayoutNode[] = []
  const nodeW = 52
  const nodeH = 30
  const verticalGap = 18

  // Get max depth for spacing
  function getMaxDepth(n: TreeNode | null): number {
    if (!n) return 0
    return Math.max(getMaxDepth(n.left), getMaxDepth(n.right)) + 1
  }
  const maxD = getMaxDepth(tree)
  const levelH = maxD > 1 ? Math.min((height - nodeH) / (maxD - 1), nodeH + verticalGap + 20) : 0

  function traverse(node: TreeNode, cx: number, cy: number, spread: number, parentX?: number, parentY?: number) {
    nodes.push({
      id: node.id,
      x: cx - nodeW / 2,
      y: cy,
      width: nodeW,
      height: nodeH,
      classCounts: node.classCounts,
      prediction: node.prediction,
      isLeaf: node.isLeaf,
      featureIndex: node.featureIndex,
      threshold: node.threshold,
      impurity: node.impurity,
      samples: node.samples.length,
      parentX,
      parentY: parentY !== undefined ? parentY + nodeH : undefined,
      depth: node.depth,
    })
    const childSpread = spread / 2
    if (node.left) {
      traverse(node.left, cx - spread, cy + levelH, childSpread, cx, cy)
    }
    if (node.right) {
      traverse(node.right, cx + spread, cy + levelH, childSpread, cx, cy)
    }
  }

  const initialSpread = Math.min(width / 4, 120)
  traverse(tree, width / 2, 5, initialSpread)
  return nodes
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function DecisionTreeViz() {
  const [maxDepth, setMaxDepth] = useState(3)
  const [criterion, setCriterion] = useState<'gini' | 'entropy'>('gini')
  const [noise, setNoise] = useState(0.1)

  const data = useMemo(() => makeNoisyClassification(200, noise, 42), [noise])
  const snapshots = useMemo(() => buildDecisionTreeSnapshots(data, maxDepth, criterion), [data, maxDepth, criterion])
  const player = useAlgorithmPlayer({ snapshots, baseFps: 1.5 })
  const snap = player.currentSnapshot

  const accuracies = useMemo(() => snapshots.map(s => s.accuracy), [snapshots])

  return (
    <GlassCard className="p-8">
      {/* Controls */}
      <div className="flex flex-wrap gap-6 mb-6 items-end">
        <Slider
          label="Max Depth"
          value={maxDepth}
          min={1}
          max={8}
          step={1}
          onChange={(v) => { setMaxDepth(v); player.reset() }}
          className="w-44"
        />
        <Slider
          label="Noise"
          value={noise}
          min={0}
          max={0.3}
          step={0.05}
          onChange={(v) => { setNoise(v); player.reset() }}
          formatValue={(v) => v.toFixed(2)}
          className="w-44"
        />
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">Criterion</span>
          <div className="flex gap-1">
            <Button
              variant="secondary"
              size="sm"
              active={criterion === 'gini'}
              onClick={() => { setCriterion('gini'); player.reset() }}
            >
              Gini
            </Button>
            <Button
              variant="secondary"
              size="sm"
              active={criterion === 'entropy'}
              onClick={() => { setCriterion('entropy'); player.reset() }}
            >
              Entropy
            </Button>
          </div>
        </div>
      </div>

      {/* Dual panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: scatter plot with decision boundaries */}
        <div className="bg-obsidian-surface/40 rounded-lg border border-obsidian-border p-2">
          <div className="text-xs text-text-tertiary uppercase tracking-wider mb-2 px-1">Decision Boundaries</div>
          <SVGContainer aspectRatio={1} minHeight={300} maxHeight={450} padding={{ top: 15, right: 15, bottom: 30, left: 40 }}>
            {({ innerWidth, innerHeight }) => {
              const xExtent = d3.extent(data, d => d.x) as [number, number]
              const yExtent = d3.extent(data, d => d.y) as [number, number]
              const xPad = (xExtent[1] - xExtent[0]) * 0.05
              const yPad = (yExtent[1] - yExtent[0]) * 0.05
              const xScale = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([0, innerWidth]).nice()
              const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([innerHeight, 0]).nice()

              return (
                <>
                  {/* Boundary regions */}
                  {snap.regions.map((r, i) => {
                    const rx = xScale(Math.max(r.xMin, xScale.domain()[0]))
                    const ry = yScale(Math.min(r.yMax, yScale.domain()[1]))
                    const rw = xScale(Math.min(r.xMax, xScale.domain()[1])) - rx
                    const rh = yScale(Math.max(r.yMin, yScale.domain()[0])) - ry
                    return (
                      <motion.rect
                        key={`region-${i}`}
                        x={Math.max(0, rx)}
                        y={Math.max(0, ry)}
                        width={Math.max(0, Math.min(rw, innerWidth))}
                        height={Math.max(0, Math.min(rh, innerHeight))}
                        fill={COLORS.clusters[r.prediction % COLORS.clusters.length]}
                        fillOpacity={0.08 + r.depth * 0.02}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.4 }}
                      />
                    )
                  })}

                  {/* Split lines */}
                  {snap.splits.map((s, i) => {
                    if (s.featureIndex === 0) {
                      const sx = xScale(s.threshold)
                      const sy1 = yScale(Math.min(s.yMax, yScale.domain()[1]))
                      const sy2 = yScale(Math.max(s.yMin, yScale.domain()[0]))
                      return (
                        <motion.line
                          key={`split-${i}`}
                          x1={sx} y1={sy1} x2={sx} y2={sy2}
                          stroke="rgba(255,255,255,0.5)"
                          strokeWidth={1.5}
                          strokeDasharray="4,3"
                          initial={{ pathLength: 0, opacity: 0 }}
                          animate={{ pathLength: 1, opacity: 1 }}
                          transition={{ duration: 0.5 }}
                        />
                      )
                    } else {
                      const sy = yScale(s.threshold)
                      const sx1 = xScale(Math.max(s.xMin, xScale.domain()[0]))
                      const sx2 = xScale(Math.min(s.xMax, xScale.domain()[1]))
                      return (
                        <motion.line
                          key={`split-${i}`}
                          x1={sx1} y1={sy} x2={sx2} y2={sy}
                          stroke="rgba(255,255,255,0.5)"
                          strokeWidth={1.5}
                          strokeDasharray="4,3"
                          initial={{ pathLength: 0, opacity: 0 }}
                          animate={{ pathLength: 1, opacity: 1 }}
                          transition={{ duration: 0.5 }}
                        />
                      )
                    }
                  })}

                  {/* Data points */}
                  {data.map((pt, i) => (
                    <motion.circle
                      key={`pt-${i}`}
                      cx={xScale(pt.x)}
                      cy={yScale(pt.y)}
                      r={3}
                      fill={COLORS.clusters[pt.label % COLORS.clusters.length]}
                      fillOpacity={0.75}
                      stroke={COLORS.clusters[pt.label % COLORS.clusters.length]}
                      strokeWidth={0.5}
                      strokeOpacity={0.3}
                    />
                  ))}

                  {/* X axis */}
                  {xScale.ticks(5).map(tick => (
                    <text key={`xt-${tick}`} x={xScale(tick)} y={innerHeight + 18} textAnchor="middle" className="text-[9px] fill-text-tertiary">
                      {tick.toFixed(1)}
                    </text>
                  ))}
                  {/* Y axis */}
                  {yScale.ticks(5).map(tick => (
                    <text key={`yt-${tick}`} x={-8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" className="text-[9px] fill-text-tertiary">
                      {tick.toFixed(1)}
                    </text>
                  ))}
                </>
              )
            }}
          </SVGContainer>
        </div>

        {/* Right: tree diagram */}
        <div className="bg-obsidian-surface/40 rounded-lg border border-obsidian-border p-2">
          <div className="text-xs text-text-tertiary uppercase tracking-wider mb-2 px-1">Tree Structure</div>
          <SVGContainer aspectRatio={1} minHeight={300} maxHeight={450} padding={{ top: 10, right: 10, bottom: 10, left: 10 }}>
            {({ innerWidth, innerHeight }) => {
              const layoutNodes = layoutTree(snap.tree, innerWidth, innerHeight)
              return (
                <>
                  {/* Connection lines */}
                  {layoutNodes.filter(n => n.parentX !== undefined).map(n => (
                    <motion.line
                      key={`edge-${n.id}`}
                      x1={n.parentX!}
                      y1={n.parentY!}
                      x2={n.x + n.width / 2}
                      y2={n.y}
                      stroke="rgba(255,255,255,0.2)"
                      strokeWidth={1}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.4 }}
                    />
                  ))}

                  {/* Nodes */}
                  {layoutNodes.map(n => {
                    const isGrowing = n.id === snap.growingNodeId
                    const totalSamples = n.classCounts.reduce((a, b) => a + b, 0)
                    const ratio0 = totalSamples > 0 ? n.classCounts[0] / totalSamples : 0
                    const barWidth = n.width - 8
                    return (
                      <motion.g
                        key={`node-${n.id}`}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                      >
                        {/* Pulsing glow for growing node */}
                        {isGrowing && (
                          <motion.rect
                            x={n.x - 3}
                            y={n.y - 3}
                            width={n.width + 6}
                            height={n.height + 6}
                            rx={8}
                            fill="none"
                            stroke={COLORS.accent}
                            strokeWidth={2}
                            animate={{ opacity: [0.3, 0.8, 0.3] }}
                            transition={{ duration: 1.5, repeat: Infinity }}
                          />
                        )}
                        {/* Node rect */}
                        <rect
                          x={n.x}
                          y={n.y}
                          width={n.width}
                          height={n.height}
                          rx={6}
                          fill={n.isLeaf ? 'rgba(15,15,17,0.9)' : 'rgba(25,25,35,0.9)'}
                          stroke={isGrowing ? COLORS.accent : 'rgba(255,255,255,0.12)'}
                          strokeWidth={isGrowing ? 1.5 : 0.8}
                        />
                        {/* Mini class distribution bar */}
                        <rect
                          x={n.x + 4}
                          y={n.y + n.height - 10}
                          width={barWidth * ratio0}
                          height={5}
                          rx={1.5}
                          fill={COLORS.clusters[0]}
                          fillOpacity={0.8}
                        />
                        <rect
                          x={n.x + 4 + barWidth * ratio0}
                          y={n.y + n.height - 10}
                          width={barWidth * (1 - ratio0)}
                          height={5}
                          rx={1.5}
                          fill={COLORS.clusters[1]}
                          fillOpacity={0.8}
                        />
                        {/* Samples count text */}
                        <text
                          x={n.x + n.width / 2}
                          y={n.y + 13}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          className="text-[8px] fill-text-secondary"
                        >
                          {n.isLeaf ? `n=${n.samples}` : `${n.featureIndex === 0 ? 'x' : 'y'}<=${n.threshold?.toFixed(1)}`}
                        </text>
                      </motion.g>
                    )
                  })}
                </>
              )
            }}
          </SVGContainer>
        </div>
      </div>

      {/* Transport + info */}
      <div className="mt-4 flex flex-wrap items-start gap-4">
        <TransportControls
          isPlaying={player.isPlaying}
          isAtStart={player.isAtStart}
          isAtEnd={player.isAtEnd}
          currentStep={player.currentStep}
          totalSteps={player.totalSteps}
          speed={player.speed}
          onPlay={player.play}
          onPause={player.pause}
          onTogglePlay={player.togglePlay}
          onStepForward={player.stepForward}
          onStepBack={player.stepBack}
          onReset={player.reset}
          onSetSpeed={player.setSpeed}
        />
        <ConvergenceChart
          values={accuracies}
          currentIndex={player.currentStep}
          label="Accuracy"
          width={200}
          height={80}
          color={COLORS.success}
        />
        <div className="text-xs text-text-tertiary self-center space-y-1">
          <div>{snap.description}</div>
          <div className="text-text-secondary font-mono">
            Accuracy: <span className="text-success">{(snap.accuracy * 100).toFixed(1)}%</span>
            {' | '}Nodes: {snap.nodeCount}
          </div>
        </div>
      </div>
    </GlassCard>
  )
}
