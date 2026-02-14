import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'
import { COLORS } from '../../../types'
import {
  makeNoisyClassification,
  makeMoonsClassification,
  makeTreeFriendly,
  type ClassificationPoint,
} from '../../../lib/data/treeDataGenerators'
import { createRng } from '../../../lib/math/random'

/* ------------------------------------------------------------------ */
/*  Self-contained simplified algorithms for classification            */
/* ------------------------------------------------------------------ */

interface SimpleNode {
  featureIndex: number | null
  threshold: number | null
  prediction: number
  left: SimpleNode | null
  right: SimpleNode | null
  isLeaf: boolean
}

// --- Decision Tree ---

function gini(counts: number[], total: number): number {
  if (total === 0) return 0
  let sum = 0
  for (const c of counts) {
    const p = c / total
    sum += p * p
  }
  return 1 - sum
}

function buildTree(
  data: ClassificationPoint[],
  indices: number[],
  maxDepth: number,
  depth: number = 0,
): SimpleNode {
  const counts = [0, 0]
  for (const i of indices) counts[data[i].label]++
  const prediction = counts[0] >= counts[1] ? 0 : 1
  const total = indices.length

  if (depth >= maxDepth || total < 4 || gini(counts, total) <= 0.001) {
    return { featureIndex: null, threshold: null, prediction, left: null, right: null, isLeaf: true }
  }

  let bestGain = 0
  let bestSplit: { featureIndex: number; threshold: number; leftIdx: number[]; rightIdx: number[] } | null = null
  const parentImp = gini(counts, total)

  for (let feat = 0; feat < 2; feat++) {
    const values = indices.map(i => feat === 0 ? data[i].x : data[i].y)
    const sorted = [...new Set(values)].sort((a, b) => a - b)
    const step = Math.max(1, Math.floor((sorted.length - 1) / 15))

    for (let ti = 0; ti < sorted.length - 1; ti += step) {
      const thresh = (sorted[ti] + sorted[ti + 1]) / 2
      const leftIdx: number[] = []
      const rightIdx: number[] = []
      for (const i of indices) {
        const val = feat === 0 ? data[i].x : data[i].y
        if (val <= thresh) leftIdx.push(i)
        else rightIdx.push(i)
      }
      if (leftIdx.length === 0 || rightIdx.length === 0) continue

      const lc = [0, 0], rc = [0, 0]
      for (const i of leftIdx) lc[data[i].label]++
      for (const i of rightIdx) rc[data[i].label]++
      const gain = parentImp
        - (leftIdx.length / total) * gini(lc, leftIdx.length)
        - (rightIdx.length / total) * gini(rc, rightIdx.length)

      if (gain > bestGain) {
        bestGain = gain
        bestSplit = { featureIndex: feat, threshold: thresh, leftIdx, rightIdx }
      }
    }
  }

  if (!bestSplit) {
    return { featureIndex: null, threshold: null, prediction, left: null, right: null, isLeaf: true }
  }

  return {
    featureIndex: bestSplit.featureIndex,
    threshold: bestSplit.threshold,
    prediction,
    left: buildTree(data, bestSplit.leftIdx, maxDepth, depth + 1),
    right: buildTree(data, bestSplit.rightIdx, maxDepth, depth + 1),
    isLeaf: false,
  }
}

function predictNode(node: SimpleNode, x: number, y: number): number {
  if (node.isLeaf) return node.prediction
  const val = node.featureIndex === 0 ? x : y
  return val <= node.threshold! ? predictNode(node.left!, x, y) : predictNode(node.right!, x, y)
}

// --- Random Forest ---

function buildForest(
  data: ClassificationPoint[],
  nTrees: number,
  maxDepth: number,
  seed: number = 123,
): SimpleNode[] {
  const rng = createRng(seed)
  const trees: SimpleNode[] = []

  for (let t = 0; t < nTrees; t++) {
    // Bootstrap
    const bootstrap: number[] = []
    for (let i = 0; i < data.length; i++) {
      bootstrap.push(Math.floor(rng() * data.length))
    }
    trees.push(buildTree(data, bootstrap, maxDepth))
  }

  return trees
}

function predictForest(trees: SimpleNode[], x: number, y: number): number {
  const votes = [0, 0]
  for (const t of trees) votes[predictNode(t, x, y)]++
  return votes[0] >= votes[1] ? 0 : 1
}

// --- Gradient Boosted (classification via soft labels) ---

function buildGBTClassifier(
  data: ClassificationPoint[],
  nEstimators: number,
  learningRate: number,
  maxDepth: number,
): { trees: SimpleNode[]; lr: number } {
  // Simple approach: convert to +-1, fit regression stumps, threshold at 0
  const n = data.length
  const targets = data.map(d => d.label === 1 ? 1 : -1)
  const scores = new Array(n).fill(0)
  const stumps: SimpleNode[] = []

  for (let t = 0; t < nEstimators; t++) {
    // Compute pseudo-residuals (simplified gradient)
    const residuals = targets.map((y, i) => y - Math.tanh(scores[i]))

    // Build regression stump by converting residuals to binary
    const medianRes = [...residuals].sort((a, b) => a - b)[Math.floor(n / 2)]
    const binaryResiduals: ClassificationPoint[] = data.map((d, i) => ({
      x: d.x, y: d.y, label: residuals[i] > medianRes ? 1 : 0,
    }))
    const allIdx = data.map((_, i) => i)
    const stump = buildTree(binaryResiduals, allIdx, maxDepth)

    // Update scores using the stump predictions
    for (let i = 0; i < n; i++) {
      const pred = predictNode(stump, data[i].x, data[i].y)
      scores[i] += learningRate * (pred === 1 ? 1 : -1) * Math.abs(residuals[i])
    }

    stumps.push(stump)
  }

  return { trees: stumps, lr: learningRate }
}

function predictGBT(model: { trees: SimpleNode[]; lr: number }, data: ClassificationPoint[], x: number, y: number): number {
  let score = 0
  const n = data.length
  const targets = data.map(d => d.label === 1 ? 1 : -1)

  // Replay forward pass
  const scores = new Array(n).fill(0)
  for (let t = 0; t < model.trees.length; t++) {
    const residuals = targets.map((yt, i) => yt - Math.tanh(scores[i]))
    const pred = predictNode(model.trees[t], x, y)
    // Approximate: use the stump prediction direction
    score += model.lr * (pred === 1 ? 0.5 : -0.5)

    for (let i = 0; i < n; i++) {
      const pi = predictNode(model.trees[t], data[i].x, data[i].y)
      scores[i] += model.lr * (pi === 1 ? 1 : -1) * Math.abs(residuals[i])
    }
  }

  return score > 0 ? 1 : 0
}

// --- Grid computation ---

interface ModelResult {
  grid: number[][]
  trainAccuracy: number
  testAccuracy: number
}

const GRID_RES = 40

function computeGrid(
  predictFn: (x: number, y: number) => number,
  xMin: number, xMax: number, yMin: number, yMax: number,
  trainData: ClassificationPoint[],
  testData: ClassificationPoint[],
): ModelResult {
  const grid: number[][] = []
  for (let gy = 0; gy < GRID_RES; gy++) {
    grid[gy] = []
    for (let gx = 0; gx < GRID_RES; gx++) {
      const px = xMin + (gx / (GRID_RES - 1)) * (xMax - xMin)
      const py = yMin + (gy / (GRID_RES - 1)) * (yMax - yMin)
      grid[gy][gx] = predictFn(px, py)
    }
  }

  let trainCorrect = 0
  for (const pt of trainData) {
    if (predictFn(pt.x, pt.y) === pt.label) trainCorrect++
  }

  let testCorrect = 0
  for (const pt of testData) {
    if (predictFn(pt.x, pt.y) === pt.label) testCorrect++
  }

  return {
    grid,
    trainAccuracy: trainData.length > 0 ? trainCorrect / trainData.length : 0,
    testAccuracy: testData.length > 0 ? testCorrect / testData.length : 0,
  }
}

/* ------------------------------------------------------------------ */
/*  Dataset selector                                                   */
/* ------------------------------------------------------------------ */

type DatasetType = 'clean' | 'noisy' | 'moons'

function generateDataset(type: DatasetType): { train: ClassificationPoint[]; test: ClassificationPoint[] } {
  let train: ClassificationPoint[]
  let test: ClassificationPoint[]

  switch (type) {
    case 'clean':
      train = makeTreeFriendly(200, 3, 42)
      test = makeTreeFriendly(60, 3, 99)
      break
    case 'noisy':
      train = makeNoisyClassification(200, 0.2, 42)
      test = makeNoisyClassification(60, 0.2, 99)
      break
    case 'moons':
      train = makeMoonsClassification(200, 0.15, 42)
      test = makeMoonsClassification(60, 0.15, 99)
      break
  }

  return { train, test }
}

/* ------------------------------------------------------------------ */
/*  Individual panel component                                         */
/* ------------------------------------------------------------------ */

interface PanelProps {
  title: string
  grid: number[][]
  trainData: ClassificationPoint[]
  testData: ClassificationPoint[]
  trainAccuracy: number
  testAccuracy: number
  xMin: number; xMax: number; yMin: number; yMax: number
  highlight?: boolean
}

function ComparisonPanel({ title, grid, trainData, testData, trainAccuracy, testAccuracy, xMin, xMax, yMin, yMax }: PanelProps) {
  return (
    <div className="bg-obsidian-surface/40 rounded-lg border border-obsidian-border p-2 flex flex-col">
      <div className="text-xs text-text-tertiary uppercase tracking-wider mb-1 px-1">{title}</div>
      <SVGContainer aspectRatio={1} minHeight={200} maxHeight={350} padding={{ top: 10, right: 10, bottom: 25, left: 30 }}>
        {({ innerWidth, innerHeight }) => {
          const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerWidth])
          const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerHeight, 0])
          const cellW = innerWidth / GRID_RES
          const cellH = innerHeight / GRID_RES

          return (
            <>
              {/* Decision boundary grid */}
              {grid.map((row, gy) =>
                row.map((pred, gx) => (
                  <rect
                    key={`c-${gy}-${gx}`}
                    x={gx * cellW}
                    y={gy * cellH}
                    width={cellW + 0.5}
                    height={cellH + 0.5}
                    fill={COLORS.clusters[pred % COLORS.clusters.length]}
                    fillOpacity={0.1}
                  />
                ))
              )}

              {/* Training data points */}
              {trainData.map((pt, i) => (
                <motion.circle
                  key={`tr-${i}`}
                  cx={xScale(pt.x)}
                  cy={yScale(pt.y)}
                  r={2.5}
                  fill={COLORS.clusters[pt.label % COLORS.clusters.length]}
                  fillOpacity={0.75}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3, delay: i * 0.002 }}
                />
              ))}

              {/* Test data points (diamonds/squares) */}
              {testData.map((pt, i) => (
                <motion.rect
                  key={`te-${i}`}
                  x={xScale(pt.x) - 2.5}
                  y={yScale(pt.y) - 2.5}
                  width={5}
                  height={5}
                  rx={1}
                  fill="none"
                  stroke={COLORS.clusters[pt.label % COLORS.clusters.length]}
                  strokeWidth={1.2}
                  strokeOpacity={0.8}
                  transform={`rotate(45, ${xScale(pt.x)}, ${yScale(pt.y)})`}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3, delay: 0.2 + i * 0.003 }}
                />
              ))}

              {/* Axes */}
              {xScale.ticks(4).map(tick => (
                <text key={`xt-${tick}`} x={xScale(tick)} y={innerHeight + 16} textAnchor="middle" className="text-[8px] fill-text-tertiary">
                  {tick.toFixed(1)}
                </text>
              ))}
              {yScale.ticks(4).map(tick => (
                <text key={`yt-${tick}`} x={-6} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" className="text-[8px] fill-text-tertiary">
                  {tick.toFixed(1)}
                </text>
              ))}
            </>
          )
        }}
      </SVGContainer>
      {/* Accuracy badges */}
      <div className="flex gap-3 mt-2 px-1">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.success }} />
          <span className="text-[10px] font-mono text-text-tertiary">
            Train: <span className="text-text-secondary">{(trainAccuracy * 100).toFixed(1)}%</span>
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: COLORS.accent }} />
          <span className="text-[10px] font-mono text-text-tertiary">
            Test: <span className="text-text-secondary">{(testAccuracy * 100).toFixed(1)}%</span>
          </span>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

export function TreeComparisonViz() {
  const [dataset, setDataset] = useState<DatasetType>('moons')
  const [dtDepth, setDtDepth] = useState(5)
  const [rfTrees, setRfTrees] = useState(10)
  const [rfDepth, setRfDepth] = useState(3)
  const [gbtTrees, setGbtTrees] = useState(10)
  const [gbtLr, setGbtLr] = useState(0.3)
  const [gbtDepth, setGbtDepth] = useState(2)

  const { train, test } = useMemo(() => generateDataset(dataset), [dataset])

  // Compute bounds from train data
  const allPts = useMemo(() => [...train, ...test], [train, test])
  const xExtent = useMemo(() => d3.extent(allPts, d => d.x) as [number, number], [allPts])
  const yExtent = useMemo(() => d3.extent(allPts, d => d.y) as [number, number], [allPts])
  const xPad = (xExtent[1] - xExtent[0]) * 0.1
  const yPad = (yExtent[1] - yExtent[0]) * 0.1
  const xMin = xExtent[0] - xPad
  const xMax = xExtent[1] + xPad
  const yMin = yExtent[0] - yPad
  const yMax = yExtent[1] + yPad

  // Decision Tree
  const dtResult = useMemo(() => {
    const allIdx = train.map((_, i) => i)
    const tree = buildTree(train, allIdx, dtDepth)
    return computeGrid(
      (x, y) => predictNode(tree, x, y),
      xMin, xMax, yMin, yMax, train, test,
    )
  }, [train, test, dtDepth, xMin, xMax, yMin, yMax])

  // Random Forest
  const rfResult = useMemo(() => {
    const trees = buildForest(train, rfTrees, rfDepth)
    return computeGrid(
      (x, y) => predictForest(trees, x, y),
      xMin, xMax, yMin, yMax, train, test,
    )
  }, [train, test, rfTrees, rfDepth, xMin, xMax, yMin, yMax])

  // GBT
  const gbtResult = useMemo(() => {
    const model = buildGBTClassifier(train, gbtTrees, gbtLr, gbtDepth)
    return computeGrid(
      (x, y) => predictGBT(model, train, x, y),
      xMin, xMax, yMin, yMax, train, test,
    )
  }, [train, test, gbtTrees, gbtLr, gbtDepth, xMin, xMax, yMin, yMax])

  return (
    <GlassCard className="p-8">
      {/* Dataset selector + global controls */}
      <div className="flex flex-wrap gap-4 mb-6 items-end">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">Dataset</span>
          <div className="flex gap-1">
            {(['clean', 'noisy', 'moons'] as const).map(d => (
              <Button
                key={d}
                variant="secondary"
                size="sm"
                active={dataset === d}
                onClick={() => setDataset(d)}
              >
                {d.charAt(0).toUpperCase() + d.slice(1)}
              </Button>
            ))}
          </div>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-3 text-[10px] text-text-tertiary">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-white/40" />
            <span>Train points (circles)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rotate-45 border border-white/40" />
            <span>Test points (diamonds)</span>
          </div>
        </div>
      </div>

      {/* Per-model controls row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <div className="flex flex-wrap gap-3">
          <Slider label="DT Depth" value={dtDepth} min={1} max={8} step={1} onChange={setDtDepth} className="w-full" />
        </div>
        <div className="flex flex-wrap gap-3">
          <Slider label="RF Trees" value={rfTrees} min={1} max={20} step={1} onChange={setRfTrees} className="flex-1 min-w-[100px]" />
          <Slider label="RF Depth" value={rfDepth} min={1} max={5} step={1} onChange={setRfDepth} className="flex-1 min-w-[100px]" />
        </div>
        <div className="flex flex-wrap gap-3">
          <Slider label="GBT Trees" value={gbtTrees} min={1} max={20} step={1} onChange={setGbtTrees} className="flex-1 min-w-[80px]" />
          <Slider label="GBT LR" value={gbtLr} min={0.05} max={1} step={0.05} onChange={setGbtLr}
            formatValue={v => v.toFixed(2)} className="flex-1 min-w-[80px]" />
          <Slider label="GBT Depth" value={gbtDepth} min={1} max={3} step={1} onChange={setGbtDepth} className="flex-1 min-w-[80px]" />
        </div>
      </div>

      {/* Three comparison panels */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ComparisonPanel
          title="Decision Tree"
          grid={dtResult.grid}
          trainData={train}
          testData={test}
          trainAccuracy={dtResult.trainAccuracy}
          testAccuracy={dtResult.testAccuracy}
          xMin={xMin} xMax={xMax} yMin={yMin} yMax={yMax}
        />
        <ComparisonPanel
          title="Random Forest"
          grid={rfResult.grid}
          trainData={train}
          testData={test}
          trainAccuracy={rfResult.trainAccuracy}
          testAccuracy={rfResult.testAccuracy}
          xMin={xMin} xMax={xMax} yMin={yMin} yMax={yMax}
        />
        <ComparisonPanel
          title="Gradient Boosted"
          grid={gbtResult.grid}
          trainData={train}
          testData={test}
          trainAccuracy={gbtResult.trainAccuracy}
          testAccuracy={gbtResult.testAccuracy}
          xMin={xMin} xMax={xMax} yMin={yMin} yMax={yMax}
        />
      </div>

      {/* Summary comparison table */}
      <div className="mt-4 bg-obsidian-surface/40 rounded-lg border border-obsidian-border overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-obsidian-border">
              <th className="text-left text-text-tertiary font-medium uppercase tracking-wider py-2 px-3">Model</th>
              <th className="text-center text-text-tertiary font-medium uppercase tracking-wider py-2 px-3">Train Accuracy</th>
              <th className="text-center text-text-tertiary font-medium uppercase tracking-wider py-2 px-3">Test Accuracy</th>
              <th className="text-center text-text-tertiary font-medium uppercase tracking-wider py-2 px-3">Gap</th>
            </tr>
          </thead>
          <tbody>
            {[
              { name: 'Decision Tree', train: dtResult.trainAccuracy, test: dtResult.testAccuracy },
              { name: 'Random Forest', train: rfResult.trainAccuracy, test: rfResult.testAccuracy },
              { name: 'Gradient Boosted', train: gbtResult.trainAccuracy, test: gbtResult.testAccuracy },
            ].map((row, i) => {
              const gap = row.train - row.test
              const gapColor = gap > 0.15 ? COLORS.error : gap > 0.05 ? COLORS.clusters[3] : COLORS.success
              return (
                <tr key={row.name} className={i < 2 ? 'border-b border-obsidian-border/50' : ''}>
                  <td className="py-2 px-3 font-mono text-text-secondary">{row.name}</td>
                  <td className="py-2 px-3 text-center font-mono text-text-secondary">{(row.train * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-center font-mono text-text-secondary">{(row.test * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-center font-mono" style={{ color: gapColor }}>
                    {gap > 0 ? '+' : ''}{(gap * 100).toFixed(1)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </GlassCard>
  )
}
