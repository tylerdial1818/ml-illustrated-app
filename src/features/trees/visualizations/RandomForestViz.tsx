import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { ConvergenceChart } from '../../../components/viz/ConvergenceChart'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { COLORS } from '../../../types'
import { makeMoonsClassification, type ClassificationPoint } from '../../../lib/data/treeDataGenerators'
import { createRng } from '../../../lib/math/random'

/* ------------------------------------------------------------------ */
/*  Self-contained simplified random forest algorithm                 */
/* ------------------------------------------------------------------ */

interface SimpleTreeNode {
  featureIndex: number | null
  threshold: number | null
  prediction: number
  left: SimpleTreeNode | null
  right: SimpleTreeNode | null
  isLeaf: boolean
}

interface BoundaryCell {
  xMin: number; xMax: number; yMin: number; yMax: number
  prediction: number
}

interface TreeInfo {
  root: SimpleTreeNode
  bootstrapIndices: number[]
  boundaries: BoundaryCell[]
  featureImportance: [number, number]  // [x_importance, y_importance]
}

interface RFSnapshot {
  trees: TreeInfo[]
  nTreesBuilt: number
  aggregatedGrid: number[][]  // prediction grid for aggregated view
  aggregatedAccuracy: number
  featureImportance: [number, number]
  description: string
}

function gini(counts: number[], total: number): number {
  if (total === 0) return 0
  let sum = 0
  for (const c of counts) {
    const p = c / total
    sum += p * p
  }
  return 1 - sum
}

function buildSimpleTree(
  data: ClassificationPoint[],
  indices: number[],
  maxDepth: number,
  rng: () => number,
  xMin: number, xMax: number, yMin: number, yMax: number,
  depth: number = 0,
  importance: [number, number] = [0, 0],
): { node: SimpleTreeNode; boundaries: BoundaryCell[]; importance: [number, number] } {
  const counts = [0, 0]
  for (const i of indices) counts[data[i].label]++
  const prediction = counts[0] >= counts[1] ? 0 : 1
  const total = indices.length

  if (depth >= maxDepth || total < 4 || gini(counts, total) <= 0.001) {
    return {
      node: { featureIndex: null, threshold: null, prediction, left: null, right: null, isLeaf: true },
      boundaries: [{ xMin, xMax, yMin, yMax, prediction }],
      importance,
    }
  }

  // Random feature subset (select 1 feature randomly out of 2 with 50% chance, or both)
  const useBoth = rng() > 0.3
  const featuresToTry = useBoth ? [0, 1] : [rng() > 0.5 ? 0 : 1]

  let bestGain = 0
  let bestSplit: { featureIndex: number; threshold: number; leftIdx: number[]; rightIdx: number[] } | null = null

  for (const feat of featuresToTry) {
    const values = indices.map(i => feat === 0 ? data[i].x : data[i].y)
    const sorted = [...new Set(values)].sort((a, b) => a - b)
    const nCandidates = Math.min(sorted.length - 1, 10)
    const step = Math.max(1, Math.floor((sorted.length - 1) / nCandidates))

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

      const leftCounts = [0, 0]
      const rightCounts = [0, 0]
      for (const i of leftIdx) leftCounts[data[i].label]++
      for (const i of rightIdx) rightCounts[data[i].label]++

      const parentGini = gini(counts, total)
      const childGini = (leftIdx.length / total) * gini(leftCounts, leftIdx.length)
        + (rightIdx.length / total) * gini(rightCounts, rightIdx.length)
      const gain = parentGini - childGini

      if (gain > bestGain) {
        bestGain = gain
        bestSplit = { featureIndex: feat, threshold: thresh, leftIdx, rightIdx }
      }
    }
  }

  if (!bestSplit) {
    return {
      node: { featureIndex: null, threshold: null, prediction, left: null, right: null, isLeaf: true },
      boundaries: [{ xMin, xMax, yMin, yMax, prediction }],
      importance,
    }
  }

  importance[bestSplit.featureIndex] += bestGain * total

  let leftBounds: { xMin: number; xMax: number; yMin: number; yMax: number }
  let rightBounds: { xMin: number; xMax: number; yMin: number; yMax: number }
  if (bestSplit.featureIndex === 0) {
    leftBounds = { xMin, xMax: bestSplit.threshold, yMin, yMax }
    rightBounds = { xMin: bestSplit.threshold, xMax, yMin, yMax }
  } else {
    leftBounds = { xMin, xMax, yMin, yMax: bestSplit.threshold }
    rightBounds = { xMin, xMax, yMin: bestSplit.threshold, yMax }
  }

  const leftResult = buildSimpleTree(data, bestSplit.leftIdx, maxDepth, rng, leftBounds.xMin, leftBounds.xMax, leftBounds.yMin, leftBounds.yMax, depth + 1, importance)
  const rightResult = buildSimpleTree(data, bestSplit.rightIdx, maxDepth, rng, rightBounds.xMin, rightBounds.xMax, rightBounds.yMin, rightBounds.yMax, depth + 1, importance)

  return {
    node: {
      featureIndex: bestSplit.featureIndex,
      threshold: bestSplit.threshold,
      prediction,
      left: leftResult.node,
      right: rightResult.node,
      isLeaf: false,
    },
    boundaries: [...leftResult.boundaries, ...rightResult.boundaries],
    importance: leftResult.importance,  // accumulated in-place
  }
}

function predictTree(node: SimpleTreeNode, x: number, y: number): number {
  if (node.isLeaf) return node.prediction
  const val = node.featureIndex === 0 ? x : y
  return val <= node.threshold! ? predictTree(node.left!, x, y) : predictTree(node.right!, x, y)
}

function runRandomForest(
  data: ClassificationPoint[],
  nEstimators: number,
  maxDepth: number,
): RFSnapshot[] {
  const snapshots: RFSnapshot[] = []
  const rng = createRng(123)
  const xExtent = d3.extent(data, d => d.x) as [number, number]
  const yExtent = d3.extent(data, d => d.y) as [number, number]
  const xPad = (xExtent[1] - xExtent[0]) * 0.1
  const yPad = (yExtent[1] - yExtent[0]) * 0.1
  const xMin = xExtent[0] - xPad
  const xMax = xExtent[1] + xPad
  const yMin = yExtent[0] - yPad
  const yMax = yExtent[1] + yPad

  const GRID_RES = 30
  const trees: TreeInfo[] = []

  // Compute aggregated grid
  function computeAggregatedGrid(currentTrees: TreeInfo[]): number[][] {
    const grid: number[][] = []
    for (let gy = 0; gy < GRID_RES; gy++) {
      grid[gy] = []
      for (let gx = 0; gx < GRID_RES; gx++) {
        const px = xMin + (gx / (GRID_RES - 1)) * (xMax - xMin)
        const py = yMin + (gy / (GRID_RES - 1)) * (yMax - yMin)
        let votes = [0, 0]
        for (const t of currentTrees) {
          const pred = predictTree(t.root, px, py)
          votes[pred]++
        }
        grid[gy][gx] = votes[0] >= votes[1] ? 0 : 1
      }
    }
    return grid
  }

  function computeAccuracy(currentTrees: TreeInfo[]): number {
    if (currentTrees.length === 0) return 0
    let correct = 0
    for (const pt of data) {
      const votes = [0, 0]
      for (const t of currentTrees) {
        votes[predictTree(t.root, pt.x, pt.y)]++
      }
      if ((votes[0] >= votes[1] ? 0 : 1) === pt.label) correct++
    }
    return correct / data.length
  }

  // Initial empty snapshot
  snapshots.push({
    trees: [],
    nTreesBuilt: 0,
    aggregatedGrid: Array.from({ length: GRID_RES }, () => new Array(GRID_RES).fill(0)),
    aggregatedAccuracy: 0,
    featureImportance: [0, 0],
    description: 'Ready to build forest',
  })

  // Build trees one at a time
  for (let t = 0; t < nEstimators; t++) {
    // Bootstrap sample
    const bootstrapIndices: number[] = []
    for (let i = 0; i < data.length; i++) {
      bootstrapIndices.push(Math.floor(rng() * data.length))
    }

    const result = buildSimpleTree(data, bootstrapIndices, maxDepth, rng, xMin, xMax, yMin, yMax)
    const totalImp = result.importance[0] + result.importance[1]
    const normImp: [number, number] = totalImp > 0
      ? [result.importance[0] / totalImp, result.importance[1] / totalImp]
      : [0.5, 0.5]

    const treeInfo: TreeInfo = {
      root: result.node,
      bootstrapIndices,
      boundaries: result.boundaries,
      featureImportance: normImp,
    }
    trees.push(treeInfo)

    // Average feature importance across all trees so far
    const avgImp: [number, number] = [0, 0]
    for (const ti of trees) {
      avgImp[0] += ti.featureImportance[0]
      avgImp[1] += ti.featureImportance[1]
    }
    avgImp[0] /= trees.length
    avgImp[1] /= trees.length

    snapshots.push({
      trees: trees.map(ti => ({ ...ti })),
      nTreesBuilt: t + 1,
      aggregatedGrid: computeAggregatedGrid(trees),
      aggregatedAccuracy: computeAccuracy(trees),
      featureImportance: avgImp,
      description: `Built tree ${t + 1} of ${nEstimators}`,
    })
  }

  return snapshots
}

/* ------------------------------------------------------------------ */
/*  Mini scatter plot for individual tree                              */
/* ------------------------------------------------------------------ */

interface MiniTreePlotProps {
  data: ClassificationPoint[]
  boundaries: BoundaryCell[]
  bootstrapIndices: number[]
  width: number
  height: number
  xScale: d3.ScaleLinear<number, number>
  yScale: d3.ScaleLinear<number, number>
  treeIndex: number
  isNew: boolean
}

function MiniTreePlot({ data, boundaries, bootstrapIndices, width, height, xScale, yScale, treeIndex, isNew }: MiniTreePlotProps) {
  const bootstrapSet = useMemo(() => new Set(bootstrapIndices), [bootstrapIndices])

  return (
    <motion.g
      initial={isNew ? { opacity: 0, scale: 0.9 } : false}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
    >
      {/* Boundary regions */}
      {boundaries.map((b, i) => {
        const rx = xScale(b.xMin)
        const ry = yScale(b.yMax)
        const rw = xScale(b.xMax) - rx
        const rh = yScale(b.yMin) - ry
        return (
          <rect
            key={`b-${i}`}
            x={rx} y={ry} width={Math.max(0, rw)} height={Math.max(0, rh)}
            fill={COLORS.clusters[b.prediction % COLORS.clusters.length]}
            fillOpacity={0.12}
          />
        )
      })}

      {/* Data points */}
      {data.map((pt, i) => {
        const inBootstrap = bootstrapSet.has(i)
        return (
          <circle
            key={`pt-${i}`}
            cx={xScale(pt.x)}
            cy={yScale(pt.y)}
            r={1.5}
            fill={COLORS.clusters[pt.label % COLORS.clusters.length]}
            fillOpacity={inBootstrap ? 0.7 : 0.1}
          />
        )
      })}

      {/* Label */}
      <text x={3} y={10} className="text-[8px] fill-text-tertiary">
        Tree {treeIndex + 1}
      </text>

      {/* Border */}
      <rect x={0} y={0} width={width} height={height} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={0.5} rx={3} />
    </motion.g>
  )
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function RandomForestViz() {
  const [nEstimators, setNEstimators] = useState(9)
  const [maxDepth, setMaxDepth] = useState(3)

  const data = useMemo(() => makeMoonsClassification(200, 0.15, 42), [])
  const snapshots = useMemo(() => runRandomForest(data, nEstimators, maxDepth), [data, nEstimators, maxDepth])
  const player = useAlgorithmPlayer({ snapshots, baseFps: 1.2 })
  const snap = player.currentSnapshot

  const accuracies = useMemo(() => snapshots.map(s => s.aggregatedAccuracy), [snapshots])

  const xExtent = useMemo(() => d3.extent(data, d => d.x) as [number, number], [data])
  const yExtent = useMemo(() => d3.extent(data, d => d.y) as [number, number], [data])
  const xPad = (xExtent[1] - xExtent[0]) * 0.1
  const yPad = (yExtent[1] - yExtent[0]) * 0.1

  return (
    <GlassCard className="p-8">
      {/* Controls */}
      <div className="flex flex-wrap gap-6 mb-6">
        <Slider
          label="Trees"
          value={nEstimators}
          min={1}
          max={20}
          step={1}
          onChange={(v) => { setNEstimators(v); player.reset() }}
          className="w-44"
        />
        <Slider
          label="Max Depth"
          value={maxDepth}
          min={1}
          max={5}
          step={1}
          onChange={(v) => { setMaxDepth(v); player.reset() }}
          className="w-44"
        />
      </div>

      {/* Grid of mini tree plots */}
      <div className="mb-4">
        <div className="text-xs text-text-tertiary uppercase tracking-wider mb-2">Individual Trees (Bootstrap Samples)</div>
        <SVGContainer aspectRatio={3 / 1} minHeight={200} maxHeight={300} padding={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          {({ innerWidth, innerHeight }) => {
            const cols = Math.min(snap.nTreesBuilt, 9) >= 7 ? 3 : snap.nTreesBuilt >= 4 ? 3 : Math.min(snap.nTreesBuilt, 3)
            const actualCols = Math.max(cols, 3)
            const rows = Math.ceil(Math.min(snap.nTreesBuilt, 9) / actualCols) || 1
            const cellW = (innerWidth - (actualCols - 1) * 4) / actualCols
            const cellH = (innerHeight - (rows - 1) * 4) / rows
            const treesToShow = snap.trees.slice(0, 9)

            const xS = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([0, cellW])
            const yS = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([cellH, 0])

            return (
              <>
                {treesToShow.map((tree, i) => {
                  const col = i % actualCols
                  const row = Math.floor(i / actualCols)
                  const tx = col * (cellW + 4)
                  const ty = row * (cellH + 4)
                  return (
                    <g key={`tree-${i}`} transform={`translate(${tx}, ${ty})`}>
                      <MiniTreePlot
                        data={data}
                        boundaries={tree.boundaries}
                        bootstrapIndices={tree.bootstrapIndices}
                        width={cellW}
                        height={cellH}
                        xScale={xS}
                        yScale={yS}
                        treeIndex={i}
                        isNew={i === snap.nTreesBuilt - 1}
                      />
                    </g>
                  )
                })}
                {/* Empty cells */}
                {snap.nTreesBuilt === 0 && (
                  <text x={innerWidth / 2} y={innerHeight / 2} textAnchor="middle" className="text-xs fill-text-tertiary">
                    No trees built yet
                  </text>
                )}
              </>
            )
          }}
        </SVGContainer>
      </div>

      {/* Aggregated boundary plot + Feature importance */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Aggregated scatter */}
        <div className="lg:col-span-2 bg-obsidian-surface/40 rounded-lg border border-obsidian-border p-2">
          <div className="text-xs text-text-tertiary uppercase tracking-wider mb-2 px-1">Aggregated Forest Prediction (Majority Vote)</div>
          <SVGContainer aspectRatio={16 / 10} minHeight={250} maxHeight={400} padding={{ top: 15, right: 15, bottom: 30, left: 40 }}>
            {({ innerWidth, innerHeight }) => {
              const xScale = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([0, innerWidth]).nice()
              const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([innerHeight, 0]).nice()
              const GRID_RES = snap.aggregatedGrid.length

              const cellW = innerWidth / GRID_RES
              const cellH = innerHeight / GRID_RES

              return (
                <>
                  {/* Grid cells for aggregated boundary */}
                  {snap.aggregatedGrid.map((row, gy) =>
                    row.map((pred, gx) => (
                      <rect
                        key={`ag-${gy}-${gx}`}
                        x={gx * cellW}
                        y={gy * cellH}
                        width={cellW + 0.5}
                        height={cellH + 0.5}
                        fill={COLORS.clusters[pred % COLORS.clusters.length]}
                        fillOpacity={0.1}
                      />
                    ))
                  )}

                  {/* Data points */}
                  {data.map((pt, i) => (
                    <motion.circle
                      key={`apt-${i}`}
                      cx={xScale(pt.x)}
                      cy={yScale(pt.y)}
                      r={3}
                      fill={COLORS.clusters[pt.label % COLORS.clusters.length]}
                      fillOpacity={0.8}
                      stroke={COLORS.clusters[pt.label % COLORS.clusters.length]}
                      strokeWidth={0.5}
                      strokeOpacity={0.3}
                    />
                  ))}

                  {/* Axes */}
                  {xScale.ticks(5).map(tick => (
                    <text key={`xt-${tick}`} x={xScale(tick)} y={innerHeight + 18} textAnchor="middle" className="text-[9px] fill-text-tertiary">
                      {tick.toFixed(1)}
                    </text>
                  ))}
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

        {/* Feature importance */}
        <div className="bg-obsidian-surface/40 rounded-lg border border-obsidian-border p-4">
          <div className="text-xs text-text-tertiary uppercase tracking-wider mb-4">Feature Importance</div>
          <div className="space-y-4 mt-6">
            {(['x', 'y'] as const).map((feat, i) => {
              const imp = snap.featureImportance[i]
              return (
                <div key={feat}>
                  <div className="flex justify-between text-xs text-text-secondary mb-1">
                    <span className="font-mono">{feat}</span>
                    <span className="font-mono text-text-tertiary">{(imp * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-5 bg-obsidian-surface rounded-md overflow-hidden">
                    <motion.div
                      className="h-full rounded-md"
                      style={{ backgroundColor: COLORS.clusters[i + 2] }}
                      initial={{ width: 0 }}
                      animate={{ width: `${imp * 100}%` }}
                      transition={{ duration: 0.5, ease: 'easeOut' }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
          <div className="mt-8 pt-4 border-t border-obsidian-border">
            <div className="text-xs text-text-tertiary mb-2">Forest Stats</div>
            <div className="space-y-1 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-text-tertiary">Trees built</span>
                <span className="text-text-secondary">{snap.nTreesBuilt}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-tertiary">Accuracy</span>
                <span className="text-success">{(snap.aggregatedAccuracy * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Transport */}
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
          label="Ensemble Accuracy"
          width={200}
          height={80}
          color={COLORS.success}
        />
        <div className="text-xs text-text-tertiary self-center">
          {snap.description}
        </div>
      </div>
    </GlassCard>
  )
}
