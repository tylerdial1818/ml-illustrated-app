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
import { makeRegressionCurve, type RegressionPoint1D } from '../../../lib/data/treeDataGenerators'

/* ------------------------------------------------------------------ */
/*  Self-contained simplified GBT for 1D regression                    */
/* ------------------------------------------------------------------ */

interface StumpNode {
  featureThreshold: number | null
  leftValue: number
  rightValue: number
  isLeaf: boolean
  value: number  // leaf value for single-leaf case
  // For deeper trees
  left: StumpNode | null
  right: StumpNode | null
}

interface TreeContribution {
  stump: StumpNode
  learningRate: number
  mse: number  // MSE of residuals this tree was fit to
}

interface GBTSnapshot {
  nTrees: number
  predictions: number[]         // current prediction for each point
  residuals: number[]           // current residuals
  smoothPredictions: number[]   // prediction curve sampled on grid
  smoothTrue: number[]          // true function on grid
  gridX: number[]               // x values for smooth curve
  treesBuilt: TreeContribution[]
  mse: number
  description: string
}

function trueFunction(x: number): number {
  return Math.sin(x * 1.5) + 0.5 * Math.sin(x * 3)
}

function buildRegressionStump(
  xs: number[],
  ys: number[],   // residuals to fit
  maxDepth: number,
  depth: number = 0,
): StumpNode {
  const mean = ys.reduce((a, b) => a + b, 0) / ys.length

  if (depth >= maxDepth || ys.length < 4) {
    return { featureThreshold: null, leftValue: mean, rightValue: mean, isLeaf: true, value: mean, left: null, right: null }
  }

  // Find best split
  const indices = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b])
  let bestMSE = Infinity
  let bestThresh = 0
  let bestLeftIdx: number[] = []
  let bestRightIdx: number[] = []

  const nCandidates = Math.min(indices.length - 1, 20)
  const step = Math.max(1, Math.floor((indices.length - 1) / nCandidates))

  for (let s = 1; s < indices.length; s += step) {
    const leftIdx = indices.slice(0, s)
    const rightIdx = indices.slice(s)
    if (leftIdx.length === 0 || rightIdx.length === 0) continue

    const leftMean = leftIdx.reduce((acc, i) => acc + ys[i], 0) / leftIdx.length
    const rightMean = rightIdx.reduce((acc, i) => acc + ys[i], 0) / rightIdx.length

    let mse = 0
    for (const i of leftIdx) mse += (ys[i] - leftMean) ** 2
    for (const i of rightIdx) mse += (ys[i] - rightMean) ** 2
    mse /= ys.length

    if (mse < bestMSE) {
      bestMSE = mse
      bestThresh = (xs[indices[s - 1]] + xs[indices[s]]) / 2
      bestLeftIdx = leftIdx
      bestRightIdx = rightIdx
    }
  }

  if (bestLeftIdx.length === 0 || bestRightIdx.length === 0) {
    return { featureThreshold: null, leftValue: mean, rightValue: mean, isLeaf: true, value: mean, left: null, right: null }
  }

  const leftXs = bestLeftIdx.map(i => xs[i])
  const leftYs = bestLeftIdx.map(i => ys[i])
  const rightXs = bestRightIdx.map(i => xs[i])
  const rightYs = bestRightIdx.map(i => ys[i])

  const leftChild = buildRegressionStump(leftXs, leftYs, maxDepth, depth + 1)
  const rightChild = buildRegressionStump(rightXs, rightYs, maxDepth, depth + 1)

  const leftMean = bestLeftIdx.reduce((acc, i) => acc + ys[i], 0) / bestLeftIdx.length
  const rightMean = bestRightIdx.reduce((acc, i) => acc + ys[i], 0) / bestRightIdx.length

  return {
    featureThreshold: bestThresh,
    leftValue: leftMean,
    rightValue: rightMean,
    isLeaf: false,
    value: mean,
    left: leftChild,
    right: rightChild,
  }
}

function predictStump(stump: StumpNode, x: number): number {
  if (stump.isLeaf) return stump.value
  if (x <= stump.featureThreshold!) {
    return stump.left ? predictStump(stump.left, x) : stump.leftValue
  } else {
    return stump.right ? predictStump(stump.right, x) : stump.rightValue
  }
}

function runGBTRegression(
  data: RegressionPoint1D[],
  nEstimators: number,
  learningRate: number,
  maxDepth: number,
): GBTSnapshot[] {
  const snapshots: GBTSnapshot[] = []
  const xs = data.map(d => d.x)
  const ys = data.map(d => d.y)
  const n = data.length

  // Grid for smooth prediction curve
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const gridSize = 100
  const gridX: number[] = []
  for (let i = 0; i < gridSize; i++) {
    gridX.push(xMin + (i / (gridSize - 1)) * (xMax - xMin))
  }
  const smoothTrue = gridX.map(x => trueFunction(x))

  // Start with mean prediction
  const meanY = ys.reduce((a, b) => a + b, 0) / n
  let predictions = new Array(n).fill(meanY)
  let gridPredictions = new Array(gridSize).fill(meanY)
  const treesBuilt: TreeContribution[] = []

  function computeMSE(preds: number[]): number {
    let sum = 0
    for (let i = 0; i < n; i++) sum += (ys[i] - preds[i]) ** 2
    return sum / n
  }

  // Initial snapshot
  const initialResiduals = ys.map((y, i) => y - predictions[i])
  snapshots.push({
    nTrees: 0,
    predictions: [...predictions],
    residuals: initialResiduals,
    smoothPredictions: [...gridPredictions],
    smoothTrue,
    gridX,
    treesBuilt: [],
    mse: computeMSE(predictions),
    description: `Initial prediction: mean = ${meanY.toFixed(3)}`,
  })

  for (let t = 0; t < nEstimators; t++) {
    // Compute residuals
    const residuals = ys.map((y, i) => y - predictions[i])

    // Fit stump to residuals
    const stump = buildRegressionStump(xs, residuals, maxDepth)

    // Update predictions
    const stumpMSE = residuals.reduce((a, r) => a + r * r, 0) / n
    predictions = predictions.map((p, i) => p + learningRate * predictStump(stump, xs[i]))
    gridPredictions = gridPredictions.map((p, i) => p + learningRate * predictStump(stump, gridX[i]))

    treesBuilt.push({
      stump,
      learningRate,
      mse: stumpMSE,
    })

    const newResiduals = ys.map((y, i) => y - predictions[i])

    snapshots.push({
      nTrees: t + 1,
      predictions: [...predictions],
      residuals: newResiduals,
      smoothPredictions: [...gridPredictions],
      smoothTrue,
      gridX,
      treesBuilt: treesBuilt.map(tb => ({ ...tb })),
      mse: computeMSE(predictions),
      description: `Added tree ${t + 1}: residual MSE = ${stumpMSE.toFixed(4)}`,
    })
  }

  return snapshots
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function GradientBoostedViz() {
  const [nEstimators, setNEstimators] = useState(15)
  const [learningRate, setLearningRate] = useState(0.3)
  const [maxDepth, setMaxDepth] = useState(2)

  const data = useMemo(() => makeRegressionCurve(80, 'sine', 0.3, 42), [])
  const snapshots = useMemo(
    () => runGBTRegression(data, nEstimators, learningRate, maxDepth),
    [data, nEstimators, learningRate, maxDepth],
  )
  const player = useAlgorithmPlayer({ snapshots, baseFps: 1.5 })
  const snap = player.currentSnapshot

  const mseValues = useMemo(() => snapshots.map(s => s.mse), [snapshots])

  return (
    <GlassCard className="p-8">
      {/* Controls */}
      <div className="flex flex-wrap gap-6 mb-6">
        <Slider
          label="Estimators"
          value={nEstimators}
          min={1}
          max={30}
          step={1}
          onChange={(v) => { setNEstimators(v); player.reset() }}
          className="w-44"
        />
        <Slider
          label="Learning Rate"
          value={learningRate}
          min={0.01}
          max={1.0}
          step={0.01}
          onChange={(v) => { setLearningRate(v); player.reset() }}
          formatValue={(v) => v.toFixed(2)}
          className="w-44"
        />
        <Slider
          label="Max Depth"
          value={maxDepth}
          min={1}
          max={3}
          step={1}
          onChange={(v) => { setMaxDepth(v); player.reset() }}
          className="w-36"
        />
      </div>

      {/* Top: Prediction curve */}
      <div className="bg-obsidian-surface/40 rounded-lg border border-obsidian-border p-2 mb-4">
        <div className="text-xs text-text-tertiary uppercase tracking-wider mb-2 px-1">Prediction vs True Function</div>
        <SVGContainer aspectRatio={16 / 6} minHeight={200} maxHeight={300} padding={{ top: 15, right: 20, bottom: 30, left: 45 }}>
          {({ innerWidth, innerHeight }) => {
            const xExtent = d3.extent(data, d => d.x) as [number, number]
            const allYs = [...data.map(d => d.y), ...snap.smoothPredictions, ...snap.smoothTrue]
            const yExtent = d3.extent(allYs) as [number, number]
            const yPad = (yExtent[1] - yExtent[0]) * 0.1
            const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
            const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([innerHeight, 0]).nice()

            // True function path
            const trueLine = d3.line<number>()
              .x((_, i) => xScale(snap.gridX[i]))
              .y(d => yScale(d))
              .curve(d3.curveMonotoneX)

            const predLine = d3.line<number>()
              .x((_, i) => xScale(snap.gridX[i]))
              .y(d => yScale(d))
              .curve(d3.curveMonotoneX)

            return (
              <>
                {/* Grid */}
                {yScale.ticks(5).map(tick => (
                  <line key={`grid-${tick}`} x1={0} x2={innerWidth} y1={yScale(tick)} y2={yScale(tick)}
                    stroke="rgba(255,255,255,0.04)" strokeDasharray="3,3" />
                ))}

                {/* True function (faint) */}
                <path
                  d={trueLine(snap.smoothTrue) ?? ''}
                  fill="none"
                  stroke="rgba(255,255,255,0.2)"
                  strokeWidth={1.5}
                  strokeDasharray="6,4"
                />

                {/* Model prediction (bold) */}
                <motion.path
                  d={predLine(snap.smoothPredictions) ?? ''}
                  fill="none"
                  stroke={COLORS.accent}
                  strokeWidth={2.5}
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.5 }}
                />

                {/* Data points */}
                {data.map((pt, i) => (
                  <motion.circle
                    key={`pt-${i}`}
                    cx={xScale(pt.x)}
                    cy={yScale(pt.y)}
                    r={2.5}
                    fill={COLORS.clusters[0]}
                    fillOpacity={0.6}
                    stroke={COLORS.clusters[0]}
                    strokeWidth={0.5}
                    strokeOpacity={0.3}
                  />
                ))}

                {/* Legend */}
                <g transform={`translate(${innerWidth - 140}, 5)`}>
                  <line x1={0} y1={5} x2={20} y2={5} stroke="rgba(255,255,255,0.2)" strokeWidth={1.5} strokeDasharray="6,4" />
                  <text x={25} y={8} className="text-[9px] fill-text-tertiary">True function</text>
                  <line x1={0} y1={20} x2={20} y2={20} stroke={COLORS.accent} strokeWidth={2.5} />
                  <text x={25} y={23} className="text-[9px] fill-text-secondary">GBT prediction</text>
                </g>

                {/* Axes */}
                {xScale.ticks(6).map(tick => (
                  <text key={`xt-${tick}`} x={xScale(tick)} y={innerHeight + 18} textAnchor="middle" className="text-[9px] fill-text-tertiary">
                    {tick.toFixed(1)}
                  </text>
                ))}
                {yScale.ticks(5).map(tick => (
                  <text key={`yt-${tick}`} x={-8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" className="text-[9px] fill-text-tertiary">
                    {tick.toFixed(2)}
                  </text>
                ))}
              </>
            )
          }}
        </SVGContainer>
      </div>

      {/* Middle: Residual plot */}
      <div className="bg-obsidian-surface/40 rounded-lg border border-obsidian-border p-2 mb-4">
        <div className="text-xs text-text-tertiary uppercase tracking-wider mb-2 px-1">Current Residuals</div>
        <SVGContainer aspectRatio={16 / 4} minHeight={120} maxHeight={180} padding={{ top: 10, right: 20, bottom: 25, left: 45 }}>
          {({ innerWidth, innerHeight }) => {
            const xExtent = d3.extent(data, d => d.x) as [number, number]
            const resExtent = d3.extent(snap.residuals) as [number, number]
            const maxAbs = Math.max(Math.abs(resExtent[0]), Math.abs(resExtent[1]), 0.1)
            const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
            const yScale = d3.scaleLinear().domain([-maxAbs * 1.1, maxAbs * 1.1]).range([innerHeight, 0])

            return (
              <>
                {/* Zero line */}
                <line x1={0} x2={innerWidth} y1={yScale(0)} y2={yScale(0)} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />

                {/* Residual stems */}
                {data.map((pt, i) => (
                  <motion.g key={`res-${i}`}>
                    <motion.line
                      x1={xScale(pt.x)}
                      y1={yScale(0)}
                      x2={xScale(pt.x)}
                      y2={yScale(snap.residuals[i])}
                      stroke={snap.residuals[i] >= 0 ? COLORS.error : COLORS.clusters[4]}
                      strokeWidth={1}
                      strokeOpacity={0.4}
                      animate={{ y2: yScale(snap.residuals[i]) }}
                      transition={{ duration: 0.3 }}
                    />
                    <motion.circle
                      cx={xScale(pt.x)}
                      cy={yScale(snap.residuals[i])}
                      r={2}
                      fill={snap.residuals[i] >= 0 ? COLORS.error : COLORS.clusters[4]}
                      fillOpacity={0.7}
                      animate={{ cy: yScale(snap.residuals[i]) }}
                      transition={{ duration: 0.3 }}
                    />
                  </motion.g>
                ))}

                {/* Axes */}
                {xScale.ticks(6).map(tick => (
                  <text key={`xt-${tick}`} x={xScale(tick)} y={innerHeight + 16} textAnchor="middle" className="text-[9px] fill-text-tertiary">
                    {tick.toFixed(1)}
                  </text>
                ))}
                {yScale.ticks(3).map(tick => (
                  <text key={`yt-${tick}`} x={-8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" className="text-[9px] fill-text-tertiary">
                    {tick.toFixed(2)}
                  </text>
                ))}
              </>
            )
          }}
        </SVGContainer>
      </div>

      {/* Bottom: Ensemble composition strip */}
      <div className="bg-obsidian-surface/40 rounded-lg border border-obsidian-border p-3 mb-4">
        <div className="text-xs text-text-tertiary uppercase tracking-wider mb-3">
          Ensemble Composition ({snap.nTrees} tree{snap.nTrees !== 1 ? 's' : ''})
        </div>
        <div className="flex gap-2 overflow-x-auto pb-2">
          {snap.nTrees === 0 && (
            <div className="text-xs text-text-tertiary italic px-2">No trees yet — starting with mean prediction</div>
          )}
          {/* Mean base prediction card */}
          <motion.div
            className="flex-shrink-0 bg-obsidian-surface/80 rounded-lg border border-obsidian-border p-2 min-w-[70px]"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="text-[9px] text-text-tertiary uppercase tracking-wider mb-1">Base</div>
            <div className="text-xs font-mono text-text-secondary">
              {data.length > 0 ? (data.reduce((a, d) => a + d.y, 0) / data.length).toFixed(3) : '0'}
            </div>
            <div className="h-1 mt-1 rounded-full bg-white/10" />
          </motion.div>

          {/* Tree contribution cards */}
          {snap.treesBuilt.map((tree, i) => {
            const isNewest = i === snap.nTrees - 1
            return (
              <motion.div
                key={`tc-${i}`}
                className={`flex-shrink-0 rounded-lg border p-2 min-w-[70px] ${
                  isNewest
                    ? 'bg-accent/10 border-accent/30'
                    : 'bg-obsidian-surface/80 border-obsidian-border'
                }`}
                initial={{ opacity: 0, y: 10, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.3, delay: isNewest ? 0.1 : 0 }}
              >
                <div className="text-[9px] text-text-tertiary uppercase tracking-wider mb-1">
                  T{i + 1}
                </div>
                <div className="text-[10px] font-mono text-text-secondary">
                  lr={tree.learningRate.toFixed(2)}
                </div>
                <div className="mt-1 h-1.5 rounded-full overflow-hidden bg-obsidian-surface">
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      backgroundColor: COLORS.clusters[i % COLORS.clusters.length],
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, (1 - tree.mse / (mseValues[0] || 1)) * 100)}%` }}
                    transition={{ duration: 0.4 }}
                  />
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* Transport + convergence */}
      <div className="mt-2 flex flex-wrap items-start gap-4">
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
          values={mseValues}
          currentIndex={player.currentStep}
          label="MSE"
          width={200}
          height={80}
          color={COLORS.error}
        />
        <div className="text-xs text-text-tertiary self-center space-y-1">
          <div>{snap.description}</div>
          <div className="font-mono text-text-secondary">
            MSE: <span className="text-accent">{snap.mse.toFixed(4)}</span>
          </div>
        </div>
      </div>
    </GlassCard>
  )
}
