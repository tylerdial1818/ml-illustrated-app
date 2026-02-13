import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { makeClassification } from '../../../lib/data/regressionGenerators'
import { runLogisticRegression } from '../../../lib/algorithms/regression/logistic'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { ConvergenceChart } from '../../../components/viz/ConvergenceChart'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function LogisticRegressionViz() {
  const [separation, setSeparation] = useState(2.5)
  const [threshold, setThreshold] = useState(0.5)

  const classData = useMemo(() => makeClassification(120, separation, 42), [separation])

  const { X, y } = useMemo(() => {
    const X = classData.map((p) => [p.x, p.y])
    const y = classData.map((p) => p.label)
    return { X, y }
  }, [classData])

  const snapshots = useMemo(
    () => runLogisticRegression(X, y, 0.1, 100),
    [X, y]
  )

  const player = useAlgorithmPlayer({ snapshots, baseFps: 5 })
  const snap = player.currentSnapshot

  const losses = useMemo(() => snapshots.map((s) => s.loss), [snapshots])

  // Compute confusion matrix
  const { tp, fp, tn, fn } = useMemo(() => {
    let tp = 0, fp = 0, tn = 0, fn = 0
    snap.predictions.forEach((pred, i) => {
      const predicted = pred >= threshold ? 1 : 0
      if (predicted === 1 && y[i] === 1) tp++
      else if (predicted === 1 && y[i] === 0) fp++
      else if (predicted === 0 && y[i] === 0) tn++
      else fn++
    })
    return { tp, fp, tn, fn }
  }, [snap.predictions, y, threshold])

  return (
    <GlassCard className="p-8">
      <div className="flex flex-wrap gap-6 mb-6">
        <Slider
          label="Class separation"
          value={separation}
          min={0.5}
          max={5}
          step={0.1}
          onChange={(v) => { setSeparation(v); player.reset() }}
          formatValue={(v) => v.toFixed(1)}
          className="w-48"
        />
        <Slider
          label="Decision threshold"
          value={threshold}
          min={0.1}
          max={0.9}
          step={0.05}
          onChange={setThreshold}
          formatValue={(v) => v.toFixed(2)}
          className="w-48"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Main classification plot */}
        <div className="lg:col-span-2">
          <SVGContainer aspectRatio={16 / 10} minHeight={300} maxHeight={450}>
            {({ innerWidth, innerHeight }) => {
              const xExtent = d3.extent(classData, (d) => d.x) as [number, number]
              const yExtent = d3.extent(classData, (d) => d.y) as [number, number]
              const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
              const yScale = d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice()

              // Decision boundary: w0 + w1*x + w2*y = logit(threshold)
              const [bias, w1, w2] = snap.weights
              const logitThreshold = Math.log(threshold / (1 - threshold))

              let boundaryPoints: { x1: number; y1: number; x2: number; y2: number } | null = null
              if (Math.abs(w2) > 1e-10) {
                const x1 = xExtent[0]
                const x2 = xExtent[1]
                const boundaryY1 = (logitThreshold - bias - w1 * x1) / w2
                const boundaryY2 = (logitThreshold - bias - w1 * x2) / w2
                boundaryPoints = {
                  x1: xScale(x1),
                  y1: yScale(boundaryY1),
                  x2: xScale(x2),
                  y2: yScale(boundaryY2),
                }
              }

              // Probability field as colored rectangles
              const gridSize = 15
              const gridCells: { x: number; y: number; prob: number }[] = []
              for (let gx = 0; gx < gridSize; gx++) {
                for (let gy = 0; gy < gridSize; gy++) {
                  const px = xExtent[0] + (gx / (gridSize - 1)) * (xExtent[1] - xExtent[0])
                  const py = yExtent[0] + (gy / (gridSize - 1)) * (yExtent[1] - yExtent[0])
                  const z = bias + w1 * px + w2 * py
                  const prob = 1 / (1 + Math.exp(-z))
                  gridCells.push({ x: px, y: py, prob })
                }
              }

              const cellW = innerWidth / (gridSize - 1)
              const cellH = innerHeight / (gridSize - 1)

              return (
                <>
                  {/* Probability field */}
                  {gridCells.map((cell, i) => {
                    const color = d3.interpolateRgb(COLORS.clusters[1], COLORS.clusters[0])(cell.prob)
                    return (
                      <rect
                        key={`grid-${i}`}
                        x={xScale(cell.x) - cellW / 2}
                        y={yScale(cell.y) - cellH / 2}
                        width={cellW}
                        height={cellH}
                        fill={color}
                        fillOpacity={0.12}
                      />
                    )
                  })}

                  {/* Decision boundary */}
                  {boundaryPoints && (
                    <line
                      x1={boundaryPoints.x1}
                      y1={boundaryPoints.y1}
                      x2={boundaryPoints.x2}
                      y2={boundaryPoints.y2}
                      stroke="#fff"
                      strokeWidth={2}
                      strokeOpacity={0.6}
                      strokeDasharray="6,4"
                    />
                  )}

                  {/* Data points */}
                  {classData.map((point, i) => (
                    <motion.circle
                      key={i}
                      cx={xScale(point.x)}
                      cy={yScale(point.y)}
                      r={4}
                      fill={point.label === 1 ? COLORS.clusters[0] : COLORS.clusters[1]}
                      fillOpacity={0.8}
                      stroke="#0F0F11"
                      strokeWidth={1}
                    />
                  ))}
                </>
              )
            }}
          </SVGContainer>
        </div>

        {/* Confusion matrix */}
        <div className="space-y-4">
          <div>
            <p className="text-xs text-text-tertiary uppercase tracking-wider mb-2">Confusion Matrix</p>
            <div className="grid grid-cols-2 gap-1 text-center text-xs font-mono">
              <div className="bg-success/20 text-success p-2 rounded">TP: {tp}</div>
              <div className="bg-error/20 text-error p-2 rounded">FP: {fp}</div>
              <div className="bg-error/20 text-error p-2 rounded">FN: {fn}</div>
              <div className="bg-success/20 text-success p-2 rounded">TN: {tn}</div>
            </div>
          </div>
          <div className="text-xs text-text-tertiary space-y-1 font-mono">
            <p>Accuracy: {((tp + tn) / (tp + fp + tn + fn) * 100).toFixed(1)}%</p>
            <p>Precision: {(tp / (tp + fp + 1e-10) * 100).toFixed(1)}%</p>
            <p>Recall: {(tp / (tp + fn + 1e-10) * 100).toFixed(1)}%</p>
          </div>
          <ConvergenceChart
            values={losses}
            currentIndex={player.currentStep}
            label="Loss (BCE)"
            width={180}
            height={70}
            color={COLORS.error}
          />
        </div>
      </div>

      <div className="mt-4">
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
      </div>
    </GlassCard>
  )
}
