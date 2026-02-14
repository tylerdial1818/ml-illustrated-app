import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { makeLinearSeparable, makeXOR } from '../../../lib/data/nnDataGenerators'
import { runPerceptronTraining } from '../../../lib/algorithms/neural-networks/perceptron'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { ConvergenceChart } from '../../../components/viz/ConvergenceChart'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { Select } from '../../../components/ui/Select'
import { Toggle } from '../../../components/ui/Toggle'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

// ── Neuron Diagram (Left Panel) ────────────────────────────────────────

function NeuronDiagram({
  w1,
  w2,
  bias,
  activation,
  width,
  height,
}: {
  w1: number
  w2: number
  bias: number
  activation: 'step' | 'sigmoid'
  width: number
  height: number
}) {
  const cx = width / 2
  const cy = height / 2
  const nodeR = 28

  // Clamp arrow thickness between 1 and 8
  const maxWeight = Math.max(Math.abs(w1), Math.abs(w2), 0.01)
  const thick1 = 1 + (Math.abs(w1) / maxWeight) * 7
  const thick2 = 1 + (Math.abs(w2) / maxWeight) * 7

  // Arrow start positions
  const inputX = 30
  const input1Y = cy - 40
  const input2Y = cy + 40

  // Summation node center
  const sumX = cx - 20
  const sumY = cy

  // Activation curve position
  const actX = cx + 40
  const actY = cy

  // Output position
  const outX = width - 30
  const outY = cy

  // Draw a small sigmoid or step curve
  const curveW = 36
  const curveH = 28
  const curvePath =
    activation === 'sigmoid'
      ? (() => {
          const pts: string[] = []
          for (let i = 0; i <= 20; i++) {
            const t = (i / 20) * 6 - 3
            const sx = actX - curveW / 2 + (i / 20) * curveW
            const sy = actY + curveH / 2 - (1 / (1 + Math.exp(-t))) * curveH
            pts.push(`${i === 0 ? 'M' : 'L'}${sx},${sy}`)
          }
          return pts.join(' ')
        })()
      : `M${actX - curveW / 2},${actY + curveH / 2} L${actX},${actY + curveH / 2} L${actX},${actY - curveH / 2} L${actX + curveW / 2},${actY - curveH / 2}`

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* Background grid */}
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6" fill="rgba(255,255,255,0.4)" />
        </marker>
      </defs>

      {/* Input labels */}
      <text x={inputX - 10} y={input1Y + 4} textAnchor="end" className="text-[11px] fill-text-secondary font-mono">
        x&#8321;
      </text>
      <text x={inputX - 10} y={input2Y + 4} textAnchor="end" className="text-[11px] fill-text-secondary font-mono">
        x&#8322;
      </text>

      {/* Input arrows with weight-proportional thickness */}
      <motion.line
        x1={inputX}
        y1={input1Y}
        x2={sumX - nodeR - 2}
        y2={sumY - 6}
        stroke={w1 >= 0 ? COLORS.clusters[0] : COLORS.error}
        strokeOpacity={0.7}
        animate={{ strokeWidth: thick1 }}
        transition={{ duration: 0.3 }}
        markerEnd="url(#arrow)"
      />
      <motion.line
        x1={inputX}
        y1={input2Y}
        x2={sumX - nodeR - 2}
        y2={sumY + 6}
        stroke={w2 >= 0 ? COLORS.clusters[0] : COLORS.error}
        strokeOpacity={0.7}
        animate={{ strokeWidth: thick2 }}
        transition={{ duration: 0.3 }}
        markerEnd="url(#arrow)"
      />

      {/* Weight labels */}
      <text
        x={(inputX + sumX - nodeR) / 2}
        y={input1Y - 14}
        textAnchor="middle"
        className="text-[10px] fill-text-tertiary font-mono"
      >
        w&#8321;={w1.toFixed(2)}
      </text>
      <text
        x={(inputX + sumX - nodeR) / 2}
        y={input2Y + 18}
        textAnchor="middle"
        className="text-[10px] fill-text-tertiary font-mono"
      >
        w&#8322;={w2.toFixed(2)}
      </text>

      {/* Bias arrow from below */}
      <line
        x1={sumX}
        y1={sumY + 50}
        x2={sumX}
        y2={sumY + nodeR + 2}
        stroke="rgba(255,255,255,0.3)"
        strokeWidth={1.5}
        markerEnd="url(#arrow)"
      />
      <text
        x={sumX}
        y={sumY + 65}
        textAnchor="middle"
        className="text-[10px] fill-text-tertiary font-mono"
      >
        b={bias.toFixed(2)}
      </text>

      {/* Summation node */}
      <circle cx={sumX} cy={sumY} r={nodeR} fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.2)" strokeWidth={1.5} />
      <text x={sumX} y={sumY + 6} textAnchor="middle" className="text-base fill-text-primary font-light">
        &Sigma;
      </text>

      {/* Arrow from sum to activation */}
      <line
        x1={sumX + nodeR + 2}
        y1={sumY}
        x2={actX - curveW / 2 - 8}
        y2={actY}
        stroke="rgba(255,255,255,0.3)"
        strokeWidth={1.5}
        markerEnd="url(#arrow)"
      />

      {/* Activation curve box */}
      <rect
        x={actX - curveW / 2 - 6}
        y={actY - curveH / 2 - 6}
        width={curveW + 12}
        height={curveH + 12}
        rx={6}
        fill="rgba(255,255,255,0.03)"
        stroke="rgba(255,255,255,0.1)"
        strokeWidth={1}
      />
      <path d={curvePath} fill="none" stroke={COLORS.accent} strokeWidth={2} />
      <text
        x={actX}
        y={actY + curveH / 2 + 18}
        textAnchor="middle"
        className="text-[9px] fill-text-tertiary"
      >
        {activation === 'sigmoid' ? 'sigmoid' : 'step'}
      </text>

      {/* Arrow from activation to output */}
      <line
        x1={actX + curveW / 2 + 8}
        y1={actY}
        x2={outX - 8}
        y2={outY}
        stroke="rgba(255,255,255,0.3)"
        strokeWidth={1.5}
        markerEnd="url(#arrow)"
      />

      {/* Output label */}
      <text x={outX + 4} y={outY + 4} textAnchor="start" className="text-[11px] fill-text-secondary font-mono">
        &ycirc;
      </text>
    </svg>
  )
}

// ── Main Component ─────────────────────────────────────────────────────

export function PerceptronViz() {
  const [learningRate, setLearningRate] = useState(0.1)
  const [activation, setActivation] = useState<'step' | 'sigmoid'>('step')
  const [useXOR, setUseXOR] = useState(false)
  const [seed, setSeed] = useState(42)

  const data = useMemo(
    () => (useXOR ? makeXOR(100) : makeLinearSeparable(100, 2, 0.3, 42)),
    [useXOR]
  )

  const snapshots = useMemo(
    () =>
      runPerceptronTraining(data, {
        learningRate,
        activation,
        epochs: 50,
        seed,
      }),
    [data, learningRate, activation, seed]
  )

  const player = useAlgorithmPlayer({ snapshots, baseFps: 4 })
  const snap = player.currentSnapshot

  const handleReset = useCallback(() => {
    setSeed((s) => s + 1)
    player.reset()
  }, [player])

  const losses = useMemo(() => snapshots.map((s) => s.loss), [snapshots])

  return (
    <GlassCard className="p-6 lg:p-8">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 mb-6">
        <Slider
          label="Learning Rate"
          value={learningRate}
          min={0.01}
          max={1}
          step={0.01}
          onChange={(v) => {
            setLearningRate(v)
            player.reset()
          }}
          formatValue={(v) => v.toFixed(2)}
          className="w-44"
        />
        <Select
          label="Activation"
          value={activation}
          options={[
            { value: 'step', label: 'Step' },
            { value: 'sigmoid', label: 'Sigmoid' },
          ]}
          onChange={(v) => {
            setActivation(v as 'step' | 'sigmoid')
            player.reset()
          }}
          className="w-36"
        />
        <Toggle
          label="XOR data (failure case)"
          checked={useXOR}
          onChange={(v) => {
            setUseXOR(v)
            player.reset()
          }}
        />
      </div>

      {/* Dual panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left — Neuron diagram */}
        <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
          <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-2">
            Neuron Anatomy
          </p>
          <NeuronDiagram
            w1={snap.weights[0]}
            w2={snap.weights[1]}
            bias={snap.bias}
            activation={activation}
            width={360}
            height={200}
          />
          {/* Weight readout */}
          <div className="flex gap-4 mt-2 text-xs font-mono text-text-tertiary">
            <span>
              w&#8321; = <span className="text-text-secondary">{snap.weights[0].toFixed(3)}</span>
            </span>
            <span>
              w&#8322; = <span className="text-text-secondary">{snap.weights[1].toFixed(3)}</span>
            </span>
            <span>
              b = <span className="text-text-secondary">{snap.bias.toFixed(3)}</span>
            </span>
            <span>
              acc = <span className="text-success">{(snap.accuracy * 100).toFixed(0)}%</span>
            </span>
          </div>
        </div>

        {/* Right — Decision boundary scatter plot */}
        <div>
          <SVGContainer
            aspectRatio={1}
            minHeight={280}
            maxHeight={420}
            padding={{ top: 15, right: 15, bottom: 30, left: 40 }}
          >
            {({ innerWidth, innerHeight }) => {
              const xExtent = d3.extent(data, (d) => d.x) as [number, number]
              const yExtent = d3.extent(data, (d) => d.y) as [number, number]
              const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
              const yScale = d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice()

              // Confidence gradient heatmap
              const gridSize = 20
              const cellW = innerWidth / gridSize
              const cellH = innerHeight / gridSize
              const [w1, w2] = snap.weights
              const bias = snap.bias

              const heatCells: { gx: number; gy: number; prob: number }[] = []
              for (let gi = 0; gi < gridSize; gi++) {
                for (let gj = 0; gj < gridSize; gj++) {
                  const px = xScale.invert((gi + 0.5) * cellW)
                  const py = yScale.invert((gj + 0.5) * cellH)
                  const z = w1 * px + w2 * py + bias
                  const prob = 1 / (1 + Math.exp(-z))
                  heatCells.push({ gx: gi, gy: gj, prob })
                }
              }

              // Decision boundary line
              let boundaryLine: { x1: number; y1: number; x2: number; y2: number } | null = null
              if (snap.decisionBoundary) {
                const { slope, intercept } = snap.decisionBoundary
                const xDomain = xScale.domain()
                const y1 = slope * xDomain[0] + intercept
                const y2 = slope * xDomain[1] + intercept
                boundaryLine = {
                  x1: xScale(xDomain[0]),
                  y1: yScale(y1),
                  x2: xScale(xDomain[1]),
                  y2: yScale(y2),
                }
              }

              return (
                <>
                  {/* Clip path for plot area */}
                  <defs>
                    <clipPath id="perceptron-plot-clip">
                      <rect x={0} y={0} width={innerWidth} height={innerHeight} />
                    </clipPath>
                  </defs>

                  {/* Confidence heatmap */}
                  {heatCells.map((cell, i) => {
                    const color = d3.interpolateRgb(COLORS.clusters[1], COLORS.clusters[0])(cell.prob)
                    return (
                      <rect
                        key={`h-${i}`}
                        x={cell.gx * cellW}
                        y={cell.gy * cellH}
                        width={cellW + 0.5}
                        height={cellH + 0.5}
                        fill={color}
                        fillOpacity={0.15}
                      />
                    )
                  })}

                  {/* Decision boundary (clipped to plot area) */}
                  {boundaryLine && (
                    <g clipPath="url(#perceptron-plot-clip)">
                      <motion.line
                        x1={boundaryLine.x1}
                        y1={boundaryLine.y1}
                        x2={boundaryLine.x2}
                        y2={boundaryLine.y2}
                        stroke="#fff"
                        strokeWidth={2}
                        strokeOpacity={0.7}
                        strokeDasharray="6,4"
                        animate={{
                          x1: boundaryLine.x1,
                          y1: boundaryLine.y1,
                          x2: boundaryLine.x2,
                          y2: boundaryLine.y2,
                        }}
                        transition={{ duration: 0.25 }}
                      />
                    </g>
                  )}

                  {/* Data points */}
                  {data.map((pt, i) => {
                    const isHighlighted = snap.highlightedPoint === i
                    return (
                      <motion.circle
                        key={i}
                        cx={xScale(pt.x)}
                        cy={yScale(pt.y)}
                        r={isHighlighted ? 6 : 3.5}
                        fill={pt.label === 1 ? COLORS.clusters[0] : COLORS.clusters[1]}
                        fillOpacity={0.8}
                        stroke={isHighlighted ? '#fff' : '#0F0F11'}
                        strokeWidth={isHighlighted ? 2 : 0.5}
                        animate={{
                          r: isHighlighted ? 6 : 3.5,
                          strokeWidth: isHighlighted ? 2 : 0.5,
                        }}
                        transition={{ duration: 0.15 }}
                      />
                    )
                  })}

                  {/* Axis labels */}
                  {xScale.ticks(5).map((t) => (
                    <text
                      key={`xt-${t}`}
                      x={xScale(t)}
                      y={innerHeight + 18}
                      textAnchor="middle"
                      className="text-[9px] fill-text-tertiary"
                    >
                      {t.toFixed(1)}
                    </text>
                  ))}
                  {yScale.ticks(5).map((t) => (
                    <text
                      key={`yt-${t}`}
                      x={-8}
                      y={yScale(t) + 3}
                      textAnchor="end"
                      className="text-[9px] fill-text-tertiary"
                    >
                      {t.toFixed(1)}
                    </text>
                  ))}
                </>
              )
            }}
          </SVGContainer>
        </div>
      </div>

      {/* XOR warning */}
      {useXOR && (
        <div className="mt-3 px-4 py-2 bg-error/10 border border-error/20 rounded-lg text-xs text-error">
          A single perceptron cannot solve XOR — the data is not linearly separable. Watch the boundary oscillate without converging.
        </div>
      )}

      {/* Transport + convergence */}
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
          onReset={handleReset}
          onSetSpeed={player.setSpeed}
        />
        <ConvergenceChart
          values={losses}
          currentIndex={player.currentStep}
          label={activation === 'sigmoid' ? 'Loss (BCE)' : 'Misclassification'}
          width={200}
          height={80}
          color={COLORS.error}
        />
        <div className="text-xs text-text-tertiary self-center font-mono">
          Step {snap.step}
          {snap.accuracy === 1 && (
            <span className="ml-2 text-success">Converged</span>
          )}
        </div>
      </div>
    </GlassCard>
  )
}
