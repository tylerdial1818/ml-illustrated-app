import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { makeSequence } from '../../../lib/data/nnDataGenerators'
import {
  runRNN,
  runLSTM,
  type RNNResult,
  type LSTMResult,
} from '../../../lib/algorithms/neural-networks/rnn'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { Slider } from '../../../components/ui/Slider'
import { Select } from '../../../components/ui/Select'
import { Toggle } from '../../../components/ui/Toggle'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

const SEQUENCE_OPTIONS = [
  { value: 'sine', label: 'Sine Wave' },
  { value: 'sawtooth', label: 'Sawtooth' },
  { value: 'square', label: 'Square Wave' },
]

const SEQ_LENGTH = 16

// ── Unrolled RNN Diagram ───────────────────────────────────────────────

function UnrolledRNNDiagram({
  rnnResult,
  currentStep,
  width,
  height,
}: {
  rnnResult: RNNResult
  currentStep: number
  width: number
  height: number
}) {
  const T = rnnResult.steps.length
  if (T === 0) return null

  const margin = { top: 15, right: 20, bottom: 35, left: 20 }
  const innerW = width - margin.left - margin.right
  const innerH = height - margin.top - margin.bottom

  const cellSpacing = innerW / T
  const cellY = innerH * 0.4
  const cellR = 16
  const inputY = innerH * 0.85
  const outputY = innerH * 0.05

  // Gradient magnitude for fading
  const maxGrad = Math.max(...rnnResult.gradientMagnitudes, 1e-10)

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <g transform={`translate(${margin.left}, ${margin.top})`}>
        <defs>
          <marker id="rnn-arrow" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
            <path d="M0,0 L6,2.5 L0,5" fill="rgba(255,255,255,0.3)" />
          </marker>
        </defs>

        {rnnResult.steps.map((step, t) => {
          const x = (t + 0.5) * cellSpacing
          const isActive = t <= currentStep
          const isCurrent = t === currentStep
          const gradFade = rnnResult.gradientMagnitudes[t] / maxGrad

          return (
            <g key={t}>
              {/* Hidden state arrow (horizontal connection) */}
              {t < T - 1 && (
                <line
                  x1={x + cellR + 2}
                  y1={cellY}
                  x2={(t + 1.5) * cellSpacing - cellR - 6}
                  y2={cellY}
                  stroke={COLORS.accent}
                  strokeWidth={1.5}
                  strokeOpacity={isActive ? 0.2 + gradFade * 0.6 : 0.08}
                  markerEnd="url(#rnn-arrow)"
                />
              )}

              {/* Input arrow (vertical up) */}
              <line
                x1={x}
                y1={inputY - 6}
                x2={x}
                y2={cellY + cellR + 4}
                stroke="rgba(255,255,255,0.2)"
                strokeWidth={1}
                markerEnd="url(#rnn-arrow)"
              />

              {/* Output arrow (vertical up from cell) */}
              <line
                x1={x}
                y1={cellY - cellR - 2}
                x2={x}
                y2={outputY + 10}
                stroke="rgba(255,255,255,0.15)"
                strokeWidth={1}
                markerEnd="url(#rnn-arrow)"
              />

              {/* Cell */}
              <motion.circle
                cx={x}
                cy={cellY}
                r={cellR}
                fill={isActive ? `rgba(99, 102, 241, ${0.15 + gradFade * 0.4})` : 'rgba(255,255,255,0.03)'}
                stroke={isCurrent ? COLORS.accent : 'rgba(255,255,255,0.15)'}
                strokeWidth={isCurrent ? 2 : 1}
                animate={{
                  fill: isActive ? `rgba(99, 102, 241, ${0.15 + gradFade * 0.4})` : 'rgba(255,255,255,0.03)',
                }}
                transition={{ duration: 0.2 }}
              />
              <text
                x={x}
                y={cellY + 4}
                textAnchor="middle"
                className="text-[7px] fill-text-secondary font-mono"
              >
                h{t}
              </text>

              {/* Input value */}
              <text
                x={x}
                y={inputY + 4}
                textAnchor="middle"
                className="text-[8px] fill-text-tertiary font-mono"
              >
                {step.input.toFixed(1)}
              </text>

              {/* Output value */}
              {isActive && (
                <text
                  x={x}
                  y={outputY + 4}
                  textAnchor="middle"
                  className="text-[8px] fill-text-secondary font-mono"
                >
                  {step.output.toFixed(2)}
                </text>
              )}

              {/* Time step label */}
              <text
                x={x}
                y={innerH + 12}
                textAnchor="middle"
                className="text-[7px] fill-text-tertiary"
              >
                t={t}
              </text>
            </g>
          )
        })}

        {/* Labels */}
        <text x={-6} y={inputY + 4} textAnchor="end" className="text-[8px] fill-text-tertiary">
          x
        </text>
        <text x={-6} y={cellY + 4} textAnchor="end" className="text-[8px] fill-text-tertiary">
          h
        </text>
        <text x={-6} y={outputY + 4} textAnchor="end" className="text-[8px] fill-text-tertiary">
          y
        </text>
      </g>
    </svg>
  )
}

// ── Gradient Magnitude Chart ──────────────────────────────────────────

function GradientChart({
  rnnGradients,
  lstmGradients,
  currentStep,
  width,
  height,
}: {
  rnnGradients: number[]
  lstmGradients: number[]
  currentStep: number
  width: number
  height: number
}) {
  const padding = { top: 15, right: 15, bottom: 25, left: 40 }
  const innerW = width - padding.left - padding.right
  const innerH = height - padding.top - padding.bottom

  const T = Math.max(rnnGradients.length, lstmGradients.length)
  if (T === 0) return null

  const xScale = d3.scaleLinear().domain([0, T - 1]).range([0, innerW])

  const allVals = [...rnnGradients, ...lstmGradients].filter((v) => v > 0)
  const yMax = Math.max(...allVals, 1)
  const yScale = d3.scaleLog().domain([Math.max(d3.min(allVals) || 0.001, 0.001), yMax]).range([innerH, 0]).clamp(true)

  const rnnLine = d3
    .line<number>()
    .x((_, i) => xScale(i))
    .y((d) => yScale(Math.max(d, 0.001)))
    .curve(d3.curveMonotoneX)

  const lstmLine = d3
    .line<number>()
    .x((_, i) => xScale(i))
    .y((d) => yScale(Math.max(d, 0.001)))
    .curve(d3.curveMonotoneX)

  return (
    <div className="bg-obsidian-surface/50 rounded-lg border border-obsidian-border p-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] uppercase tracking-wider text-text-tertiary">
          Gradient Magnitude (log scale)
        </span>
      </div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {/* Grid lines */}
          {yScale.ticks(3).map((tick) => (
            <line
              key={tick}
              x1={0}
              x2={innerW}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke="rgba(255,255,255,0.05)"
              strokeDasharray="2,2"
            />
          ))}

          {/* RNN gradient line */}
          <path
            d={rnnLine(rnnGradients) || ''}
            fill="none"
            stroke={COLORS.error}
            strokeWidth={1.5}
            strokeOpacity={0.8}
          />

          {/* LSTM gradient line */}
          <path
            d={lstmLine(lstmGradients) || ''}
            fill="none"
            stroke={COLORS.success}
            strokeWidth={1.5}
            strokeOpacity={0.8}
          />

          {/* Current step indicator */}
          {currentStep < T && (
            <line
              x1={xScale(currentStep)}
              y1={0}
              x2={xScale(currentStep)}
              y2={innerH}
              stroke="rgba(255,255,255,0.2)"
              strokeWidth={1}
              strokeDasharray="3,3"
            />
          )}

          {/* Y axis labels */}
          {yScale.ticks(3).map((tick) => (
            <text
              key={tick}
              x={-4}
              y={yScale(tick) + 3}
              textAnchor="end"
              className="text-[8px] fill-text-tertiary"
            >
              {tick < 0.01 ? tick.toExponential(0) : tick.toFixed(2)}
            </text>
          ))}

          {/* X axis label */}
          <text
            x={innerW / 2}
            y={innerH + 18}
            textAnchor="middle"
            className="text-[8px] fill-text-tertiary"
          >
            Time step (backward from T)
          </text>
        </g>
      </svg>
      <div className="flex items-center gap-4 mt-1 px-1">
        <div className="flex items-center gap-1">
          <div className="w-3 h-[2px] rounded" style={{ backgroundColor: COLORS.error }} />
          <span className="text-[9px] text-text-tertiary">RNN (vanishing)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-[2px] rounded" style={{ backgroundColor: COLORS.success }} />
          <span className="text-[9px] text-text-tertiary">LSTM (stable)</span>
        </div>
      </div>
    </div>
  )
}

// ── LSTM Cell Anatomy ─────────────────────────────────────────────────

function LSTMCellDiagram({
  lstmResult,
  currentStep,
  width,
  height,
}: {
  lstmResult: LSTMResult
  currentStep: number
  width: number
  height: number
}) {
  const step = lstmResult.steps[currentStep]
  if (!step) return null

  const hiddenSize = step.hiddenState.length
  const gateBarW = Math.min(width / (hiddenSize * 4 + 10), 12)
  const barMaxH = height * 0.3

  // Gate data arrays
  const gates = [
    { name: 'Forget', values: step.forgetGate, color: COLORS.clusters[3] },
    { name: 'Input', values: step.inputGate, color: COLORS.clusters[0] },
    { name: 'Output', values: step.outputGate, color: COLORS.clusters[2] },
  ]

  const gateBlockW = (hiddenSize * gateBarW + 8)
  const totalGateW = gates.length * gateBlockW + (gates.length - 1) * 12
  const startX = (width - totalGateW) / 2

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* Cell state conveyor belt */}
      <rect
        x={20}
        y={10}
        width={width - 40}
        height={24}
        rx={4}
        fill="rgba(99, 102, 241, 0.08)"
        stroke="rgba(99, 102, 241, 0.2)"
        strokeWidth={1}
      />
      <text
        x={width / 2}
        y={25}
        textAnchor="middle"
        className="text-[9px] fill-text-secondary font-mono"
      >
        Cell State c_t
      </text>

      {/* Arrows on conveyor belt */}
      <line
        x1={25}
        y1={22}
        x2={15}
        y2={22}
        stroke="rgba(99, 102, 241, 0.3)"
        strokeWidth={1.5}
        markerEnd="url(#rnn-arrow)"
      />
      <line
        x1={width - 25}
        y1={22}
        x2={width - 15}
        y2={22}
        stroke="rgba(99, 102, 241, 0.3)"
        strokeWidth={1.5}
      />

      {/* Gate bar charts */}
      {gates.map((gate, gi) => {
        const blockX = startX + gi * (gateBlockW + 12)
        const baseY = height - 25

        return (
          <g key={gate.name}>
            {/* Gate label */}
            <text
              x={blockX + gateBlockW / 2}
              y={height - 8}
              textAnchor="middle"
              className="text-[8px] fill-text-tertiary"
            >
              {gate.name}
            </text>

            {/* Bars */}
            {gate.values.map((v, i) => {
              const barH = Math.abs(v) * barMaxH
              const barX = blockX + i * gateBarW
              return (
                <g key={i}>
                  {/* Background */}
                  <rect
                    x={barX}
                    y={baseY - barMaxH}
                    width={gateBarW - 1}
                    height={barMaxH}
                    fill="rgba(255,255,255,0.02)"
                    rx={1}
                  />
                  {/* Value bar */}
                  <motion.rect
                    x={barX}
                    y={baseY - barH}
                    width={gateBarW - 1}
                    height={barH}
                    fill={gate.color}
                    fillOpacity={0.6}
                    rx={1}
                    animate={{ y: baseY - barH, height: barH }}
                    transition={{ duration: 0.2 }}
                  />
                </g>
              )
            })}

            {/* 0-1 scale labels */}
            <text
              x={blockX - 3}
              y={baseY + 3}
              textAnchor="end"
              className="text-[7px] fill-text-tertiary"
            >
              0
            </text>
            <text
              x={blockX - 3}
              y={baseY - barMaxH + 5}
              textAnchor="end"
              className="text-[7px] fill-text-tertiary"
            >
              1
            </text>
          </g>
        )
      })}

      {/* Hidden state mini-heatmap */}
      <g>
        <text
          x={width / 2}
          y={52}
          textAnchor="middle"
          className="text-[8px] fill-text-tertiary"
        >
          Hidden State h_t
        </text>
        {step.hiddenState.map((v, i) => {
          const cellW = Math.min(gateBarW, 14)
          const startHeatX = width / 2 - (hiddenSize * cellW) / 2
          const absV = Math.min(Math.abs(v), 1)
          const color = v >= 0 ? COLORS.clusters[0] : COLORS.error
          return (
            <rect
              key={`hs-${i}`}
              x={startHeatX + i * cellW}
              y={56}
              width={cellW - 1}
              height={14}
              rx={1}
              fill={color}
              fillOpacity={absV * 0.7 + 0.05}
            />
          )
        })}
      </g>
    </svg>
  )
}

// ── Hidden State Heatmap Row ──────────────────────────────────────────

function HiddenStateTimeline({
  steps,
  currentStep,
  width,
  height,
}: {
  steps: { hiddenState: number[] }[]
  currentStep: number
  width: number
  height: number
}) {
  const T = steps.length
  if (T === 0) return null
  const hiddenSize = steps[0].hiddenState.length

  const cellW = Math.max(width / T, 2)
  const cellH = Math.max(height / hiddenSize, 3)

  return (
    <div className="bg-obsidian-surface/50 rounded-lg border border-obsidian-border p-2">
      <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-1">
        Hidden State Over Time
      </p>
      <svg width={width} height={Math.max(cellH * hiddenSize, 30)} viewBox={`0 0 ${width} ${cellH * hiddenSize}`}>
        {steps.map((step, t) =>
          step.hiddenState.map((v, h) => {
            const absV = Math.min(Math.abs(v), 1)
            const color = v >= 0 ? COLORS.clusters[0] : COLORS.error
            const isActive = t <= currentStep
            return (
              <rect
                key={`${t}-${h}`}
                x={t * cellW}
                y={h * cellH}
                width={cellW - 0.5}
                height={cellH - 0.5}
                fill={isActive ? color : 'rgba(255,255,255,0.02)'}
                fillOpacity={isActive ? absV * 0.7 + 0.05 : 0.03}
              />
            )
          })
        )}
        {/* Current step indicator */}
        <rect
          x={currentStep * cellW}
          y={0}
          width={cellW}
          height={cellH * hiddenSize}
          fill="none"
          stroke={COLORS.accent}
          strokeWidth={1}
          strokeOpacity={0.5}
        />
      </svg>
      <div className="flex justify-between mt-0.5">
        <span className="text-[7px] text-text-tertiary">t=0</span>
        <span className="text-[7px] text-text-tertiary">t={T - 1}</span>
      </div>
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────

export function RNNLSTMViz() {
  const [hiddenSize, setHiddenSize] = useState(4)
  const [seqType, setSeqType] = useState<'sine' | 'sawtooth' | 'square'>('sine')
  const [showLSTM, setShowLSTM] = useState(false)

  const sequence = useMemo(() => makeSequence(SEQ_LENGTH, seqType, 42), [seqType])

  const rnnResult = useMemo(
    () => runRNN(sequence, { hiddenSize, seed: 42 }),
    [sequence, hiddenSize]
  )

  const lstmResult = useMemo(
    () => runLSTM(sequence, { hiddenSize, seed: 42 }),
    [sequence, hiddenSize]
  )

  // Build snapshots for the player (one per time step)
  const snapshots = useMemo(() => {
    const active = showLSTM ? lstmResult.steps : rnnResult.steps
    return active.map((_, i) => i)
  }, [showLSTM, rnnResult, lstmResult])

  const player = useAlgorithmPlayer({ snapshots, baseFps: 2 })
  const currentTimeStep = player.currentSnapshot

  return (
    <GlassCard className="p-6 lg:p-8">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 mb-6">
        <Slider
          label="Hidden Size"
          value={hiddenSize}
          min={2}
          max={8}
          step={1}
          onChange={(v) => {
            setHiddenSize(v)
            player.reset()
          }}
          className="w-32"
        />
        <Select
          label="Sequence"
          value={seqType}
          options={SEQUENCE_OPTIONS}
          onChange={(v) => {
            setSeqType(v as 'sine' | 'sawtooth' | 'square')
            player.reset()
          }}
          className="w-36"
        />
        <Toggle
          label="Show LSTM"
          checked={showLSTM}
          onChange={(v) => {
            setShowLSTM(v)
            player.reset()
          }}
        />
      </div>

      {/* Two-section layout */}
      <div className="space-y-6">
        {/* RNN Section: Unrolled diagram */}
        {!showLSTM && (
          <div>
            <p className="text-xs font-medium text-text-secondary mb-2">
              Vanilla RNN -- Unrolled Through Time
            </p>
            <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-3 overflow-x-auto">
              <UnrolledRNNDiagram
                rnnResult={rnnResult}
                currentStep={currentTimeStep}
                width={Math.max(700, SEQ_LENGTH * 50)}
                height={200}
              />
            </div>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
              <HiddenStateTimeline
                steps={rnnResult.steps}
                currentStep={currentTimeStep}
                width={350}
                height={80}
              />
              <GradientChart
                rnnGradients={rnnResult.gradientMagnitudes}
                lstmGradients={lstmResult.gradientMagnitudes}
                currentStep={currentTimeStep}
                width={350}
                height={120}
              />
            </div>
            <p className="mt-2 text-[10px] text-text-tertiary leading-relaxed max-w-xl">
              Notice how the gradient magnitude decays exponentially as it propagates backward through time
              (red line). This is the vanishing gradient problem -- the RNN cannot learn long-range
              dependencies because early time steps receive negligibly small gradient signals.
            </p>
          </div>
        )}

        {/* LSTM Section: Cell anatomy + gates */}
        {showLSTM && (
          <div>
            <p className="text-xs font-medium text-text-secondary mb-2">
              LSTM Cell -- Gate Activations at t={currentTimeStep}
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-3">
                <LSTMCellDiagram
                  lstmResult={lstmResult}
                  currentStep={currentTimeStep}
                  width={380}
                  height={260}
                />
              </div>
              <div className="space-y-3">
                <HiddenStateTimeline
                  steps={lstmResult.steps}
                  currentStep={currentTimeStep}
                  width={350}
                  height={80}
                />
                <GradientChart
                  rnnGradients={rnnResult.gradientMagnitudes}
                  lstmGradients={lstmResult.gradientMagnitudes}
                  currentStep={currentTimeStep}
                  width={350}
                  height={120}
                />
              </div>
            </div>
            <p className="mt-2 text-[10px] text-text-tertiary leading-relaxed max-w-xl">
              The LSTM's cell state acts as a "gradient highway" -- the forget gate controls how much
              of the gradient flows through unchanged (green line stays high). This is why LSTMs can
              learn long-range dependencies where vanilla RNNs fail.
            </p>

            {/* Gate explanations */}
            <div className="mt-3 grid grid-cols-3 gap-2">
              {[
                {
                  name: 'Forget Gate',
                  desc: 'Controls what to erase from cell state',
                  color: COLORS.clusters[3],
                },
                {
                  name: 'Input Gate',
                  desc: 'Controls what new info to store',
                  color: COLORS.clusters[0],
                },
                {
                  name: 'Output Gate',
                  desc: 'Controls what to output from cell',
                  color: COLORS.clusters[2],
                },
              ].map((gate) => (
                <div
                  key={gate.name}
                  className="bg-obsidian-surface/30 rounded-lg border border-obsidian-border p-2"
                >
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: gate.color }} />
                    <span className="text-[10px] font-medium text-text-secondary">{gate.name}</span>
                  </div>
                  <p className="text-[9px] text-text-tertiary">{gate.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Transport controls */}
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
