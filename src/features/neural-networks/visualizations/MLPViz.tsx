import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import {
  makeXOR,
  makeMoons,
  makeConcentricCircles,
  makeSpirals,
  makeGaussianBlobs,
} from '../../../lib/data/nnDataGenerators'
import { runMLPTraining, type MLPSnapshot } from '../../../lib/algorithms/neural-networks/mlp'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { ConvergenceChart } from '../../../components/viz/ConvergenceChart'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { Select } from '../../../components/ui/Select'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

// ── Dataset factory ────────────────────────────────────────────────────

function makeDataset(name: string) {
  switch (name) {
    case 'xor':
      return makeXOR(120)
    case 'moons':
      return makeMoons(150, 0.12)
    case 'circles':
      return makeConcentricCircles(150, 0.06)
    case 'spirals':
      return makeSpirals(150, 0.25)
    case 'blobs':
      return makeGaussianBlobs(120, 3)
    default:
      return makeMoons(150, 0.12)
  }
}

const DATASET_OPTIONS = [
  { value: 'xor', label: 'XOR' },
  { value: 'moons', label: 'Moons' },
  { value: 'circles', label: 'Circles' },
  { value: 'spirals', label: 'Spirals' },
  { value: 'blobs', label: 'Blobs' },
]

const ACTIVATION_OPTIONS = [
  { value: 'relu', label: 'ReLU' },
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'tanh', label: 'Tanh' },
]

// ── Network Diagram ────────────────────────────────────────────────────

function NetworkDiagram({
  layerSizes,
  snapshot,
  width,
  height,
}: {
  layerSizes: number[]
  snapshot: MLPSnapshot
  width: number
  height: number
}) {
  const numLayers = layerSizes.length
  const layerSpacing = width / (numLayers + 1)

  // Compute node positions
  const nodePositions: { x: number; y: number }[][] = []
  for (let l = 0; l < numLayers; l++) {
    const n = layerSizes[l]
    const x = (l + 1) * layerSpacing
    const ySpacing = Math.min((height - 40) / (n + 1), 36)
    const yStart = height / 2 - ((n - 1) * ySpacing) / 2
    const layer: { x: number; y: number }[] = []
    for (let i = 0; i < n; i++) {
      layer.push({ x, y: yStart + i * ySpacing })
    }
    nodePositions.push(layer)
  }

  // Find max weight for scaling
  let maxWeight = 0.01
  if (snapshot.weights) {
    for (const layer of snapshot.weights) {
      for (const row of layer) {
        for (const w of row) {
          const abs = Math.abs(w)
          if (abs > maxWeight) maxWeight = abs
        }
      }
    }
  }

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* Edges colored by weight sign, thickness by magnitude */}
      {snapshot.weights &&
        snapshot.weights.map((layerWeights, l) =>
          layerWeights.map((fromWeights, i) =>
            fromWeights.map((w, j) => {
              const from = nodePositions[l]?.[i]
              const to = nodePositions[l + 1]?.[j]
              if (!from || !to) return null
              const thickness = 0.5 + (Math.abs(w) / maxWeight) * 3.5
              const color = w >= 0 ? COLORS.clusters[0] : COLORS.error
              const opacity = 0.15 + (Math.abs(w) / maxWeight) * 0.4
              return (
                <line
                  key={`e-${l}-${i}-${j}`}
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={color}
                  strokeWidth={thickness}
                  strokeOpacity={opacity}
                />
              )
            })
          )
        )}

      {/* Nodes */}
      {nodePositions.map((layer, l) =>
        layer.map((pos, i) => {
          const isInput = l === 0
          const isOutput = l === numLayers - 1
          const nodeColor = isInput
            ? 'rgba(255,255,255,0.08)'
            : isOutput
              ? COLORS.accent
              : 'rgba(255,255,255,0.06)'
          const nodeStroke = isOutput
            ? COLORS.accent
            : 'rgba(255,255,255,0.15)'
          return (
            <g key={`n-${l}-${i}`}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={9}
                fill={nodeColor}
                stroke={nodeStroke}
                strokeWidth={1}
              />
              {isInput && (
                <text
                  x={pos.x}
                  y={pos.y + 3.5}
                  textAnchor="middle"
                  className="text-[7px] fill-text-secondary font-mono"
                >
                  x{i + 1}
                </text>
              )}
              {isOutput && (
                <text
                  x={pos.x}
                  y={pos.y + 3.5}
                  textAnchor="middle"
                  className="text-[7px] fill-text-primary font-mono"
                >
                  y
                </text>
              )}
            </g>
          )
        })
      )}

      {/* Layer labels */}
      {nodePositions.map((layer, l) => (
        <text
          key={`label-${l}`}
          x={layer[0].x}
          y={height - 2}
          textAnchor="middle"
          className="text-[7px] fill-text-tertiary"
        >
          {l === 0 ? 'Input' : l === numLayers - 1 ? 'Output' : `H${l}`}
        </text>
      ))}
    </svg>
  )
}

// ── Main Component ─────────────────────────────────────────────────────

export function MLPViz() {
  const [hiddenLayers, setHiddenLayers] = useState(2)
  const [neuronsPerLayer, setNeuronsPerLayer] = useState(4)
  const [activationName, setActivationName] = useState<'relu' | 'sigmoid' | 'tanh'>('relu')
  const [learningRate, setLearningRate] = useState(0.3)
  const [datasetName, setDatasetName] = useState('moons')
  const [seed, setSeed] = useState(42)

  const data = useMemo(() => makeDataset(datasetName), [datasetName])

  const layerSizes = useMemo(() => {
    const sizes = [2]
    for (let i = 0; i < hiddenLayers; i++) {
      sizes.push(neuronsPerLayer)
    }
    sizes.push(1)
    return sizes
  }, [hiddenLayers, neuronsPerLayer])

  const snapshots = useMemo(
    () =>
      runMLPTraining(data, {
        layerSizes,
        learningRate,
        activation: activationName,
        epochs: 100,
        seed,
      }),
    [data, layerSizes, learningRate, activationName, seed]
  )

  const player = useAlgorithmPlayer({ snapshots, baseFps: 6 })
  const snap = player.currentSnapshot

  const handleReset = useCallback(() => {
    setSeed((s) => s + 1)
    player.reset()
  }, [player])

  const losses = useMemo(() => snapshots.map((s) => s.trainLoss), [snapshots])

  return (
    <GlassCard className="p-6 lg:p-8">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 mb-6">
        <Select
          label="Dataset"
          value={datasetName}
          options={DATASET_OPTIONS}
          onChange={(v) => {
            setDatasetName(v)
            player.reset()
          }}
          className="w-32"
        />
        <Slider
          label="Hidden Layers"
          value={hiddenLayers}
          min={1}
          max={3}
          step={1}
          onChange={(v) => {
            setHiddenLayers(v)
            player.reset()
          }}
          className="w-36"
        />
        <Slider
          label="Neurons / Layer"
          value={neuronsPerLayer}
          min={2}
          max={6}
          step={1}
          onChange={(v) => {
            setNeuronsPerLayer(v)
            player.reset()
          }}
          className="w-36"
        />
        <Select
          label="Activation"
          value={activationName}
          options={ACTIVATION_OPTIONS}
          onChange={(v) => {
            setActivationName(v as 'relu' | 'sigmoid' | 'tanh')
            player.reset()
          }}
          className="w-28"
        />
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
          className="w-36"
        />
      </div>

      {/* Triple panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left -- Network architecture diagram */}
        <div className="lg:col-span-3 bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
          <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-2">
            Network Architecture
          </p>
          <NetworkDiagram
            layerSizes={layerSizes}
            snapshot={snap}
            width={200}
            height={220}
          />
          <div className="mt-2 space-y-0.5 text-xs text-text-tertiary font-mono">
            <p>Epoch {snap.epoch}</p>
            <p>
              Accuracy: <span className="text-success">{(snap.trainAccuracy * 100).toFixed(1)}%</span>
            </p>
            <p className="text-[10px] mt-1 text-text-tertiary">
              {layerSizes.map((s) => s).join(' \u2192 ')}
            </p>
          </div>
        </div>

        {/* Center -- Decision boundary scatter plot with heatmap */}
        <div className="lg:col-span-6">
          <SVGContainer
            aspectRatio={1}
            minHeight={300}
            maxHeight={450}
            padding={{ top: 15, right: 15, bottom: 30, left: 40 }}
          >
            {({ innerWidth, innerHeight }) => {
              const xExtent = d3.extent(data, (d) => d.x) as [number, number]
              const yExtent = d3.extent(data, (d) => d.y) as [number, number]
              const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
              const yScale = d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice()

              // Decision boundary grid from the snapshot
              const grid = snap.decisionBoundaryGrid || []
              const gridSize = Math.round(Math.sqrt(grid.length))

              return (
                <>
                  {/* Confidence heatmap */}
                  {grid.map((cell, i) => {
                    const color = d3.interpolateRgb(COLORS.clusters[1], COLORS.clusters[0])(cell.value)
                    // Compute cell dimensions from data-space coordinates
                    const px = xScale(cell.x)
                    const py = yScale(cell.y)
                    const halfCellW = gridSize > 1 ? innerWidth / gridSize / 2 + 0.5 : innerWidth / 2
                    const halfCellH = gridSize > 1 ? innerHeight / gridSize / 2 + 0.5 : innerHeight / 2
                    return (
                      <rect
                        key={`g-${i}`}
                        x={px - halfCellW}
                        y={py - halfCellH}
                        width={halfCellW * 2}
                        height={halfCellH * 2}
                        fill={color}
                        fillOpacity={0.2}
                      />
                    )
                  })}

                  {/* Data points */}
                  {data.map((pt, i) => (
                    <circle
                      key={i}
                      cx={xScale(pt.x)}
                      cy={yScale(pt.y)}
                      r={3.5}
                      fill={pt.label === 1 ? COLORS.clusters[0] : COLORS.clusters[1]}
                      fillOpacity={0.85}
                      stroke="#0F0F11"
                      strokeWidth={0.5}
                    />
                  ))}

                  {/* Axis tick labels */}
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

        {/* Right -- Loss curve and stats */}
        <div className="lg:col-span-3 space-y-4">
          <ConvergenceChart
            values={losses}
            currentIndex={player.currentStep}
            label="Train Loss (BCE)"
            width={200}
            height={100}
            color={COLORS.error}
          />
          <div className="bg-obsidian-surface/50 rounded-lg border border-obsidian-border p-3">
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-2">
              Configuration
            </p>
            <div className="space-y-1 text-xs font-mono text-text-tertiary">
              <p>
                Layers: {layerSizes.join(' \u2192 ')}
              </p>
              <p>Activation: {activationName}</p>
              <p>LR: {learningRate.toFixed(2)}</p>
              <p>
                Loss: <span className="text-text-secondary">{snap.trainLoss.toFixed(4)}</span>
              </p>
              <p>
                Acc: <span className="text-success">{(snap.trainAccuracy * 100).toFixed(1)}%</span>
              </p>
            </div>
          </div>
          <div className="text-[10px] text-text-tertiary leading-relaxed">
            <p className="mb-1 font-medium text-text-secondary">Legend</p>
            <div className="flex items-center gap-1.5">
              <div className="w-6 h-[2px] rounded" style={{ backgroundColor: COLORS.clusters[0] }} />
              <span>Positive weight</span>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <div className="w-6 h-[2px] rounded" style={{ backgroundColor: COLORS.error }} />
              <span>Negative weight</span>
            </div>
          </div>
        </div>
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
          onReset={handleReset}
          onSetSpeed={player.setSpeed}
        />
      </div>
    </GlassCard>
  )
}
