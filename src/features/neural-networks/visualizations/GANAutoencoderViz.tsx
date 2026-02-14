import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { createRng, normalRandom } from '../../../lib/math/random'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

// ── Autoencoder Architecture Diagram ──────────────────────────────────

function AutoencoderDiagram({
  width,
  height,
  animationProgress,
}: {
  width: number
  height: number
  animationProgress: number
}) {
  const margin = { top: 20, bottom: 30, left: 30, right: 30 }
  const innerW = width - margin.left - margin.right
  const innerH = height - margin.top - margin.bottom

  // Layer widths represent the encoding/decoding progression
  const layerWidths = [0.9, 0.7, 0.4, 0.2, 0.4, 0.7, 0.9]
  const labels = ['Input', 'Enc 1', 'Enc 2', 'Bottleneck', 'Dec 1', 'Dec 2', 'Output']
  const numLayers = layerWidths.length
  const layerSpacing = innerW / (numLayers - 1)

  // Data flow animation: a "pulse" moving through layers
  const pulseLayer = Math.floor(animationProgress * (numLayers - 1))
  const pulseFraction = (animationProgress * (numLayers - 1)) % 1

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <g transform={`translate(${margin.left}, ${margin.top})`}>
        {/* Connect adjacent layers */}
        {layerWidths.map((w, i) => {
          if (i >= numLayers - 1) return null
          const x1 = i * layerSpacing
          const x2 = (i + 1) * layerSpacing
          const h1 = w * innerH
          const h2 = layerWidths[i + 1] * innerH
          const y1Top = (innerH - h1) / 2
          const y1Bot = y1Top + h1
          const y2Top = (innerH - h2) / 2
          const y2Bot = y2Top + h2

          const isPassed = i < pulseLayer
          const isActive = i === pulseLayer

          return (
            <g key={`conn-${i}`}>
              {/* Trapezoid fill */}
              <path
                d={`M${x1},${y1Top} L${x2},${y2Top} L${x2},${y2Bot} L${x1},${y1Bot} Z`}
                fill={
                  i < 3
                    ? `rgba(99, 102, 241, ${isPassed || isActive ? 0.08 : 0.03})`
                    : `rgba(52, 211, 153, ${isPassed || isActive ? 0.08 : 0.03})`
                }
                stroke="none"
              />
              {/* Top line */}
              <line
                x1={x1}
                y1={y1Top}
                x2={x2}
                y2={y2Top}
                stroke={i < 3 ? COLORS.clusters[0] : COLORS.clusters[2]}
                strokeWidth={1}
                strokeOpacity={isPassed || isActive ? 0.4 : 0.12}
              />
              {/* Bottom line */}
              <line
                x1={x1}
                y1={y1Bot}
                x2={x2}
                y2={y2Bot}
                stroke={i < 3 ? COLORS.clusters[0] : COLORS.clusters[2]}
                strokeWidth={1}
                strokeOpacity={isPassed || isActive ? 0.4 : 0.12}
              />
            </g>
          )
        })}

        {/* Layer bars */}
        {layerWidths.map((w, i) => {
          const x = i * layerSpacing
          const barH = w * innerH
          const y = (innerH - barH) / 2
          const barW = 8
          const isBottleneck = i === 3
          const isPassed = i <= pulseLayer
          const color = isBottleneck
            ? COLORS.accent
            : i < 3
              ? COLORS.clusters[0]
              : COLORS.clusters[2]

          return (
            <g key={`bar-${i}`}>
              <motion.rect
                x={x - barW / 2}
                y={y}
                width={barW}
                height={barH}
                rx={3}
                fill={color}
                fillOpacity={isPassed ? 0.5 : 0.15}
                stroke={color}
                strokeWidth={isBottleneck ? 1.5 : 1}
                strokeOpacity={isPassed ? 0.6 : 0.2}
                animate={{
                  fillOpacity: isPassed ? 0.5 : 0.15,
                }}
                transition={{ duration: 0.3 }}
              />
              <text
                x={x}
                y={innerH + 14}
                textAnchor="middle"
                className="text-[7px] fill-text-tertiary"
              >
                {labels[i]}
              </text>
            </g>
          )
        })}

        {/* Data flow pulse indicator */}
        {pulseLayer < numLayers - 1 && (
          <motion.circle
            cx={pulseLayer * layerSpacing + pulseFraction * layerSpacing}
            cy={innerH / 2}
            r={4}
            fill="#fff"
            fillOpacity={0.6}
            animate={{
              cx: pulseLayer * layerSpacing + pulseFraction * layerSpacing,
            }}
            transition={{ duration: 0.1 }}
          />
        )}

        {/* Encoder/Decoder labels */}
        <text
          x={1.5 * layerSpacing}
          y={-6}
          textAnchor="middle"
          className="text-[9px] fill-text-secondary"
        >
          Encoder
        </text>
        <text
          x={5 * layerSpacing}
          y={-6}
          textAnchor="middle"
          className="text-[9px] fill-text-secondary"
        >
          Decoder
        </text>

        {/* Compression arrows */}
        <line
          x1={0.5 * layerSpacing}
          y1={-2}
          x2={2.5 * layerSpacing}
          y2={-2}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={1}
        />
        <line
          x1={4 * layerSpacing}
          y1={-2}
          x2={6 * layerSpacing}
          y2={-2}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={1}
        />
      </g>
    </svg>
  )
}

// ── Simple 1D GAN Training ────────────────────────────────────────────

interface GANSnapshot {
  round: number
  realHistogram: number[]
  generatedHistogram: number[]
  generatorMean: number
  generatorStd: number
  discriminatorLoss: number
  generatorLoss: number
}

function trainSimpleGAN(
  targetMean: number,
  targetStd: number,
  rounds: number,
  seed: number = 42
): GANSnapshot[] {
  const rng = createRng(seed)
  const nBins = 30
  const binRange: [number, number] = [-4, 4]
  const binWidth = (binRange[1] - binRange[0]) / nBins

  // Generator parameters (learns to match target distribution)
  let genMean = normalRandom(rng, 0, 0.5)
  let genStd = 0.5 + rng() * 0.5
  const lr = 0.05

  const snapshots: GANSnapshot[] = []

  // Build real histogram
  const buildHistogram = (mean: number, std: number, n: number): number[] => {
    const hist = new Array(nBins).fill(0)
    const localRng = createRng(seed + 9999)
    for (let i = 0; i < n; i++) {
      const v = normalRandom(localRng, mean, std)
      const bin = Math.floor((v - binRange[0]) / binWidth)
      if (bin >= 0 && bin < nBins) hist[bin]++
    }
    // Normalize
    const total = hist.reduce((a: number, b: number) => a + b, 0)
    return hist.map((h: number) => (total > 0 ? h / total : 0))
  }

  const realHist = buildHistogram(targetMean, targetStd, 500)

  // Initial snapshot
  const genHist0 = buildHistogram(genMean, genStd, 500)
  snapshots.push({
    round: 0,
    realHistogram: realHist,
    generatedHistogram: genHist0,
    generatorMean: genMean,
    generatorStd: genStd,
    discriminatorLoss: 1,
    generatorLoss: 1,
  })

  for (let r = 1; r <= rounds; r++) {
    // Simple gradient: push generator mean toward target mean
    const meanGrad = targetMean - genMean
    const stdGrad = targetStd - genStd

    // Add noise for realism
    genMean += lr * meanGrad + normalRandom(rng, 0, 0.01)
    genStd += lr * 0.5 * stdGrad + normalRandom(rng, 0, 0.005)
    genStd = Math.max(genStd, 0.1) // Clamp std

    const genHist = buildHistogram(genMean, genStd, 500)

    // Compute a simple "distance" as proxy for loss
    let dist = 0
    for (let b = 0; b < nBins; b++) {
      dist += Math.abs(realHist[b] - genHist[b])
    }
    const dLoss = dist
    const gLoss = dist

    snapshots.push({
      round: r,
      realHistogram: realHist,
      generatedHistogram: genHist,
      generatorMean: genMean,
      generatorStd: genStd,
      discriminatorLoss: dLoss,
      generatorLoss: gLoss,
    })
  }

  return snapshots
}

// ── GAN Distribution Visualization ────────────────────────────────────

function GANDistributionChart({
  snapshot,
  width,
  height,
}: {
  snapshot: GANSnapshot
  width: number
  height: number
}) {
  const padding = { top: 15, right: 15, bottom: 25, left: 35 }
  const innerW = width - padding.left - padding.right
  const innerH = height - padding.top - padding.bottom

  const nBins = snapshot.realHistogram.length
  const barWidth = innerW / nBins

  const maxVal = Math.max(
    ...snapshot.realHistogram,
    ...snapshot.generatedHistogram,
    0.01
  )
  const yScale = d3.scaleLinear().domain([0, maxVal]).range([innerH, 0])

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <g transform={`translate(${padding.left}, ${padding.top})`}>
        {/* Real distribution bars */}
        {snapshot.realHistogram.map((v, i) => (
          <rect
            key={`real-${i}`}
            x={i * barWidth + 1}
            y={yScale(v)}
            width={barWidth - 2}
            height={innerH - yScale(v)}
            fill={COLORS.clusters[0]}
            fillOpacity={0.3}
            rx={1}
          />
        ))}

        {/* Generated distribution bars */}
        {snapshot.generatedHistogram.map((v, i) => (
          <motion.rect
            key={`gen-${i}`}
            x={i * barWidth + 1}
            y={yScale(v)}
            width={barWidth - 2}
            height={innerH - yScale(v)}
            fill={COLORS.clusters[1]}
            fillOpacity={0.35}
            rx={1}
            animate={{
              y: yScale(v),
              height: innerH - yScale(v),
            }}
            transition={{ duration: 0.15 }}
          />
        ))}

        {/* Baseline */}
        <line
          x1={0}
          y1={innerH}
          x2={innerW}
          y2={innerH}
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={1}
        />

        {/* Y axis ticks */}
        {yScale.ticks(3).map((t) => (
          <g key={t}>
            <line
              x1={0}
              x2={innerW}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="2,2"
            />
            <text
              x={-4}
              y={yScale(t) + 3}
              textAnchor="end"
              className="text-[8px] fill-text-tertiary"
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}
      </g>
    </svg>
  )
}

// ── GAN Training Loop Diagram ─────────────────────────────────────────

function GANArchDiagram({
  width,
  height,
  round,
}: {
  width: number
  height: number
  round: number
}) {
  const cx = width / 2
  const genX = cx - 80
  const discX = cx + 80
  const y = height / 2

  // Animate a "training pulse" cycling between G and D
  const isGeneratorTurn = round % 2 === 0
  const genGlow = isGeneratorTurn ? 0.5 : 0.15
  const discGlow = !isGeneratorTurn ? 0.5 : 0.15

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* Arrows forming a loop */}
      {/* G -> "Fake Data" -> D */}
      <line
        x1={genX + 35}
        y1={y - 8}
        x2={discX - 35}
        y2={y - 8}
        stroke="rgba(255,255,255,0.2)"
        strokeWidth={1}
        strokeDasharray="4,3"
      />
      <text
        x={cx}
        y={y - 14}
        textAnchor="middle"
        className="text-[8px] fill-text-tertiary"
      >
        Fake samples
      </text>

      {/* D -> "Feedback" -> G */}
      <line
        x1={discX - 35}
        y1={y + 8}
        x2={genX + 35}
        y2={y + 8}
        stroke="rgba(255,255,255,0.2)"
        strokeWidth={1}
        strokeDasharray="4,3"
      />
      <text
        x={cx}
        y={y + 22}
        textAnchor="middle"
        className="text-[8px] fill-text-tertiary"
      >
        Gradient signal
      </text>

      {/* Generator box */}
      <motion.rect
        x={genX - 32}
        y={y - 22}
        width={64}
        height={44}
        rx={8}
        fill={COLORS.clusters[1]}
        fillOpacity={genGlow}
        stroke={COLORS.clusters[1]}
        strokeWidth={1.5}
        strokeOpacity={0.5}
        animate={{ fillOpacity: genGlow }}
        transition={{ duration: 0.3 }}
      />
      <text
        x={genX}
        y={y + 4}
        textAnchor="middle"
        className="text-[10px] fill-text-primary font-medium"
      >
        Generator
      </text>

      {/* Discriminator box */}
      <motion.rect
        x={discX - 32}
        y={y - 22}
        width={64}
        height={44}
        rx={8}
        fill={COLORS.clusters[2]}
        fillOpacity={discGlow}
        stroke={COLORS.clusters[2]}
        strokeWidth={1.5}
        strokeOpacity={0.5}
        animate={{ fillOpacity: discGlow }}
        transition={{ duration: 0.3 }}
      />
      <text
        x={discX}
        y={y + 4}
        textAnchor="middle"
        className="text-[10px] fill-text-primary font-medium"
      >
        Discriminator
      </text>

      {/* Noise input to Generator */}
      <text
        x={genX - 55}
        y={y + 4}
        textAnchor="end"
        className="text-[9px] fill-text-tertiary font-mono"
      >
        z ~ N(0,1)
      </text>
      <line
        x1={genX - 50}
        y1={y}
        x2={genX - 34}
        y2={y}
        stroke="rgba(255,255,255,0.15)"
        strokeWidth={1}
      />

      {/* Real data to Discriminator */}
      <text
        x={discX + 55}
        y={y + 4}
        textAnchor="start"
        className="text-[9px] fill-text-tertiary font-mono"
      >
        Real data
      </text>
      <line
        x1={discX + 34}
        y1={y}
        x2={discX + 50}
        y2={y}
        stroke="rgba(255,255,255,0.15)"
        strokeWidth={1}
      />

      {/* Output labels */}
      <text
        x={discX}
        y={y + 38}
        textAnchor="middle"
        className="text-[8px] fill-text-tertiary"
      >
        Real or Fake?
      </text>
    </svg>
  )
}

// ── Main Component ─────────────────────────────────────────────────────

export function GANAutoencoderViz() {
  const [seed] = useState(42)

  // Autoencoder animation: cycle through layers
  const aeSnapshots = useMemo(
    () => Array.from({ length: 30 }, (_, i) => i / 29),
    []
  )
  const aePlayer = useAlgorithmPlayer({ snapshots: aeSnapshots, baseFps: 3 })

  // GAN training
  const ganSnapshots = useMemo(
    () => trainSimpleGAN(0, 1, 60, seed),
    [seed]
  )
  const ganPlayer = useAlgorithmPlayer({ snapshots: ganSnapshots, baseFps: 4 })
  const ganSnap = ganPlayer.currentSnapshot

  return (
    <div className="space-y-6">
      {/* ── Autoencoder Section ──────────────────────────────────────── */}
      <GlassCard className="p-6 lg:p-8">
        <h3 className="text-sm font-semibold text-text-primary mb-1">Autoencoder</h3>
        <p className="text-xs text-text-secondary mb-4 max-w-lg leading-relaxed">
          An autoencoder learns to compress data through a narrow bottleneck and then reconstruct it.
          The bottleneck forces the network to learn the most important features -- a compact
          representation of the input.
        </p>

        <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
          <AutoencoderDiagram
            width={500}
            height={200}
            animationProgress={aePlayer.currentSnapshot}
          />
        </div>

        <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
          {[
            {
              title: 'Encoding',
              desc: 'Compress input to a lower-dimensional representation',
              color: COLORS.clusters[0],
            },
            {
              title: 'Bottleneck',
              desc: 'The compressed representation (latent space)',
              color: COLORS.accent,
            },
            {
              title: 'Decoding',
              desc: 'Reconstruct the original input from the compressed form',
              color: COLORS.clusters[2],
            },
          ].map((item) => (
            <div
              key={item.title}
              className="bg-obsidian-surface/30 rounded-lg border border-obsidian-border p-3"
            >
              <div className="flex items-center gap-1.5 mb-1">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-[10px] font-medium text-text-secondary">{item.title}</span>
              </div>
              <p className="text-[9px] text-text-tertiary">{item.desc}</p>
            </div>
          ))}
        </div>

        <div className="mt-3">
          <TransportControls
            isPlaying={aePlayer.isPlaying}
            isAtStart={aePlayer.isAtStart}
            isAtEnd={aePlayer.isAtEnd}
            currentStep={aePlayer.currentStep}
            totalSteps={aePlayer.totalSteps}
            speed={aePlayer.speed}
            onPlay={aePlayer.play}
            onPause={aePlayer.pause}
            onTogglePlay={aePlayer.togglePlay}
            onStepForward={aePlayer.stepForward}
            onStepBack={aePlayer.stepBack}
            onReset={aePlayer.reset}
            onSetSpeed={aePlayer.setSpeed}
          />
        </div>
      </GlassCard>

      {/* ── GAN Section ──────────────────────────────────────────────── */}
      <GlassCard className="p-6 lg:p-8">
        <h3 className="text-sm font-semibold text-text-primary mb-1">
          Generative Adversarial Network (GAN)
        </h3>
        <p className="text-xs text-text-secondary mb-4 max-w-lg leading-relaxed">
          A GAN pits two networks against each other: a Generator that creates fake data and a
          Discriminator that tries to tell real from fake. Through this adversarial game, the
          Generator learns to produce increasingly realistic data.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Architecture diagram */}
          <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-3">
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-2">
              Training Loop
            </p>
            <GANArchDiagram width={380} height={120} round={ganSnap.round} />
          </div>

          {/* Distribution matching */}
          <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-3">
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-2">
              Distribution Matching (Round {ganSnap.round})
            </p>
            <GANDistributionChart
              snapshot={ganSnap}
              width={350}
              height={150}
            />
            <div className="flex items-center gap-4 mt-1 px-1">
              <div className="flex items-center gap-1">
                <div
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: COLORS.clusters[0], opacity: 0.5 }}
                />
                <span className="text-[9px] text-text-tertiary">Real distribution</span>
              </div>
              <div className="flex items-center gap-1">
                <div
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: COLORS.clusters[1], opacity: 0.5 }}
                />
                <span className="text-[9px] text-text-tertiary">Generated distribution</span>
              </div>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="mt-3 flex flex-wrap gap-4">
          <div className="bg-obsidian-surface/30 rounded-lg border border-obsidian-border px-3 py-2">
            <span className="text-[9px] text-text-tertiary">Generator</span>
            <p className="text-xs font-mono text-text-secondary">
              \u03BC={ganSnap.generatorMean.toFixed(2)}, \u03C3={ganSnap.generatorStd.toFixed(2)}
            </p>
          </div>
          <div className="bg-obsidian-surface/30 rounded-lg border border-obsidian-border px-3 py-2">
            <span className="text-[9px] text-text-tertiary">Target</span>
            <p className="text-xs font-mono text-text-secondary">
              \u03BC=0.00, \u03C3=1.00
            </p>
          </div>
          <div className="bg-obsidian-surface/30 rounded-lg border border-obsidian-border px-3 py-2">
            <span className="text-[9px] text-text-tertiary">Distance</span>
            <p className="text-xs font-mono text-text-secondary">
              {ganSnap.generatorLoss.toFixed(3)}
            </p>
          </div>
        </div>

        <div className="mt-3">
          <TransportControls
            isPlaying={ganPlayer.isPlaying}
            isAtStart={ganPlayer.isAtStart}
            isAtEnd={ganPlayer.isAtEnd}
            currentStep={ganPlayer.currentStep}
            totalSteps={ganPlayer.totalSteps}
            speed={ganPlayer.speed}
            onPlay={ganPlayer.play}
            onPause={ganPlayer.pause}
            onTogglePlay={ganPlayer.togglePlay}
            onStepForward={ganPlayer.stepForward}
            onStepBack={ganPlayer.stepBack}
            onReset={ganPlayer.reset}
            onSetSpeed={ganPlayer.setSpeed}
          />
        </div>
      </GlassCard>
    </div>
  )
}
