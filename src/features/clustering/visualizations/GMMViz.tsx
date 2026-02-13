import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { makeBlobs } from '../../../lib/data/clusterGenerators'
import { runGMM } from '../../../lib/algorithms/clustering/gmm'
import { eigen2x2Symmetric } from '../../../lib/math/linalg'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { ConvergenceChart } from '../../../components/viz/ConvergenceChart'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { Toggle } from '../../../components/ui/Toggle'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

function blendColors(colors: string[], weights: number[]): string {
  let r = 0, g = 0, b = 0
  for (let i = 0; i < colors.length; i++) {
    const c = d3.color(colors[i]) as d3.RGBColor
    if (!c) continue
    r += c.r * weights[i]
    g += c.g * weights[i]
    b += c.b * weights[i]
  }
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`
}

export function GMMViz() {
  const [k, setK] = useState(3)
  const [seed, setSeed] = useState(42)
  const [softAssignment, setSoftAssignment] = useState(true)

  const data = useMemo(() => makeBlobs(150, 3, 1.5, 42), [])

  const snapshots = useMemo(() => runGMM(data, k, seed, 40), [data, k, seed])

  const player = useAlgorithmPlayer({ snapshots, baseFps: 1.5 })
  const snap = player.currentSnapshot

  const logLikelihoods = useMemo(() => snapshots.map((s) => s.logLikelihood), [snapshots])

  const handleReset = useCallback(() => {
    setSeed((s) => s + 1)
    player.reset()
  }, [player])

  return (
    <GlassCard className="p-6">
      <div className="flex flex-wrap gap-6 mb-6">
        <Slider
          label="k (components)"
          value={k}
          min={2}
          max={6}
          step={1}
          onChange={(v) => { setK(v); player.reset() }}
          className="w-48"
        />
        <Toggle
          label={softAssignment ? 'Soft assignment' : 'Hard assignment'}
          checked={softAssignment}
          onChange={setSoftAssignment}
        />
      </div>

      <SVGContainer aspectRatio={16 / 10} minHeight={350} maxHeight={550}>
        {({ innerWidth, innerHeight }) => {
          const xExtent = d3.extent(data, (d) => d.x) as [number, number]
          const yExtent = d3.extent(data, (d) => d.y) as [number, number]
          const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
          const yScale = d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice()

          // Scale factor for ellipses
          const xRange = xExtent[1] - xExtent[0]
          const yRange = yExtent[1] - yExtent[0]
          const sxFactor = innerWidth / xRange
          const syFactor = innerHeight / yRange

          return (
            <>
              {/* Gaussian ellipses */}
              {snap.means.map((mean, ci) => {
                const cov = snap.covariances[ci]
                if (!cov) return null
                const eigen = eigen2x2Symmetric(cov)
                const rx1 = Math.sqrt(eigen.values[0]) * sxFactor
                const ry1 = Math.sqrt(eigen.values[1]) * syFactor
                const rx2 = Math.sqrt(eigen.values[0]) * 2 * sxFactor
                const ry2 = Math.sqrt(eigen.values[1]) * 2 * syFactor
                const color = COLORS.clusters[ci % COLORS.clusters.length]

                return (
                  <g key={`ellipse-${ci}`}>
                    {/* 2-sigma ellipse */}
                    <motion.ellipse
                      cx={xScale(mean.x)}
                      cy={yScale(mean.y)}
                      rx={rx2}
                      ry={ry2}
                      fill={color}
                      fillOpacity={0.03}
                      stroke={color}
                      strokeWidth={1}
                      strokeOpacity={0.15}
                      transform={`rotate(${-eigen.angle}, ${xScale(mean.x)}, ${yScale(mean.y)})`}
                      animate={{
                        cx: xScale(mean.x),
                        cy: yScale(mean.y),
                        rx: rx2,
                        ry: ry2,
                      }}
                      transition={{ duration: 0.5 }}
                    />
                    {/* 1-sigma ellipse */}
                    <motion.ellipse
                      cx={xScale(mean.x)}
                      cy={yScale(mean.y)}
                      rx={rx1}
                      ry={ry1}
                      fill={color}
                      fillOpacity={0.06}
                      stroke={color}
                      strokeWidth={1.5}
                      strokeOpacity={0.3}
                      transform={`rotate(${-eigen.angle}, ${xScale(mean.x)}, ${yScale(mean.y)})`}
                      animate={{
                        cx: xScale(mean.x),
                        cy: yScale(mean.y),
                        rx: rx1,
                        ry: ry1,
                      }}
                      transition={{ duration: 0.5 }}
                    />
                  </g>
                )
              })}

              {/* Data points */}
              {data.map((point, i) => {
                const responsibilities = snap.responsibilities[i] ?? []
                let color: string

                if (softAssignment && responsibilities.length > 0) {
                  const colors = responsibilities.map(
                    (_, ci) => COLORS.clusters[ci % COLORS.clusters.length]
                  )
                  color = blendColors(colors, responsibilities)
                } else {
                  // Hard assignment: argmax
                  let maxIdx = 0
                  let maxVal = 0
                  responsibilities.forEach((r, j) => {
                    if (r > maxVal) { maxVal = r; maxIdx = j }
                  })
                  color = COLORS.clusters[maxIdx % COLORS.clusters.length]
                }

                return (
                  <circle
                    key={i}
                    cx={xScale(point.x)}
                    cy={yScale(point.y)}
                    r={3.5}
                    fill={color}
                    fillOpacity={0.7}
                  />
                )
              })}

              {/* Component centers */}
              {snap.means.map((mean, ci) => (
                <motion.circle
                  key={`mean-${ci}`}
                  cx={xScale(mean.x)}
                  cy={yScale(mean.y)}
                  r={6}
                  fill={COLORS.clusters[ci % COLORS.clusters.length]}
                  stroke="#0F0F11"
                  strokeWidth={2}
                  animate={{ cx: xScale(mean.x), cy: yScale(mean.y) }}
                  transition={{ duration: 0.5 }}
                />
              ))}
            </>
          )
        }}
      </SVGContainer>

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
          values={logLikelihoods}
          currentIndex={player.currentStep}
          label="Log-likelihood"
          width={200}
          height={80}
          color="#818CF8"
        />
        <div className="text-xs text-text-tertiary self-center space-y-1">
          {snap.weights.map((w, i) => (
            <div key={i} className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.clusters[i % COLORS.clusters.length] }} />
              <span className="font-mono tabular-nums">π={w.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </GlassCard>
  )
}
