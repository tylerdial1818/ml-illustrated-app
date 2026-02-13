import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { makeBlobs, addNoise } from '../../../lib/data/clusterGenerators'
import { runDBSCAN } from '../../../lib/algorithms/clustering/dbscan'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function DBSCANViz() {
  const [epsilon, setEpsilon] = useState(1.5)
  const [minPts, setMinPts] = useState(4)
  const [hoverPoint, setHoverPoint] = useState<number | null>(null)

  const data = useMemo(() => addNoise(makeBlobs(150, 3, 1.0, 42), 0.15, 99), [])

  const snapshots = useMemo(
    () => runDBSCAN(data, epsilon, minPts),
    [data, epsilon, minPts]
  )

  const player = useAlgorithmPlayer({ snapshots, baseFps: 8 })
  const snap = player.currentSnapshot

  const handleReset = useCallback(() => {
    player.reset()
  }, [player])

  return (
    <GlassCard className="p-8">
      {/* Controls */}
      <div className="flex flex-wrap gap-6 mb-6">
        <Slider
          label="ε (radius)"
          value={epsilon}
          min={0.5}
          max={3.0}
          step={0.1}
          onChange={(v) => { setEpsilon(v); player.reset() }}
          formatValue={(v) => v.toFixed(1)}
          className="w-48"
        />
        <Slider
          label="minPts"
          value={minPts}
          min={2}
          max={10}
          step={1}
          onChange={(v) => { setMinPts(v); player.reset() }}
          className="w-48"
        />
      </div>

      {/* Visualization */}
      <SVGContainer aspectRatio={16 / 10} minHeight={350} maxHeight={550}>
        {({ innerWidth, innerHeight }) => {
          const xExtent = d3.extent(data, (d) => d.x) as [number, number]
          const yExtent = d3.extent(data, (d) => d.y) as [number, number]
          const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
          const yScale = d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice()

          // Convert epsilon to pixel radius (approximate)
          const epsilonPx = Math.abs(xScale(epsilon) - xScale(0))

          return (
            <>
              {/* Epsilon neighborhood highlight */}
              {snap.neighborhoodHighlight && (
                <circle
                  cx={xScale(data[snap.neighborhoodHighlight.center].x)}
                  cy={yScale(data[snap.neighborhoodHighlight.center].y)}
                  r={epsilonPx}
                  fill={COLORS.accent}
                  fillOpacity={0.06}
                  stroke={COLORS.accent}
                  strokeOpacity={0.3}
                  strokeWidth={1}
                  strokeDasharray="4,4"
                />
              )}

              {/* Hover epsilon circle */}
              {hoverPoint !== null && (
                <circle
                  cx={xScale(data[hoverPoint].x)}
                  cy={yScale(data[hoverPoint].y)}
                  r={epsilonPx}
                  fill="none"
                  stroke={COLORS.accent}
                  strokeOpacity={0.2}
                  strokeWidth={1}
                  strokeDasharray="3,3"
                />
              )}

              {/* Data points */}
              {data.map((point, i) => {
                const classification = snap.classifications[i]
                const clusterId = snap.clusterAssignments[i]
                let color = '#3F3F46' // unvisited
                let radius = 3.5
                let opacity = 0.5

                if (classification === 'core') {
                  color = clusterId >= 0 ? COLORS.clusters[clusterId % COLORS.clusters.length] : '#A1A1AA'
                  radius = 4.5
                  opacity = 0.9
                } else if (classification === 'border') {
                  color = clusterId >= 0 ? COLORS.clusters[clusterId % COLORS.clusters.length] : '#A1A1AA'
                  radius = 3.5
                  opacity = 0.6
                } else if (classification === 'noise') {
                  color = COLORS.noise
                  radius = 3
                  opacity = 0.4
                }

                const isCurrent = snap.currentPointIndex === i

                return (
                  <motion.circle
                    key={i}
                    cx={xScale(point.x)}
                    cy={yScale(point.y)}
                    r={isCurrent ? 6 : radius}
                    fill={color}
                    fillOpacity={opacity}
                    stroke={isCurrent ? '#fff' : 'none'}
                    strokeWidth={isCurrent ? 2 : 0}
                    animate={{ fill: color, r: isCurrent ? 6 : radius }}
                    transition={{ duration: 0.2 }}
                    onMouseEnter={() => setHoverPoint(i)}
                    onMouseLeave={() => setHoverPoint(null)}
                    className="cursor-pointer"
                  />
                )
              })}
            </>
          )
        }}
      </SVGContainer>

      {/* Legend + Transport */}
      <div className="mt-4 flex flex-wrap items-center gap-4">
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
        <div className="flex items-center gap-4 text-xs text-text-tertiary">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-cluster-1 inline-block" /> Core
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-cluster-1 opacity-50 inline-block" /> Border
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-zinc-600 inline-block" /> Noise
          </span>
        </div>
      </div>
    </GlassCard>
  )
}
