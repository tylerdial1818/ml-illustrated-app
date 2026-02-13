import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { makeBlobs } from '../../../lib/data/clusterGenerators'
import { runKMeans } from '../../../lib/algorithms/clustering/kmeans'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { ConvergenceChart } from '../../../components/viz/ConvergenceChart'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function KMeansViz() {
  const [k, setK] = useState(3)
  const [seed, setSeed] = useState(42)

  const data = useMemo(() => makeBlobs(200, 4, 1.2, 42), [])

  const snapshots = useMemo(() => runKMeans(data, k, seed, 30), [data, k, seed])

  const player = useAlgorithmPlayer({ snapshots, baseFps: 2 })
  const snap = player.currentSnapshot

  const handleReset = useCallback(() => {
    setSeed((s) => s + 1)
    player.reset()
  }, [player])

  const costs = useMemo(() => snapshots.map((s) => s.cost), [snapshots])

  return (
    <GlassCard className="p-6">
      {/* Controls */}
      <div className="flex flex-wrap gap-6 mb-6">
        <Slider
          label="k (clusters)"
          value={k}
          min={2}
          max={8}
          step={1}
          onChange={(v) => { setK(v); player.reset() }}
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

          // Voronoi boundaries
          const voronoi = snap.centroids.length > 0
            ? d3.Delaunay.from(snap.centroids, (c) => xScale(c.x), (c) => yScale(c.y))
                .voronoi([0, 0, innerWidth, innerHeight])
            : null

          return (
            <>
              {/* Voronoi regions */}
              {voronoi && snap.centroids.map((_, i) => (
                <path
                  key={`voronoi-${i}`}
                  d={voronoi.renderCell(i)}
                  fill={COLORS.clusters[i % COLORS.clusters.length]}
                  fillOpacity={0.04}
                  stroke={COLORS.clusters[i % COLORS.clusters.length]}
                  strokeOpacity={0.15}
                  strokeWidth={1}
                />
              ))}

              {/* Data points */}
              {data.map((point, i) => (
                <motion.circle
                  key={i}
                  cx={xScale(point.x)}
                  cy={yScale(point.y)}
                  r={3.5}
                  fill={
                    snap.assignments[i] >= 0
                      ? COLORS.clusters[snap.assignments[i] % COLORS.clusters.length]
                      : '#52525B'
                  }
                  fillOpacity={0.7}
                  animate={{
                    fill: snap.assignments[i] >= 0
                      ? COLORS.clusters[snap.assignments[i] % COLORS.clusters.length]
                      : '#52525B',
                  }}
                  transition={{ duration: 0.3 }}
                />
              ))}

              {/* Centroid trails */}
              {snap.centroidHistory.map((trail, ci) => {
                if (trail.length < 2) return null
                const pathData = trail
                  .map((p, j) => `${j === 0 ? 'M' : 'L'}${xScale(p.x)},${yScale(p.y)}`)
                  .join(' ')
                return (
                  <path
                    key={`trail-${ci}`}
                    d={pathData}
                    fill="none"
                    stroke={COLORS.clusters[ci % COLORS.clusters.length]}
                    strokeWidth={1.5}
                    strokeOpacity={0.4}
                    strokeDasharray="4,4"
                  />
                )
              })}

              {/* Centroids */}
              {snap.centroids.map((c, i) => (
                <motion.g key={`centroid-${i}`}>
                  <motion.circle
                    cx={xScale(c.x)}
                    cy={yScale(c.y)}
                    r={8}
                    fill={COLORS.clusters[i % COLORS.clusters.length]}
                    stroke="#0F0F11"
                    strokeWidth={2}
                    animate={{ cx: xScale(c.x), cy: yScale(c.y) }}
                    transition={{ duration: 0.4, ease: 'easeInOut' }}
                  />
                  <motion.circle
                    cx={xScale(c.x)}
                    cy={yScale(c.y)}
                    r={12}
                    fill="none"
                    stroke={COLORS.clusters[i % COLORS.clusters.length]}
                    strokeWidth={1}
                    strokeOpacity={0.3}
                    animate={{ cx: xScale(c.x), cy: yScale(c.y) }}
                    transition={{ duration: 0.4, ease: 'easeInOut' }}
                  />
                </motion.g>
              ))}
            </>
          )
        }}
      </SVGContainer>

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
          values={costs}
          currentIndex={player.currentStep}
          label="Within-cluster SS"
          width={200}
          height={80}
        />
        <div className="text-xs text-text-tertiary self-center">
          {snap.converged && (
            <span className="text-success">Converged at iteration {snap.iteration}</span>
          )}
        </div>
      </div>
    </GlassCard>
  )
}
