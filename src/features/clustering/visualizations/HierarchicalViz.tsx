import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { makeBlobs } from '../../../lib/data/clusterGenerators'
import { runAgglomerative, type LinkageMethod } from '../../../lib/algorithms/clustering/agglomerative'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { Select } from '../../../components/ui/Select'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

const LINKAGE_OPTIONS = [
  { value: 'ward', label: "Ward's" },
  { value: 'single', label: 'Single' },
  { value: 'complete', label: 'Complete' },
  { value: 'average', label: 'Average' },
]

export function HierarchicalViz() {
  const [linkage, setLinkage] = useState<LinkageMethod>('ward')
  const [cutHeight, setCutHeight] = useState(0.5)

  const data = useMemo(() => makeBlobs(60, 3, 1.0, 42), [])

  const snapshots = useMemo(
    () => runAgglomerative(data, linkage),
    [data, linkage]
  )

  const player = useAlgorithmPlayer({ snapshots, baseFps: 3 })
  const snap = player.currentSnapshot

  // Determine clusters from cut height
  const maxDist = useMemo(() => {
    const distances = snapshots.flatMap((s) => s.distances)
    return Math.max(...distances, 1)
  }, [snapshots])

  const actualCutHeight = cutHeight * maxDist

  // Get the snapshot closest to the cut height
  const cutAssignments = useMemo(() => {
    // Find the state where clusters are at the cut level
    for (let i = snapshots.length - 1; i >= 0; i--) {
      const lastMergeDist = snapshots[i].distances[snapshots[i].distances.length - 1] ?? 0
      if (lastMergeDist <= actualCutHeight) {
        return snapshots[i].assignments
      }
    }
    return snapshots[0].assignments
  }, [snapshots, actualCutHeight])

  // Unique clusters for coloring
  const uniqueClusters = useMemo(() => [...new Set(cutAssignments)], [cutAssignments])
  const clusterColorMap = useMemo(() => {
    const map = new Map<number, string>()
    uniqueClusters.forEach((id, i) => {
      map.set(id, COLORS.clusters[i % COLORS.clusters.length])
    })
    return map
  }, [uniqueClusters])

  const handleReset = useCallback(() => {
    player.reset()
  }, [player])

  return (
    <GlassCard className="p-8">
      <div className="flex flex-wrap gap-6 mb-6">
        <Select
          label="Linkage"
          value={linkage}
          options={LINKAGE_OPTIONS}
          onChange={(v) => { setLinkage(v as LinkageMethod); player.reset() }}
          className="w-40"
        />
        <Slider
          label="Cut height"
          value={cutHeight}
          min={0.05}
          max={1.0}
          step={0.05}
          onChange={setCutHeight}
          formatValue={(v) => `${(v * 100).toFixed(0)}%`}
          className="w-48"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Scatter plot */}
        <div>
          <p className="text-xs text-text-tertiary uppercase tracking-wider mb-2">Scatter Plot</p>
          <SVGContainer aspectRatio={1} minHeight={280} maxHeight={400}>
            {({ innerWidth, innerHeight }) => {
              const xExtent = d3.extent(data, (d) => d.x) as [number, number]
              const yExtent = d3.extent(data, (d) => d.y) as [number, number]
              const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
              const yScale = d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice()

              return (
                <>
                  {data.map((point, i) => (
                    <motion.circle
                      key={i}
                      cx={xScale(point.x)}
                      cy={yScale(point.y)}
                      r={5}
                      fill={clusterColorMap.get(cutAssignments[i]) ?? '#A1A1AA'}
                      fillOpacity={0.8}
                      stroke="#0F0F11"
                      strokeWidth={1}
                      animate={{
                        fill: clusterColorMap.get(cutAssignments[i]) ?? '#A1A1AA',
                      }}
                      transition={{ duration: 0.3 }}
                    />
                  ))}
                </>
              )
            }}
          </SVGContainer>
        </div>

        {/* Dendrogram */}
        <div>
          <p className="text-xs text-text-tertiary uppercase tracking-wider mb-2">Dendrogram</p>
          <SVGContainer aspectRatio={1} minHeight={280} maxHeight={400} padding={{ top: 20, right: 20, bottom: 30, left: 40 }}>
            {({ innerWidth, innerHeight }) => {
              const mergeHistory = snap.mergeHistory
              if (mergeHistory.length === 0) return null

              const n = data.length
              // Build dendrogram layout
              const yScale = d3.scaleLinear().domain([0, maxDist]).range([innerHeight, 0])
              const xScale = d3.scaleLinear().domain([0, n - 1]).range([10, innerWidth - 10])

              // Track position of each cluster (leaf or merged)
              const positions = new Map<number, number>()
              // Initial leaf positions
              for (let i = 0; i < n; i++) {
                positions.set(i, xScale(i))
              }

              const links: { x1: number; y1: number; x2: number; y2: number; above: boolean }[] = []

              for (const merge of mergeHistory) {
                const x1 = positions.get(merge.cluster1) ?? 0
                const x2 = positions.get(merge.cluster2) ?? 0
                const midX = (x1 + x2) / 2
                const mergeY = yScale(merge.distance)

                const aboveCut = merge.distance > actualCutHeight

                // Horizontal lines
                links.push({ x1, y1: mergeY, x2, y2: mergeY, above: aboveCut })
                // Vertical lines
                const prevDist1 = mergeHistory.find(
                  (m) => m.newClusterId === merge.cluster1
                )?.distance ?? 0
                const prevDist2 = mergeHistory.find(
                  (m) => m.newClusterId === merge.cluster2
                )?.distance ?? 0
                links.push({ x1, y1: yScale(prevDist1), x2: x1, y2: mergeY, above: aboveCut })
                links.push({ x1: x2, y1: yScale(prevDist2), x2, y2: mergeY, above: aboveCut })

                positions.set(merge.newClusterId, midX)
              }

              return (
                <>
                  {/* Dendrogram links */}
                  {links.map((link, i) => (
                    <line
                      key={i}
                      x1={link.x1}
                      y1={link.y1}
                      x2={link.x2}
                      y2={link.y2}
                      stroke={link.above ? '#52525B' : COLORS.accent}
                      strokeWidth={1.5}
                      strokeOpacity={link.above ? 0.3 : 0.7}
                    />
                  ))}
                  {/* Cut line */}
                  <line
                    x1={0}
                    y1={yScale(actualCutHeight)}
                    x2={innerWidth}
                    y2={yScale(actualCutHeight)}
                    stroke={COLORS.error}
                    strokeWidth={1}
                    strokeDasharray="6,4"
                    strokeOpacity={0.6}
                  />
                  <text
                    x={innerWidth}
                    y={yScale(actualCutHeight) - 6}
                    textAnchor="end"
                    className="text-[10px] fill-error"
                  >
                    cut
                  </text>
                  {/* Y axis label */}
                  {yScale.ticks(4).map((tick) => (
                    <text
                      key={tick}
                      x={-6}
                      y={yScale(tick)}
                      textAnchor="end"
                      dominantBaseline="middle"
                      className="text-[9px] fill-text-tertiary"
                    >
                      {tick.toFixed(1)}
                    </text>
                  ))}
                </>
              )
            }}
          </SVGContainer>
        </div>
      </div>

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
        <span className="text-xs text-text-tertiary">
          {uniqueClusters.length} clusters at current cut
        </span>
      </div>
    </GlassCard>
  )
}
