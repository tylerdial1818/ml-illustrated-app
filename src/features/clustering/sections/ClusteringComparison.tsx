import { useMemo, useState, useRef } from 'react'
import * as d3 from 'd3'
import { motion, useInView } from 'framer-motion'
import { makeBlobs, makeMoons, makeCircles, makeVaryingDensity } from '../../../lib/data/clusterGenerators'
import { runKMeans } from '../../../lib/algorithms/clustering/kmeans'
import { runDBSCAN } from '../../../lib/algorithms/clustering/dbscan'
import { runAgglomerative } from '../../../lib/algorithms/clustering/agglomerative'
import { runGMM } from '../../../lib/algorithms/clustering/gmm'
import { Select } from '../../../components/ui/Select'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS, type Point2D } from '../../../types'

const DATASET_OPTIONS = [
  { value: 'blobs', label: 'Well-separated blobs' },
  { value: 'moons', label: 'Moons' },
  { value: 'circles', label: 'Concentric rings' },
  { value: 'varying', label: 'Varying density' },
]

function getDataset(name: string): Point2D[] {
  switch (name) {
    case 'moons': return makeMoons(200, 0.08, 42)
    case 'circles': return makeCircles(200, 0.05, 42)
    case 'varying': return makeVaryingDensity(200, 42)
    default: return makeBlobs(200, 3, 1.0, 42)
  }
}

function MiniScatter({
  data,
  assignments,
  title,
  width = 250,
  height = 200,
}: {
  data: Point2D[]
  assignments: number[]
  title: string
  width?: number
  height?: number
}) {
  const padding = 15
  const xExtent = d3.extent(data, (d) => d.x) as [number, number]
  const yExtent = d3.extent(data, (d) => d.y) as [number, number]
  const xScale = d3.scaleLinear().domain(xExtent).range([padding, width - padding])
  const yScale = d3.scaleLinear().domain(yExtent).range([height - padding, padding])

  return (
    <div>
      <p className="text-xs font-medium text-text-secondary mb-2 text-center">{title}</p>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        {data.map((point, i) => {
          const a = assignments[i]
          const color = a < 0 ? COLORS.noise : COLORS.clusters[a % COLORS.clusters.length]
          return (
            <circle
              key={i}
              cx={xScale(point.x)}
              cy={yScale(point.y)}
              r={2.5}
              fill={color}
              fillOpacity={a < 0 ? 0.3 : 0.7}
            />
          )
        })}
      </svg>
    </div>
  )
}

export function ClusteringComparison() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })
  const [dataset, setDataset] = useState('blobs')

  const data = useMemo(() => getDataset(dataset), [dataset])

  const results = useMemo(() => {
    const kmeansSnaps = runKMeans(data, 3, 42, 30)
    const kmeansAssignments = kmeansSnaps[kmeansSnaps.length - 1].assignments

    const dbscanSnaps = runDBSCAN(data, 1.2, 4)
    const dbscanAssignments = dbscanSnaps[dbscanSnaps.length - 1].clusterAssignments

    const aggSnaps = runAgglomerative(data, 'ward')
    const aggAssignments = aggSnaps[Math.max(0, aggSnaps.length - data.length + 3)].assignments

    const gmmSnaps = runGMM(data, 3, 42, 30)
    const gmmResp = gmmSnaps[gmmSnaps.length - 1].responsibilities
    const gmmAssignments = gmmResp.map((r) => {
      let maxIdx = 0; let maxVal = 0
      r.forEach((v, j) => { if (v > maxVal) { maxVal = v; maxIdx = j } })
      return maxIdx
    })

    return { kmeansAssignments, dbscanAssignments, aggAssignments, gmmAssignments }
  }, [data])

  return (
    <section id="clustering-comparison" ref={ref} className="py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-3xl font-semibold text-text-primary">Comparison</h2>
        <p className="mt-2 text-lg text-text-secondary max-w-2xl">
          The same dataset, four different algorithms. See how each one handles the data differently.
        </p>

        <GlassCard className="mt-8 p-8">
          <div className="flex items-center justify-between mb-6">
            <Select
              label="Dataset"
              value={dataset}
              options={DATASET_OPTIONS}
              onChange={setDataset}
              className="w-56"
            />
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MiniScatter data={data} assignments={results.kmeansAssignments} title="K-Means" />
            <MiniScatter data={data} assignments={results.dbscanAssignments} title="DBSCAN" />
            <MiniScatter data={data} assignments={results.aggAssignments} title="Hierarchical" />
            <MiniScatter data={data} assignments={results.gmmAssignments} title="GMM" />
          </div>
        </GlassCard>

        {/* Selection guide */}
        <GlassCard className="mt-6 p-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Which algorithm should you use?</h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-1 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Do you know how many clusters?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> K-Means or GMM. <strong>No:</strong> DBSCAN or Hierarchical.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-2 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Are clusters roughly spherical?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> K-Means. <strong>No:</strong> DBSCAN (arbitrary shapes) or GMM (elliptical).
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-3 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Is noise present?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> DBSCAN (explicitly labels noise). Others will force noisy points into clusters.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-4 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Do you need probabilistic assignments?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> GMM. It gives you a probability per cluster, not a hard boundary.
                </p>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </section>
  )
}
