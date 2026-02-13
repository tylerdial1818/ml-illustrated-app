import { useMemo, useState } from 'react'
import { motion, useInView } from 'framer-motion'
import { useRef } from 'react'
import * as d3 from 'd3'
import { makeBlobs } from '../../../lib/data/clusterGenerators'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function ClusteringOverview() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.2, once: true })
  const [showColors, setShowColors] = useState(false)

  const data = useMemo(() => makeBlobs(150, 4, 1.2, 42), [])

  const xExtent = d3.extent(data, (d) => d.x) as [number, number]
  const yExtent = d3.extent(data, (d) => d.y) as [number, number]

  const width = 600
  const height = 400
  const padding = { top: 20, right: 20, bottom: 20, left: 20 }
  const innerW = width - padding.left - padding.right
  const innerH = height - padding.top - padding.bottom

  const xScale = d3.scaleLinear().domain(xExtent).range([0, innerW]).nice()
  const yScale = d3.scaleLinear().domain(yExtent).range([innerH, 0]).nice()

  // Simple k-means for coloring (pre-computed)
  const clusterColors = useMemo(() => {
    // Quick centroid assignment for visualization
    const k = 4
    const centroids = [
      { x: -3, y: 3 }, { x: 3, y: 3 }, { x: -3, y: -3 }, { x: 3, y: -3 },
    ]
    return data.map((p) => {
      let minDist = Infinity
      let minIdx = 0
      centroids.forEach((c, i) => {
        const dist = (p.x - c.x) ** 2 + (p.y - c.y) ** 2
        if (dist < minDist) { minDist = dist; minIdx = i }
      })
      return minIdx
    })
  }, [data])

  return (
    <section id="clustering-overview" ref={ref} className="py-16 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h4 className="text-xs font-semibold uppercase tracking-[0.1em] text-text-tertiary mb-6">
          The Problem
        </h4>
        <h2 className="text-2xl font-semibold text-text-primary mb-4">
          What is Clustering?
        </h2>
        <p className="text-text-secondary max-w-2xl leading-relaxed">
          Clustering is about finding hidden groups in data — without anyone telling you what the
          groups are. You have a pile of unlabeled data points, and the algorithm's job is to figure
          out which ones belong together.
        </p>

        {/* Interactive scatter visualization */}
        <GlassCard className="mt-8 p-6">
          <div className="flex flex-col items-center">
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="max-w-full h-auto">
              <g transform={`translate(${padding.left}, ${padding.top})`}>
                {data.map((point, i) => (
                  <motion.circle
                    key={i}
                    cx={xScale(point.x)}
                    cy={yScale(point.y)}
                    r={4}
                    initial={{ opacity: 0, scale: 0 }}
                    animate={isInView ? {
                      opacity: 0.8,
                      scale: 1,
                      fill: showColors ? COLORS.clusters[clusterColors[i]] : '#A1A1AA',
                    } : {}}
                    transition={{
                      delay: i * 0.005,
                      duration: 0.3,
                      fill: { duration: 0.5 },
                    }}
                  />
                ))}
              </g>
            </svg>

            <button
              onClick={() => setShowColors(!showColors)}
              className="mt-4 px-4 py-2 text-sm bg-obsidian-hover border border-obsidian-border rounded-lg text-text-secondary hover:text-text-primary transition-colors"
            >
              {showColors ? 'Hide clusters' : 'Reveal clusters'}
            </button>
          </div>
        </GlassCard>

        {/* Taxonomy */}
        <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[
            { name: 'Partitioning', desc: 'Divide data into k groups (K-Means)', color: COLORS.clusters[0] },
            { name: 'Density-based', desc: 'Find dense regions, label sparse as noise (DBSCAN)', color: COLORS.clusters[1] },
            { name: 'Hierarchical', desc: 'Build a tree of nested clusters (Agglomerative)', color: COLORS.clusters[2] },
            { name: 'Probabilistic', desc: 'Model data as mixture of distributions (GMM)', color: COLORS.clusters[3] },
          ].map((item) => (
            <GlassCard key={item.name} className="p-4">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-sm font-medium text-text-primary">{item.name}</span>
              </div>
              <p className="text-xs text-text-secondary">{item.desc}</p>
            </GlassCard>
          ))}
        </div>
      </motion.div>
    </section>
  )
}
