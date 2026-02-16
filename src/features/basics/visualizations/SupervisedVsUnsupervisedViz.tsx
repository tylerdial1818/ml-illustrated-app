import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Toggle } from '../../../components/ui/Toggle'
import { Select } from '../../../components/ui/Select'

// ── Colors ────────────────────────────────────────────────────────────
const CLASS_COLORS = ['#6366F1', '#F472B6', '#34D399']
const NEUTRAL = '#A1A1AA'
const BOUNDARY_COLOR = 'rgba(255, 255, 255, 0.08)'

// ── Seeded random for reproducibility ─────────────────────────────────
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

// ── Data generators ───────────────────────────────────────────────────
interface DataPoint {
  x: number
  y: number
  label: number
}

function generateClusters(seed: number): DataPoint[] {
  const rng = seededRandom(seed)
  const centers = [
    { cx: -2.5, cy: 2.5 },
    { cx: 2.5, cy: 2.5 },
    { cx: 0, cy: -2.5 },
  ]
  const points: DataPoint[] = []

  centers.forEach((center, label) => {
    for (let i = 0; i < 27; i++) {
      const angle = rng() * Math.PI * 2
      const r = (rng() + rng()) * 0.8
      points.push({
        x: center.cx + Math.cos(angle) * r,
        y: center.cy + Math.sin(angle) * r,
        label,
      })
    }
  })
  return points
}

function generateCrescents(seed: number): DataPoint[] {
  const rng = seededRandom(seed)
  const points: DataPoint[] = []

  // Upper crescent (label 0)
  for (let i = 0; i < 40; i++) {
    const angle = Math.PI * rng()
    const r = 2 + (rng() - 0.5) * 0.6
    points.push({
      x: Math.cos(angle) * r,
      y: Math.sin(angle) * r,
      label: 0,
    })
  }

  // Lower crescent (label 1)
  for (let i = 0; i < 40; i++) {
    const angle = Math.PI + Math.PI * rng()
    const r = 2 + (rng() - 0.5) * 0.6
    points.push({
      x: Math.cos(angle) * r + 1,
      y: Math.sin(angle) * r + 0.5,
      label: 1,
    })
  }
  return points
}

function generateRings(seed: number): DataPoint[] {
  const rng = seededRandom(seed)
  const points: DataPoint[] = []

  // Inner ring (label 0)
  for (let i = 0; i < 30; i++) {
    const angle = rng() * Math.PI * 2
    const r = 0.8 + (rng() - 0.5) * 0.4
    points.push({
      x: Math.cos(angle) * r,
      y: Math.sin(angle) * r,
      label: 0,
    })
  }

  // Outer ring (label 1)
  for (let i = 0; i < 50; i++) {
    const angle = rng() * Math.PI * 2
    const r = 2.5 + (rng() - 0.5) * 0.5
    points.push({
      x: Math.cos(angle) * r,
      y: Math.sin(angle) * r,
      label: 1,
    })
  }
  return points
}

// ── Dataset options ───────────────────────────────────────────────────
const DATASET_OPTIONS = [
  { value: 'clusters', label: '3 Clusters' },
  { value: 'crescents', label: '2 Crescents' },
  { value: 'rings', label: 'Concentric Rings' },
]

const GENERATORS: Record<string, (seed: number) => DataPoint[]> = {
  clusters: generateClusters,
  crescents: generateCrescents,
  rings: generateRings,
}

// ── Scatter Plot (SVG) ────────────────────────────────────────────────
function ScatterPanel({
  data,
  showLabels,
  innerWidth,
  innerHeight,
  title,
  annotation,
}: {
  data: DataPoint[]
  showLabels: boolean
  innerWidth: number
  innerHeight: number
  title: string
  annotation: string
}) {
  const xExtent = d3.extent(data, (d) => d.x) as [number, number]
  const yExtent = d3.extent(data, (d) => d.y) as [number, number]
  const pad = 0.5
  const xScale = d3
    .scaleLinear()
    .domain([xExtent[0] - pad, xExtent[1] + pad])
    .range([0, innerWidth])
  const yScale = d3
    .scaleLinear()
    .domain([yExtent[0] - pad, yExtent[1] + pad])
    .range([innerHeight, 0])

  // Distinct labels for cluster circles in unsupervised mode
  const uniqueLabels = [...new Set(data.map((d) => d.label))].sort()

  // Cluster centroids for unsupervised mode circles
  const clusterInfo = useMemo(() => {
    return uniqueLabels.map((label) => {
      const pts = data.filter((d) => d.label === label)
      const cx = d3.mean(pts, (p) => p.x) ?? 0
      const cy = d3.mean(pts, (p) => p.y) ?? 0
      const maxDist = d3.max(pts, (p) => Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)) ?? 1
      return { label, cx, cy, radius: maxDist + 0.3 }
    })
  }, [data, uniqueLabels])

  return (
    <>
      {/* Title */}
      <text
        x={innerWidth / 2}
        y={-6}
        textAnchor="middle"
        className="text-[11px] font-semibold"
        fill="#E4E4E7"
      >
        {title}
      </text>

      {/* Cluster boundary circles (unsupervised mode) */}
      {!showLabels &&
        clusterInfo.map((c) => (
          <motion.ellipse
            key={`cluster-${c.label}`}
            cx={xScale(c.cx)}
            cy={yScale(c.cy)}
            rx={Math.abs(xScale(c.cx + c.radius) - xScale(c.cx))}
            ry={Math.abs(yScale(c.cy + c.radius) - yScale(c.cy))}
            fill="none"
            stroke={BOUNDARY_COLOR}
            strokeWidth={1.5}
            strokeDasharray="4 3"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: c.label * 0.1 }}
          />
        ))}

      {/* Data points */}
      {data.map((point, i) => (
        <motion.circle
          key={i}
          cx={xScale(point.x)}
          cy={yScale(point.y)}
          r={4}
          fill={showLabels ? CLASS_COLORS[point.label % CLASS_COLORS.length] : NEUTRAL}
          fillOpacity={0.8}
          animate={{
            fill: showLabels ? CLASS_COLORS[point.label % CLASS_COLORS.length] : NEUTRAL,
          }}
          transition={{ duration: 0.4, delay: i * 0.003 }}
        />
      ))}

      {/* Annotation */}
      <foreignObject x={0} y={innerHeight + 8} width={innerWidth} height={40}>
        <p className="text-[10px] text-text-tertiary text-center leading-relaxed px-2">
          {annotation}
        </p>
      </foreignObject>
    </>
  )
}

// ── Supervised examples badges ────────────────────────────────────────
function TaskExamples({ tasks }: { tasks: { label: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap gap-1.5 justify-center mt-2">
      {tasks.map((task) => (
        <span
          key={task.label}
          className="px-2 py-0.5 rounded-full text-[9px] font-mono border"
          style={{
            backgroundColor: `${task.color}10`,
            borderColor: `${task.color}30`,
            color: `${task.color}CC`,
          }}
        >
          {task.label}
        </span>
      ))}
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function SupervisedVsUnsupervisedViz() {
  const [showLabels, setShowLabels] = useState(true)
  const [dataset, setDataset] = useState('clusters')

  const data = useMemo(() => {
    const gen = GENERATORS[dataset] ?? generateClusters
    return gen(42)
  }, [dataset])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-5">
          <Toggle
            label="Show Labels"
            checked={showLabels}
            onChange={setShowLabels}
          />

          <div className="h-6 w-px bg-obsidian-border" />

          <Select
            label="Dataset"
            value={dataset}
            options={DATASET_OPTIONS}
            onChange={setDataset}
            className="w-44"
          />
        </div>
      </GlassCard>

      {/* Dual panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Supervised panel */}
        <GlassCard className="overflow-hidden">
          <SVGContainer
            aspectRatio={4 / 3}
            minHeight={260}
            maxHeight={380}
            padding={{ top: 24, right: 20, bottom: 50, left: 20 }}
          >
            {({ innerWidth, innerHeight }) => (
              <ScatterPanel
                data={data}
                showLabels={true}
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                title="Supervised Learning"
                annotation="Every point has a label. The model learns the mapping from features to labels."
              />
            )}
          </SVGContainer>
          <div className="px-4 pb-3">
            <p className="text-[10px] text-text-secondary text-center leading-relaxed">
              You give the model both the data AND the correct answers.
            </p>
            <TaskExamples
              tasks={[
                { label: 'Classification', color: '#6366F1' },
                { label: 'Regression', color: '#F472B6' },
                { label: 'Image recognition', color: '#34D399' },
              ]}
            />
          </div>
        </GlassCard>

        {/* Unsupervised panel */}
        <GlassCard className="overflow-hidden">
          <SVGContainer
            aspectRatio={4 / 3}
            minHeight={260}
            maxHeight={380}
            padding={{ top: 24, right: 20, bottom: 50, left: 20 }}
          >
            {({ innerWidth, innerHeight }) => (
              <ScatterPanel
                data={data}
                showLabels={false}
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                title="Unsupervised Learning"
                annotation="No labels. The model discovers that these points naturally group into clusters."
              />
            )}
          </SVGContainer>
          <div className="px-4 pb-3">
            <p className="text-[10px] text-text-secondary text-center leading-relaxed">
              You give the model just the data. No answers. It finds structure on its own.
            </p>
            <TaskExamples
              tasks={[
                { label: 'Clustering', color: '#FBBF24' },
                { label: 'Dimensionality reduction', color: '#38BDF8' },
                { label: 'Anomaly detection', color: '#E879F9' },
              ]}
            />
          </div>
        </GlassCard>
      </div>

      {/* Key insight */}
      <GlassCard className="p-4">
        <p className="text-sm text-center text-text-secondary leading-relaxed">
          Toggle &quot;Show Labels&quot; to see the same data tell two different stories.
          With labels, it is a supervised problem. Without, it is unsupervised.
          The data is the same. The question you ask changes everything.
        </p>
      </GlassCard>
    </div>
  )
}
