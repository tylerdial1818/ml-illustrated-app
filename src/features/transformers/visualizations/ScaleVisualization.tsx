import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import modelScalesData from '../../../lib/data/transformers/model-scales.json'

// ── Model Data ────────────────────────────────────────────────────────
interface ModelInfo {
  name: string
  params: number
  layers: number
  dModel: number
  heads: number
  year: number
  color: string
}

// Color map for known models
const MODEL_COLORS: Record<string, string> = {
  'BERT-base': '#6366F1',
  'BERT-large': '#818CF8',
  'GPT-2': '#F472B6',
  'GPT-3': '#E879F9',
  'LLaMA 2 70B': '#34D399',
}

// Import from JSON, filtering out models with null values
const MODELS: ModelInfo[] = modelScalesData.models
  .filter(
    (m): m is { name: string; params: number; layers: number; dModel: number; heads: number; year: number } =>
      m.params != null && m.layers != null && m.dModel != null && m.heads != null
  )
  .map((m) => ({
    ...m,
    color: MODEL_COLORS[m.name] ?? '#FBBF24',
  }))

// Sort by year then by params for consistent positioning
const MODELS_SORTED = [...MODELS].sort((a, b) => a.year - b.year || a.params - b.params)

// ── Formatting Helpers ────────────────────────────────────────────────
function formatParams(n: number): string {
  if (n >= 1_000_000_000) {
    const b = n / 1_000_000_000
    return b % 1 === 0 ? `${b}B` : `${b.toFixed(1)}B`
  }
  const m = n / 1_000_000
  return m % 1 === 0 ? `${m}M` : `${m.toFixed(0)}M`
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

// ── Metric Types ──────────────────────────────────────────────────────
type MetricKey = 'params' | 'layers' | 'dModel' | 'heads'

const METRICS: { key: MetricKey; label: string; format: (v: number) => string }[] = [
  { key: 'params', label: 'Parameters', format: formatParams },
  { key: 'layers', label: 'Layers', format: (v) => String(v) },
  { key: 'dModel', label: 'd_model', format: formatNumber },
  { key: 'heads', label: 'Attention Heads', format: (v) => String(v) },
]

// ── Bubble Chart (SVG) ────────────────────────────────────────────────

function BubbleChart({
  innerWidth,
  innerHeight,
  useLogScale,
  selectedMetric,
}: {
  innerWidth: number
  innerHeight: number
  useLogScale: boolean
  selectedMetric: MetricKey
}) {
  // Determine the value accessor
  const getValue = (m: ModelInfo) => m[selectedMetric]

  // X scale by year
  const years = MODELS_SORTED.map((m) => m.year)
  const uniqueYears = [...new Set(years)].sort()
  const xScale = d3
    .scalePoint<number>()
    .domain(uniqueYears)
    .range([60, innerWidth - 60])
    .padding(0.5)

  // Radius scale: area proportional to value
  const values = MODELS.map(getValue)
  const maxVal = Math.max(...values)
  const minVal = Math.min(...values)
  const maxRadius = Math.min(innerWidth / 6, innerHeight / 3.5)
  const minRadius = 14

  const radiusScale = useMemo(() => {
    if (useLogScale) {
      return d3
        .scaleSqrt()
        .domain([Math.log10(Math.max(1, minVal)), Math.log10(maxVal)])
        .range([minRadius, maxRadius])
        .clamp(true)
    }
    return d3
      .scaleSqrt()
      .domain([0, maxVal])
      .range([minRadius, maxRadius])
      .clamp(true)
  }, [useLogScale, minVal, maxVal, maxRadius, minRadius])

  const getRadius = (model: ModelInfo) => {
    const v = getValue(model)
    if (useLogScale) {
      return radiusScale(Math.log10(Math.max(1, v)))
    }
    return radiusScale(v)
  }

  // Y positioning: center with stagger for same-year models
  const centerY = innerHeight / 2

  // Group models by year for stagger
  const yearGroups: Record<number, ModelInfo[]> = {}
  for (const m of MODELS_SORTED) {
    if (!yearGroups[m.year]) yearGroups[m.year] = []
    yearGroups[m.year].push(m)
  }

  // Calculate positions
  const positions: { model: ModelInfo; cx: number; cy: number; r: number }[] = []
  for (const year of uniqueYears) {
    const group = yearGroups[year]
    const baseX = xScale(year) ?? innerWidth / 2
    for (let i = 0; i < group.length; i++) {
      const model = group[i]
      const r = getRadius(model)
      const staggerY =
        group.length > 1
          ? centerY + (i - (group.length - 1) / 2) * (r * 1.3 + 10)
          : centerY
      positions.push({ model, cx: baseX, cy: staggerY, r })
    }
  }

  return (
    <>
      {/* Year axis */}
      <line
        x1={0}
        y1={innerHeight - 5}
        x2={innerWidth}
        y2={innerHeight - 5}
        stroke="rgba(255,255,255,0.08)"
        strokeWidth={1}
      />
      {uniqueYears.map((year) => (
        <text
          key={year}
          x={xScale(year) ?? 0}
          y={innerHeight + 10}
          textAnchor="middle"
          className="text-[10px] font-mono"
          fill="#71717A"
        >
          {year}
        </text>
      ))}

      {/* Bubbles */}
      {positions.map(({ model, cx, cy, r }) => {
        const metric = METRICS.find((m) => m.key === selectedMetric)
        const formattedValue = metric ? metric.format(getValue(model)) : String(getValue(model))

        return (
          <motion.g
            key={model.name}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.6, type: 'spring', damping: 12 }}
          >
            {/* Outer glow */}
            <motion.circle
              cx={cx}
              cy={cy}
              r={r}
              fill={`${model.color}12`}
              stroke={`${model.color}40`}
              strokeWidth={1.5}
              animate={{ r }}
              transition={{ duration: 0.5, type: 'spring' }}
            />
            {/* Inner circle */}
            <motion.circle
              cx={cx}
              cy={cy}
              r={Math.max(r * 0.6, 10)}
              fill={`${model.color}25`}
              animate={{ r: Math.max(r * 0.6, 10) }}
              transition={{ duration: 0.5, type: 'spring' }}
            />

            {/* Model name */}
            <text
              x={cx}
              y={cy - (r > 30 ? 8 : 0)}
              textAnchor="middle"
              dominantBaseline="central"
              className="text-[10px] font-mono font-bold"
              fill={model.color}
            >
              {model.name}
            </text>

            {/* Value label (only if bubble big enough) */}
            {r > 25 && (
              <text
                x={cx}
                y={cy + 10}
                textAnchor="middle"
                dominantBaseline="central"
                className="text-[9px] font-mono"
                fill="#A1A1AA"
              >
                {formattedValue}
              </text>
            )}

            {/* External label for small bubbles */}
            {r <= 25 && (
              <text
                x={cx}
                y={cy + r + 14}
                textAnchor="middle"
                className="text-[8px] font-mono"
                fill="#A1A1AA"
              >
                {formattedValue}
              </text>
            )}
          </motion.g>
        )
      })}

      {/* Scale type indicator */}
      <text
        x={innerWidth - 4}
        y={12}
        textAnchor="end"
        className="text-[8px] font-mono"
        fill="#52525B"
      >
        {useLogScale ? 'Log scale (area)' : 'Linear scale (area)'}
      </text>
    </>
  )
}

// ── Bar Chart Section ─────────────────────────────────────────────────

function MetricBarChart({
  innerWidth,
  innerHeight,
}: {
  innerWidth: number
  innerHeight: number
}) {
  const barMetrics: { key: MetricKey; label: string }[] = [
    { key: 'layers', label: 'Layers' },
    { key: 'dModel', label: 'd_model' },
    { key: 'heads', label: 'Heads' },
  ]

  const rowHeight = innerHeight / barMetrics.length
  const labelWidth = 55
  const barAreaWidth = innerWidth - labelWidth - 50
  const barHeight = Math.min(10, (rowHeight - 16) / MODELS_SORTED.length - 2)

  return (
    <>
      {barMetrics.map((metric, mi) => {
        const yOffset = mi * rowHeight
        const values = MODELS_SORTED.map((m) => m[metric.key])
        const maxVal = Math.max(...values)
        const barScale = d3.scaleLinear().domain([0, maxVal]).range([0, barAreaWidth])

        return (
          <g key={metric.key} transform={`translate(0, ${yOffset})`}>
            {/* Metric label */}
            <text
              x={labelWidth - 8}
              y={rowHeight / 2}
              textAnchor="end"
              dominantBaseline="central"
              className="text-[9px] font-mono"
              fill="#A1A1AA"
            >
              {metric.label}
            </text>

            {/* Bars for each model */}
            {MODELS_SORTED.map((model, i) => {
              const val = model[metric.key]
              const barW = barScale(val)
              const barY = 6 + i * (barHeight + 2)

              return (
                <g key={model.name}>
                  {/* Background */}
                  <rect
                    x={labelWidth}
                    y={barY}
                    width={barAreaWidth}
                    height={barHeight}
                    rx={2}
                    fill="rgba(255,255,255,0.03)"
                  />
                  {/* Value bar */}
                  <motion.rect
                    x={labelWidth}
                    y={barY}
                    height={barHeight}
                    rx={2}
                    fill={model.color}
                    fillOpacity={0.6}
                    initial={{ width: 0 }}
                    animate={{ width: barW }}
                    transition={{ duration: 0.6, delay: mi * 0.1 + i * 0.05 }}
                  />
                  {/* Model label */}
                  <text
                    x={labelWidth + barW + 4}
                    y={barY + barHeight / 2}
                    dominantBaseline="central"
                    className="text-[7px] font-mono"
                    fill={model.color}
                  >
                    {model.name} ({formatNumber(val)})
                  </text>
                </g>
              )
            })}

            {/* Separator line */}
            {mi < barMetrics.length - 1 && (
              <line
                x1={0}
                y1={rowHeight - 1}
                x2={innerWidth}
                y2={rowHeight - 1}
                stroke="rgba(255,255,255,0.05)"
                strokeWidth={0.5}
              />
            )}
          </g>
        )
      })}
    </>
  )
}

// ── Main Component ────────────────────────────────────────────────────

export function ScaleVisualization() {
  const [useLogScale, setUseLogScale] = useState(true)
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>('params')

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-center gap-3">
          {/* Scale toggle */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-tertiary font-mono">Scale:</span>
            <Button
              variant="secondary"
              size="sm"
              active={useLogScale}
              onClick={() => setUseLogScale(true)}
            >
              Log
            </Button>
            <Button
              variant="secondary"
              size="sm"
              active={!useLogScale}
              onClick={() => setUseLogScale(false)}
            >
              Linear
            </Button>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          {/* Metric selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-tertiary font-mono">Size by:</span>
            {METRICS.map((metric) => (
              <Button
                key={metric.key}
                variant="secondary"
                size="sm"
                active={selectedMetric === metric.key}
                onClick={() => setSelectedMetric(metric.key)}
              >
                {metric.label}
              </Button>
            ))}
          </div>
        </div>
      </GlassCard>

      {/* Bubble chart */}
      <GlassCard className="overflow-hidden">
        <SVGContainer
          aspectRatio={16 / 8}
          minHeight={300}
          maxHeight={480}
          padding={{ top: 20, right: 30, bottom: 30, left: 20 }}
        >
          {({ innerWidth, innerHeight }) => (
            <BubbleChart
              innerWidth={innerWidth}
              innerHeight={innerHeight}
              useLogScale={useLogScale}
              selectedMetric={selectedMetric}
            />
          )}
        </SVGContainer>
      </GlassCard>

      {/* Dimension comparison bar chart */}
      <GlassCard className="overflow-hidden">
        <div className="px-4 pt-3 pb-1">
          <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary">
            Architecture Dimensions Compared
          </h4>
        </div>
        <SVGContainer
          aspectRatio={16 / 6}
          minHeight={180}
          maxHeight={280}
          padding={{ top: 10, right: 20, bottom: 10, left: 10 }}
        >
          {({ innerWidth, innerHeight }) => (
            <MetricBarChart
              innerWidth={innerWidth}
              innerHeight={innerHeight}
            />
          )}
        </SVGContainer>
      </GlassCard>

      {/* Annotation */}
      <GlassCard className="p-4">
        <p className="text-sm text-center" style={{ color: '#A1A1AA' }}>
          The architecture is the same. The difference is how many blocks you stack,
          how wide the embeddings are, and how much data you train on.
        </p>

        {/* Model legend */}
        <div className="flex flex-wrap justify-center gap-3 mt-3">
          {MODELS_SORTED.map((model) => (
            <div key={model.name} className="flex items-center gap-1.5">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: model.color }}
              />
              <span className="text-[10px] font-mono" style={{ color: '#A1A1AA' }}>
                {model.name}
              </span>
              <span className="text-[10px] font-mono" style={{ color: '#52525B' }}>
                ({formatParams(model.params)})
              </span>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  )
}
