import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import { Toggle } from '../../../components/ui/Toggle'
import {
  makeMultiScaleData,
  scaleDataset2D,
  compute2DLossSurface,
  run2DGradientDescent,
  minMaxScale,
  standardize,
  robustScale,
  type ScaleMethod,
} from '../../../lib/algorithms/basics/scaling'

// ── Constants ─────────────────────────────────────────────────────────
const N_POINTS = 40
const SURFACE_RES = 40
const GD_STEPS = 60

const METHOD_LABELS: Record<ScaleMethod, string> = {
  minmax: 'Min-Max',
  standard: 'Standardization',
  robust: 'Robust',
}

const METHOD_COLORS: Record<ScaleMethod, string> = {
  minmax: '#6366F1',
  standard: '#34D399',
  robust: '#FBBF24',
}

// ── Contour + GD Path Panel ──────────────────────────────────────────
function ContourGDPanel({
  innerWidth,
  innerHeight,
  surface,
  gdPath,
  label,
  w1Label,
  w2Label,
}: {
  innerWidth: number
  innerHeight: number
  surface: { w1s: number[]; w2s: number[]; losses: number[][] }
  gdPath: { w1: number; w2: number; loss: number }[]
  label: string
  w1Label: string
  w2Label: string
}) {
  const { w1s, w2s, losses } = surface
  const res = w1s.length

  const xScale = d3.scaleLinear().domain([w1s[0], w1s[res - 1]]).range([0, innerWidth])
  const yScale = d3.scaleLinear().domain([w2s[0], w2s[res - 1]]).range([innerHeight, 0])

  let minLoss = Infinity
  let maxLoss = -Infinity
  for (const row of losses) {
    for (const v of row) {
      if (v < minLoss) minLoss = v
      if (v > maxLoss) maxLoss = v
    }
  }
  const colorScale = d3.scaleSequential(d3.interpolateInferno).domain([maxLoss, minLoss])

  const cellW = innerWidth / res
  const cellH = innerHeight / res

  // Filter GD path to visible area
  const visiblePath = gdPath.filter(
    (s) =>
      s.w1 >= w1s[0] && s.w1 <= w1s[res - 1] &&
      s.w2 >= w2s[0] && s.w2 <= w2s[res - 1]
  )

  return (
    <>
      {/* Heatmap */}
      {losses.map((row, j) =>
        row.map((loss, i) => (
          <rect
            key={`c-${j}-${i}`}
            x={i * cellW}
            y={(res - 1 - j) * cellH}
            width={cellW + 0.5}
            height={cellH + 0.5}
            fill={colorScale(loss)}
          />
        ))
      )}

      {/* GD path */}
      {visiblePath.length > 1 && (
        <path
          d={visiblePath
            .map((s, i) => `${i === 0 ? 'M' : 'L'}${xScale(s.w1)},${yScale(s.w2)}`)
            .join(' ')}
          fill="none"
          stroke="#F472B6"
          strokeWidth={1.5}
          strokeOpacity={0.8}
          strokeLinejoin="round"
        />
      )}

      {/* Path dots */}
      {visiblePath.map((s, i) => (
        <circle
          key={`pd-${i}`}
          cx={xScale(s.w1)}
          cy={yScale(s.w2)}
          r={i === 0 ? 4 : i === visiblePath.length - 1 ? 4 : 1.5}
          fill={i === 0 ? '#F472B6' : i === visiblePath.length - 1 ? '#4ADE80' : '#F472B6'}
          fillOpacity={i === 0 || i === visiblePath.length - 1 ? 1 : 0.4}
          stroke={i === 0 || i === visiblePath.length - 1 ? 'white' : 'none'}
          strokeWidth={1.5}
        />
      ))}

      {/* Label */}
      <rect x={0} y={0} width={innerWidth} height={20} rx={0} fill="rgba(0,0,0,0.3)" />
      <text x={innerWidth / 2} y={14} textAnchor="middle" className="text-[10px] font-mono font-medium" fill="#E4E4E7">
        {label}
      </text>

      {/* Step count */}
      <text x={innerWidth - 4} y={innerHeight - 4} textAnchor="end" className="text-[9px] font-mono" fill="#71717A">
        {visiblePath.length - 1} steps
      </text>

      {/* Axis labels */}
      <text x={innerWidth / 2} y={innerHeight + 14} textAnchor="middle" className="text-[8px] font-mono" fill="#71717A">
        {w1Label}
      </text>
      <text x={-innerHeight / 2} y={-6} textAnchor="middle" className="text-[8px] font-mono" fill="#71717A" transform="rotate(-90)">
        {w2Label}
      </text>
    </>
  )
}

// ── Scaling Method Comparison Cards ──────────────────────────────────
function ScalingMethodCards({
  values,
  hasOutlier,
}: {
  values: number[]
  hasOutlier: boolean
}) {
  const methods: { key: ScaleMethod; fn: (v: number[]) => { scaled: number[] }; formula: string }[] = [
    { key: 'minmax', fn: minMaxScale, formula: '(x - min) / (max - min)' },
    { key: 'standard', fn: standardize, formula: '(x - μ) / σ' },
    { key: 'robust', fn: robustScale, formula: '(x - median) / IQR' },
  ]

  return (
    <div className="grid grid-cols-3 gap-3">
      {methods.map((m) => {
        const { scaled } = m.fn(values)
        const extent = d3.extent(scaled) as [number, number]

        return (
          <GlassCard key={m.key} className="p-3">
            <p className="text-[10px] font-mono font-medium" style={{ color: METHOD_COLORS[m.key] }}>
              {METHOD_LABELS[m.key]}
            </p>
            <p className="text-[8px] font-mono text-text-tertiary mt-0.5">{m.formula}</p>

            {/* Mini histogram */}
            <div className="mt-2">
              <MiniHistogram values={scaled} color={METHOD_COLORS[m.key]} height={40} />
            </div>

            <div className="flex justify-between mt-1">
              <p className="text-[8px] font-mono text-text-tertiary">
                [{extent[0].toFixed(1)}, {extent[1].toFixed(1)}]
              </p>
              {hasOutlier && m.key === 'minmax' && (
                <p className="text-[8px] font-mono text-error">distorted</p>
              )}
              {hasOutlier && m.key === 'robust' && (
                <p className="text-[8px] font-mono text-success">stable</p>
              )}
            </div>
          </GlassCard>
        )
      })}
    </div>
  )
}

function MiniHistogram({
  values,
  color,
  height,
}: {
  values: number[]
  color: string
  height: number
}) {
  const nBins = 15
  const extent = d3.extent(values) as [number, number]
  const binWidth = (extent[1] - extent[0]) / nBins || 1
  const bins = new Array(nBins).fill(0)

  for (const v of values) {
    const idx = Math.min(Math.floor((v - extent[0]) / binWidth), nBins - 1)
    bins[idx]++
  }

  const maxCount = Math.max(...bins)
  const barWidth = 100 / nBins

  return (
    <svg width="100%" height={height} viewBox={`0 0 100 ${height}`} preserveAspectRatio="none">
      {bins.map((count, i) => {
        const barH = (count / maxCount) * height * 0.9
        return (
          <rect
            key={i}
            x={i * barWidth}
            y={height - barH}
            width={barWidth - 0.5}
            height={barH}
            fill={color}
            fillOpacity={0.6}
          />
        )
      })}
    </svg>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function FeatureScalingViz() {
  const [scaleMethod, setScaleMethod] = useState<ScaleMethod>('standard')
  const [showOutlier, setShowOutlier] = useState(false)
  const [isScaled, setIsScaled] = useState(false)

  const rawData = useMemo(() => makeMultiScaleData(N_POINTS, 42), [])

  const data = useMemo(() => {
    if (!showOutlier) return rawData
    return {
      squareFeet: [...rawData.squareFeet, 15000],
      bedrooms: [...rawData.bedrooms, 2],
      price: [...rawData.price, 200000],
    }
  }, [rawData, showOutlier])

  // Unscaled surface + GD
  const unscaledSurface = useMemo(
    () => compute2DLossSurface(data.squareFeet, data.bedrooms, data.price, [-50, 250], [-50000, 150000], SURFACE_RES),
    [data]
  )

  const unscaledGD = useMemo(
    () => run2DGradientDescent(data.squareFeet, data.bedrooms, data.price, 200, 100000, 0.0000000015, GD_STEPS),
    [data]
  )

  // Scaled data
  const { x1Scaled, x2Scaled } = useMemo(
    () => scaleDataset2D(data.squareFeet, data.bedrooms, scaleMethod),
    [data, scaleMethod]
  )

  // Scale target too
  const priceScaled = useMemo(() => {
    const fn = scaleMethod === 'minmax' ? minMaxScale : scaleMethod === 'standard' ? standardize : robustScale
    return fn(data.price).scaled
  }, [data, scaleMethod])

  const scaledSurface = useMemo(
    () => compute2DLossSurface(x1Scaled, x2Scaled, priceScaled, [-3, 3], [-3, 3], SURFACE_RES),
    [x1Scaled, x2Scaled, priceScaled]
  )

  const scaledGD = useMemo(
    () => run2DGradientDescent(x1Scaled, x2Scaled, priceScaled, 2.5, 2.5, 0.05, GD_STEPS),
    [x1Scaled, x2Scaled, priceScaled]
  )

  // Values for scaling comparison (square feet feature)
  const comparisonValues = useMemo(() => data.squareFeet, [data])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex items-center gap-1">
            {(['minmax', 'standard', 'robust'] as ScaleMethod[]).map((m) => (
              <Button
                key={m}
                variant="secondary"
                size="sm"
                active={scaleMethod === m}
                onClick={() => setScaleMethod(m)}
              >
                {METHOD_LABELS[m]}
              </Button>
            ))}
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <Toggle label="Add Outlier" checked={showOutlier} onChange={setShowOutlier} />

          <div className="h-6 w-px bg-obsidian-border" />

          <Button
            variant={isScaled ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setIsScaled(!isScaled)}
          >
            {isScaled ? 'Showing: Scaled' : 'Showing: Unscaled'}
          </Button>
        </div>
      </GlassCard>

      {/* Dual contour panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <GlassCard className="overflow-hidden">
          <SVGContainer
            aspectRatio={1}
            minHeight={250}
            maxHeight={360}
            padding={{ top: 24, right: 16, bottom: 20, left: 24 }}
          >
            {({ innerWidth, innerHeight }) => (
              <ContourGDPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                surface={unscaledSurface}
                gdPath={unscaledGD}
                label="Unscaled: GD zigzags"
                w1Label="w₁ (sq ft)"
                w2Label="w₂ (bedrooms)"
              />
            )}
          </SVGContainer>
        </GlassCard>

        <GlassCard className="overflow-hidden">
          <SVGContainer
            aspectRatio={1}
            minHeight={250}
            maxHeight={360}
            padding={{ top: 24, right: 16, bottom: 20, left: 24 }}
          >
            {({ innerWidth, innerHeight }) => (
              <ContourGDPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                surface={scaledSurface}
                gdPath={scaledGD}
                label={`Scaled (${METHOD_LABELS[scaleMethod]}): GD goes direct`}
                w1Label="w₁ (scaled)"
                w2Label="w₂ (scaled)"
              />
            )}
          </SVGContainer>
        </GlassCard>
      </div>

      {/* Scaling methods comparison */}
      <div>
        <p className="text-xs font-medium text-text-secondary mb-2">
          Scaling Methods Compared
        </p>
        <p className="text-[10px] text-text-tertiary mb-3">
          Same feature (square footage) transformed three different ways. Each histogram shows the distribution after scaling.
          {showOutlier && (
            <span className="text-red-400 font-medium"> Notice how the outlier distorts Min-Max but not Robust scaling.</span>
          )}
        </p>
        <ScalingMethodCards values={comparisonValues} hasOutlier={showOutlier} />
      </div>
    </div>
  )
}
