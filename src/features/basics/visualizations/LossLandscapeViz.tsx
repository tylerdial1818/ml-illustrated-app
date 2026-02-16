import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'
import { Toggle } from '../../../components/ui/Toggle'
import {
  makeLinearData,
  computeResiduals,
  computeLoss,
  computeLossSurface,
  olsSolution,
  type LossType,
} from '../../../lib/algorithms/basics/linearModel'

// ── Constants ─────────────────────────────────────────────────────────
const SLOPE_RANGE: [number, number] = [-2, 4]
const INTERCEPT_RANGE: [number, number] = [-5, 10]
const TRUE_SLOPE = 1.8
const TRUE_INTERCEPT = 2.0
const NOISE_STD = 2.5
const N_POINTS = 20
const OUTLIER_X = 7
const OUTLIER_Y = -8
const SURFACE_RES = 40

const LOSS_LABELS: Record<LossType, string> = {
  mse: 'MSE (Mean Squared Error)',
  mae: 'MAE (Mean Absolute Error)',
  huber: 'Huber Loss',
}

// ── Part A: Scatter plot with manipulable line ────────────────────────
function ScatterPanel({
  innerWidth,
  innerHeight,
  data,
  slope,
  intercept,
  lossType,
}: {
  innerWidth: number
  innerHeight: number
  data: { x: number[]; y: number[] }
  slope: number
  intercept: number
  lossType: LossType
}) {
  const residuals = computeResiduals(data.x, data.y, slope, intercept)
  const loss = computeLoss(data.x, data.y, slope, intercept, lossType)

  const allY = [...data.y, ...data.x.map((xi) => slope * xi + intercept)]
  const xExtent = d3.extent(data.x) as [number, number]
  const yExtent = d3.extent(allY) as [number, number]
  const xPad = (xExtent[1] - xExtent[0]) * 0.1
  const yPad = (yExtent[1] - yExtent[0]) * 0.15

  const xScale = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([0, innerWidth])
  const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([innerHeight, 0])

  // Residual color scale
  const maxResidual = Math.max(...residuals.map(Math.abs), 1)
  const residualColor = (r: number) => {
    const t = Math.min(Math.abs(r) / maxResidual, 1)
    return d3.interpolateRgb('#4ADE80', '#F87171')(t)
  }

  // Line endpoints
  const lineX1 = xExtent[0] - xPad
  const lineX2 = xExtent[1] + xPad
  const lineY1 = slope * lineX1 + intercept
  const lineY2 = slope * lineX2 + intercept

  return (
    <>
      {/* Residual lines */}
      {data.x.map((xi, i) => {
        const predicted = slope * xi + intercept
        return (
          <motion.line
            key={`res-${i}`}
            x1={xScale(xi)}
            y1={yScale(data.y[i])}
            x2={xScale(xi)}
            y2={yScale(predicted)}
            stroke={residualColor(residuals[i])}
            strokeWidth={1.5}
            strokeDasharray="3 2"
            strokeOpacity={0.7}
            animate={{
              y2: yScale(predicted),
            }}
            transition={{ duration: 0.05 }}
          />
        )
      })}

      {/* Regression line */}
      <motion.line
        x1={xScale(lineX1)}
        y1={yScale(lineY1)}
        x2={xScale(lineX2)}
        y2={yScale(lineY2)}
        stroke="#818CF8"
        strokeWidth={2}
        strokeOpacity={0.9}
        animate={{
          y1: yScale(lineY1),
          y2: yScale(lineY2),
        }}
        transition={{ duration: 0.05 }}
      />

      {/* Data points */}
      {data.x.map((xi, i) => (
        <circle
          key={`pt-${i}`}
          cx={xScale(xi)}
          cy={yScale(data.y[i])}
          r={4}
          fill={i === data.x.length - 1 && data.x.length > N_POINTS ? '#F87171' : '#E4E4E7'}
          fillOpacity={0.9}
          stroke="rgba(255,255,255,0.2)"
          strokeWidth={0.5}
        />
      ))}

      {/* Loss display */}
      <rect
        x={innerWidth - 130}
        y={0}
        width={130}
        height={36}
        rx={6}
        fill="rgba(0,0,0,0.5)"
        stroke="rgba(255,255,255,0.1)"
        strokeWidth={0.5}
      />
      <text
        x={innerWidth - 65}
        y={14}
        textAnchor="middle"
        className="text-[8px] font-mono uppercase tracking-wider"
        fill="#71717A"
      >
        {LOSS_LABELS[lossType].split(' ')[0]}
      </text>
      <text
        x={innerWidth - 65}
        y={28}
        textAnchor="middle"
        className="text-[14px] font-mono font-bold"
        fill={loss < 10 ? '#4ADE80' : loss < 30 ? '#FBBF24' : '#F87171'}
      >
        {loss.toFixed(2)}
      </text>
    </>
  )
}

// ── Part B: Loss surface heatmap ──────────────────────────────────────
function LossSurfacePanel({
  innerWidth,
  innerHeight,
  surface,
  slope,
  intercept,
  optSlope,
  optIntercept,
}: {
  innerWidth: number
  innerHeight: number
  surface: { slopes: number[]; intercepts: number[]; losses: number[][] }
  slope: number
  intercept: number
  optSlope: number
  optIntercept: number
}) {
  const { slopes, intercepts, losses } = surface
  const res = slopes.length

  const xScale = d3.scaleLinear().domain([slopes[0], slopes[res - 1]]).range([0, innerWidth])
  const yScale = d3.scaleLinear().domain([intercepts[0], intercepts[res - 1]]).range([innerHeight, 0])

  // Find min/max loss for color scale
  let minLoss = Infinity
  let maxLoss = -Infinity
  for (const row of losses) {
    for (const v of row) {
      if (v < minLoss) minLoss = v
      if (v > maxLoss) maxLoss = v
    }
  }

  const colorScale = d3
    .scaleSequential(d3.interpolateInferno)
    .domain([maxLoss, minLoss]) // reversed: low loss = bright

  const cellW = innerWidth / res
  const cellH = innerHeight / res

  const isNearOptimum = Math.abs(slope - optSlope) < 0.3 && Math.abs(intercept - optIntercept) < 0.5

  return (
    <>
      {/* Heatmap cells */}
      {losses.map((row, j) =>
        row.map((loss, i) => (
          <rect
            key={`cell-${j}-${i}`}
            x={i * cellW}
            y={(res - 1 - j) * cellH}
            width={cellW + 0.5}
            height={cellH + 0.5}
            fill={colorScale(loss)}
          />
        ))
      )}

      {/* Axis labels */}
      <text x={innerWidth / 2} y={innerHeight + 16} textAnchor="middle" className="text-[9px] font-mono" fill="#71717A">
        Slope
      </text>
      <text
        x={-innerHeight / 2}
        y={-10}
        textAnchor="middle"
        className="text-[9px] font-mono"
        fill="#71717A"
        transform="rotate(-90)"
      >
        Intercept
      </text>

      {/* Optimum marker */}
      <circle
        cx={xScale(optSlope)}
        cy={yScale(optIntercept)}
        r={5}
        fill="none"
        stroke="#4ADE80"
        strokeWidth={1.5}
        strokeDasharray="2 2"
      />

      {/* Current position dot */}
      <motion.circle
        cx={xScale(slope)}
        cy={yScale(intercept)}
        r={6}
        fill="#818CF8"
        stroke="white"
        strokeWidth={2}
        animate={{
          cx: xScale(slope),
          cy: yScale(intercept),
        }}
        transition={{ duration: 0.1 }}
      />

      {/* Glow at optimum */}
      {isNearOptimum && (
        <motion.circle
          cx={xScale(slope)}
          cy={yScale(intercept)}
          r={12}
          fill="none"
          stroke="#4ADE80"
          strokeWidth={2}
          strokeOpacity={0.5}
          animate={{ r: [12, 18, 12], strokeOpacity: [0.5, 0.2, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}

      {/* Label */}
      <text x={4} y={innerHeight + 16} className="text-[8px] font-mono" fill="#52525B">
        {SLOPE_RANGE[0]}
      </text>
      <text x={innerWidth - 4} y={innerHeight + 16} textAnchor="end" className="text-[8px] font-mono" fill="#52525B">
        {SLOPE_RANGE[1]}
      </text>
    </>
  )
}

// ── Loss comparison mini-cards ────────────────────────────────────────
function LossComparisonCards({
  data,
  slope,
  intercept,
}: {
  data: { x: number[]; y: number[] }
  slope: number
  intercept: number
}) {
  const types: { key: LossType; label: string; color: string; desc: string }[] = [
    { key: 'mae', label: 'MAE', color: '#34D399', desc: 'Treats all errors equally' },
    { key: 'mse', label: 'MSE', color: '#6366F1', desc: 'Penalizes large errors more' },
    { key: 'huber', label: 'Huber', color: '#FBBF24', desc: 'MSE for small, MAE for large' },
  ]

  return (
    <div className="grid grid-cols-3 gap-3">
      {types.map((t) => {
        const loss = computeLoss(data.x, data.y, slope, intercept, t.key)
        return (
          <div
            key={t.key}
            className="rounded-lg border p-3 text-center"
            style={{
              backgroundColor: `${t.color}08`,
              borderColor: `${t.color}25`,
            }}
          >
            <p className="text-[10px] font-mono font-medium" style={{ color: t.color }}>
              {t.label}
            </p>
            <p className="text-lg font-mono font-bold mt-1" style={{ color: t.color }}>
              {loss.toFixed(2)}
            </p>
            <p className="text-[9px] text-text-tertiary mt-1">{t.desc}</p>
          </div>
        )
      })}
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function LossLandscapeViz() {
  const [slope, setSlope] = useState(0.5)
  const [intercept, setIntercept] = useState(0)
  const [lossType, setLossType] = useState<LossType>('mse')
  const [showOutlier, setShowOutlier] = useState(false)

  const baseData = useMemo(
    () => makeLinearData(N_POINTS, TRUE_SLOPE, TRUE_INTERCEPT, NOISE_STD, 42),
    []
  )

  const data = useMemo(() => {
    if (!showOutlier) return baseData
    return {
      x: [...baseData.x, OUTLIER_X],
      y: [...baseData.y, OUTLIER_Y],
    }
  }, [baseData, showOutlier])

  const ols = useMemo(() => olsSolution(data.x, data.y), [data])

  const surface = useMemo(
    () => computeLossSurface(data.x, data.y, SLOPE_RANGE, INTERCEPT_RANGE, SURFACE_RES, lossType),
    [data, lossType]
  )

  const handleRandomStart = useCallback(() => {
    setSlope(Math.random() * 4 - 1)
    setIntercept(Math.random() * 12 - 3)
  }, [])

  const handleBestFit = useCallback(() => {
    setSlope(ols.slope)
    setIntercept(ols.intercept)
  }, [ols])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Slider
            label="Slope"
            value={slope}
            min={SLOPE_RANGE[0]}
            max={SLOPE_RANGE[1]}
            step={0.05}
            onChange={setSlope}
            formatValue={(v) => v.toFixed(2)}
            className="w-44"
          />
          <Slider
            label="Intercept"
            value={intercept}
            min={INTERCEPT_RANGE[0]}
            max={INTERCEPT_RANGE[1]}
            step={0.1}
            onChange={setIntercept}
            formatValue={(v) => v.toFixed(1)}
            className="w-44"
          />

          <div className="h-6 w-px bg-obsidian-border" />

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handleRandomStart}>
              Random Start
            </Button>
            <Button variant="secondary" size="sm" onClick={handleBestFit}>
              Best Fit
            </Button>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <div className="flex items-center gap-1">
            {(['mse', 'mae', 'huber'] as LossType[]).map((lt) => (
              <Button
                key={lt}
                variant="secondary"
                size="sm"
                active={lossType === lt}
                onClick={() => setLossType(lt)}
              >
                {lt.toUpperCase()}
              </Button>
            ))}
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <Toggle label="Add Outlier" checked={showOutlier} onChange={setShowOutlier} />
        </div>
      </GlassCard>

      {/* Part A: Scatter plot with line */}
      <GlassCard className="overflow-hidden">
        <SVGContainer
          aspectRatio={16 / 7}
          minHeight={240}
          maxHeight={340}
          padding={{ top: 15, right: 20, bottom: 20, left: 30 }}
        >
          {({ innerWidth, innerHeight }) => (
            <ScatterPanel
              innerWidth={innerWidth}
              innerHeight={innerHeight}
              data={data}
              slope={slope}
              intercept={intercept}
              lossType={lossType}
            />
          )}
        </SVGContainer>
      </GlassCard>

      {/* Part B: Loss surface */}
      <GlassCard className="overflow-hidden">
        <div className="px-4 pt-3 pb-1">
          <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary">
            Loss Surface
          </h4>
          <p className="text-[9px] text-text-tertiary mt-0.5">
            Every combination of slope and intercept has a loss value. Training means finding the lowest point.
          </p>
        </div>
        <SVGContainer
          aspectRatio={16 / 9}
          minHeight={220}
          maxHeight={340}
          padding={{ top: 10, right: 20, bottom: 24, left: 24 }}
        >
          {({ innerWidth, innerHeight }) => (
            <LossSurfacePanel
              innerWidth={innerWidth}
              innerHeight={innerHeight}
              surface={surface}
              slope={slope}
              intercept={intercept}
              optSlope={ols.slope}
              optIntercept={ols.intercept}
            />
          )}
        </SVGContainer>
      </GlassCard>

      {/* Loss function comparison */}
      <div>
        <p className="text-xs font-medium text-text-secondary mb-2">
          Loss Function Comparison
        </p>
        <p className="text-[10px] text-text-tertiary mb-3">
          Same data, same model. Different loss functions measure &quot;wrong&quot; differently.
          {showOutlier && (
            <span className="text-red-400 font-medium"> Notice how the outlier affects MSE vs MAE.</span>
          )}
        </p>
        <LossComparisonCards data={data} slope={slope} intercept={intercept} />
      </div>
    </div>
  )
}
