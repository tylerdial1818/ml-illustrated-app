import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Toggle } from '../../../components/ui/Toggle'
import { Select } from '../../../components/ui/Select'
import { Button } from '../../../components/ui/Button'
import {
  BayesianLinearRegression,
  makeBayesianLinearData,
} from '../../../lib/algorithms/bayesian/bayesianLinearRegression'

// ── Colors (from spec, non-negotiable) ──────────────────────────────
const COLORS = {
  prior: '#A1A1AA',
  posterior: '#6366F1',
  posteriorFill: 'rgba(99, 102, 241, 0.12)',
  data: '#F4F4F5',
  ols: '#FBBF24',
  predictive: '#34D399',
  credibleBand: 'rgba(99, 102, 241, 0.15)',
  gridLine: 'rgba(255,255,255,0.06)',
}

const PRIOR_TYPES = [
  { value: 'uninformed', label: 'Uninformed (wide)' },
  { value: 'informative', label: 'Informative (narrow)' },
  { value: 'wrong', label: 'Wrong prior' },
]

const TRUE_SLOPE = 1.5
const TRUE_INTERCEPT = 2.0

function getPriorParams(type: string): { mean: number[]; cov: number[][] } {
  switch (type) {
    case 'informative':
      return { mean: [2, 1.5], cov: [[1, 0], [0, 1]] }
    case 'wrong':
      return { mean: [-3, -2], cov: [[1, 0], [0, 1]] }
    default: // uninformed
      return { mean: [0, 0], cov: [[10, 0], [0, 10]] }
  }
}

// ── Contour rendering ───────────────────────────────────────────────
function ContourPanel({
  model,
  width,
  height,
}: {
  model: BayesianLinearRegression
  width: number
  height: number
}) {
  const pad = { top: 30, right: 15, bottom: 35, left: 45 }
  const w = width - pad.left - pad.right
  const h = height - pad.top - pad.bottom

  const slopeRange: [number, number] = [-4, 6]
  const interceptRange: [number, number] = [-6, 10]
  const res = 40

  const priorContours = useMemo(
    () => model.getPriorContours(slopeRange, interceptRange, res),
    [model.priorMean, model.priorCov]
  )
  const posteriorContours = useMemo(
    () => model.hasData ? model.getPosteriorContours(slopeRange, interceptRange, res) : null,
    [model.posteriorMean, model.posteriorCov, model.hasData]
  )

  const xScale = (v: number) => pad.left + ((v - slopeRange[0]) / (slopeRange[1] - slopeRange[0])) * w
  const yScale = (v: number) => pad.top + h - ((v - interceptRange[0]) / (interceptRange[1] - interceptRange[0])) * h

  // Compute contour levels
  const maxDensity = useMemo(() => {
    let mx = 0
    for (const row of priorContours.density) for (const v of row) if (v > mx) mx = v
    if (posteriorContours) for (const row of posteriorContours.density) for (const v of row) if (v > mx) mx = v
    return mx
  }, [priorContours, posteriorContours])

  const levels = [0.1, 0.3, 0.5, 0.7, 0.9]

  // Simple contour: draw cells above threshold
  function renderContour(
    data: typeof priorContours,
    color: string,
    dashed: boolean,
    fillOpacity: number
  ) {
    const cellW = w / (res - 1)
    const cellH = h / (res - 1)
    return levels.map((level) => {
      const threshold = level * maxDensity
      const rects: JSX.Element[] = []
      for (let j = 0; j < res; j++) {
        for (let i = 0; i < res; i++) {
          if (data.density[j][i] >= threshold) {
            rects.push(
              <rect
                key={`${i}-${j}`}
                x={pad.left + i * cellW - cellW / 2}
                y={pad.top + (res - 1 - j) * cellH - cellH / 2}
                width={cellW}
                height={cellH}
                fill={color}
                fillOpacity={fillOpacity * (1 - level * 0.6)}
                stroke={dashed ? color : 'none'}
                strokeWidth={dashed ? 0.5 : 0}
                strokeDasharray={dashed ? '2,2' : undefined}
                strokeOpacity={dashed ? 0.4 : 0}
              />
            )
          }
        }
      }
      return <g key={level}>{rects}</g>
    })
  }

  // OLS and MAP markers
  const olsEst = useMemo(
    () => model.hasData ? { slope: model.posteriorMean[1], intercept: model.posteriorMean[0] } : null,
    [model.posteriorMean, model.hasData]
  )

  return (
    <svg width={width} height={height} className="block">
      {/* Grid */}
      {[-2, 0, 2, 4].map((v) => (
        <line key={`gx-${v}`} x1={xScale(v)} y1={pad.top} x2={xScale(v)} y2={pad.top + h} stroke={COLORS.gridLine} />
      ))}
      {[-4, -2, 0, 2, 4, 6, 8].map((v) => (
        <line key={`gy-${v}`} x1={pad.left} y1={yScale(v)} x2={pad.left + w} y2={yScale(v)} stroke={COLORS.gridLine} />
      ))}

      {/* Prior contours */}
      {renderContour(priorContours, COLORS.prior, true, 0.08)}

      {/* Posterior contours */}
      {posteriorContours && renderContour(posteriorContours, COLORS.posterior, false, 0.15)}

      {/* MAP marker */}
      {model.hasData && (
        <>
          <line
            x1={xScale(model.posteriorMean[1]) - 6} y1={yScale(model.posteriorMean[0])}
            x2={xScale(model.posteriorMean[1]) + 6} y2={yScale(model.posteriorMean[0])}
            stroke={COLORS.posterior} strokeWidth={2}
          />
          <line
            x1={xScale(model.posteriorMean[1])} y1={yScale(model.posteriorMean[0]) - 6}
            x2={xScale(model.posteriorMean[1])} y2={yScale(model.posteriorMean[0]) + 6}
            stroke={COLORS.posterior} strokeWidth={2}
          />
        </>
      )}

      {/* True params marker */}
      <circle cx={xScale(TRUE_SLOPE)} cy={yScale(TRUE_INTERCEPT)} r={3} fill={COLORS.data} fillOpacity={0.5} />

      {/* Axes labels */}
      <text x={pad.left + w / 2} y={height - 5} textAnchor="middle" className="text-[10px] fill-text-tertiary font-mono">slope (β₁)</text>
      <text x={12} y={pad.top + h / 2} textAnchor="middle" transform={`rotate(-90, 12, ${pad.top + h / 2})`} className="text-[10px] fill-text-tertiary font-mono">intercept (β₀)</text>

      {/* Title */}
      <text x={pad.left + w / 2} y={16} textAnchor="middle" className="text-[11px] fill-text-secondary font-medium">Parameter Space</text>

      {/* Axis ticks */}
      {[-2, 0, 2, 4].map((v) => (
        <text key={`tx-${v}`} x={xScale(v)} y={pad.top + h + 14} textAnchor="middle" className="text-[9px] fill-text-tertiary font-mono">{v}</text>
      ))}
      {[-4, 0, 4, 8].map((v) => (
        <text key={`ty-${v}`} x={pad.left - 6} y={yScale(v) + 3} textAnchor="end" className="text-[9px] fill-text-tertiary font-mono">{v}</text>
      ))}
    </svg>
  )
}

// ── Data space panel ────────────────────────────────────────────────
function DataSpacePanel({
  model,
  dataX,
  dataY,
  nSamples,
  showOLS,
  showBand,
  width,
  height,
  onAddPoint,
  hoverX,
  onHoverX,
}: {
  model: BayesianLinearRegression
  dataX: number[]
  dataY: number[]
  nSamples: number
  showOLS: boolean
  showBand: boolean
  width: number
  height: number
  onAddPoint: (x: number, y: number) => void
  hoverX: number | null
  onHoverX: (x: number | null) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const pad = { top: 30, right: 15, bottom: 35, left: 45 }
  const w = width - pad.left - pad.right
  const h = height - pad.top - pad.bottom

  const xDomain: [number, number] = [-4, 8]
  const yDomain: [number, number] = [-8, 18]

  const xScale = (v: number) => pad.left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * w
  const yScale = (v: number) => pad.top + h - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * h

  const xInv = (px: number) => xDomain[0] + ((px - pad.left) / w) * (xDomain[1] - xDomain[0])
  const yInv = (py: number) => yDomain[0] + ((pad.top + h - py) / h) * (yDomain[1] - yDomain[0])

  // Sample lines from posterior
  const sampleLines = useMemo(() => {
    if (!model.hasData) return []
    const weights = model.samplePosteriorWeights(nSamples, 42)
    return weights.map((w) => ({
      intercept: w[0],
      slope: w[1],
    }))
  }, [model.posteriorMean, model.posteriorCov, nSamples, model.hasData])

  // Credible band
  const xRange = useMemo(() => {
    const pts: number[] = []
    for (let i = 0; i <= 60; i++) pts.push(xDomain[0] + (xDomain[1] - xDomain[0]) * (i / 60))
    return pts
  }, [])
  const band = useMemo(
    () => model.hasData ? model.getCredibleBand(xRange) : null,
    [model.posteriorMean, model.posteriorCov, model.noiseVariance, model.hasData]
  )

  // OLS
  const ols = useMemo(
    () => dataX.length > 1 ? BayesianLinearRegression.olsEstimate(dataX, dataY) : null,
    [dataX, dataY]
  )

  const handleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = svgRef.current?.getBoundingClientRect()
      if (!rect) return
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      const x = xInv(px)
      const y = yInv(py)
      if (x >= xDomain[0] && x <= xDomain[1] && y >= yDomain[0] && y <= yDomain[1]) {
        onAddPoint(x, y)
      }
    },
    [onAddPoint]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = svgRef.current?.getBoundingClientRect()
      if (!rect) return
      const px = e.clientX - rect.left
      const x = xInv(px)
      if (x >= xDomain[0] && x <= xDomain[1]) {
        onHoverX(x)
      } else {
        onHoverX(null)
      }
    },
    [onHoverX]
  )

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="block cursor-crosshair"
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => onHoverX(null)}
    >
      {/* Grid */}
      {[-2, 0, 2, 4, 6].map((v) => (
        <line key={`gx-${v}`} x1={xScale(v)} y1={pad.top} x2={xScale(v)} y2={pad.top + h} stroke={COLORS.gridLine} />
      ))}
      {[-5, 0, 5, 10, 15].map((v) => (
        <line key={`gy-${v}`} x1={pad.left} y1={yScale(v)} x2={pad.left + w} y2={yScale(v)} stroke={COLORS.gridLine} />
      ))}

      {/* Credible band */}
      {showBand && band && (
        <path
          d={
            xRange.map((x, i) => `${i === 0 ? 'M' : 'L'}${xScale(x)},${yScale(band.upper[i])}`).join(' ') +
            ' ' +
            [...xRange].reverse().map((x, i) => `L${xScale(x)},${yScale(band.lower[xRange.length - 1 - i])}`).join(' ') +
            ' Z'
          }
          fill={COLORS.credibleBand}
        />
      )}

      {/* Posterior sample lines */}
      {sampleLines.map((line, i) => (
        <line
          key={i}
          x1={xScale(xDomain[0])}
          y1={yScale(line.intercept + line.slope * xDomain[0])}
          x2={xScale(xDomain[1])}
          y2={yScale(line.intercept + line.slope * xDomain[1])}
          stroke={COLORS.posterior}
          strokeWidth={1}
          strokeOpacity={0.2}
        />
      ))}

      {/* Posterior mean line */}
      {model.hasData && (
        <line
          x1={xScale(xDomain[0])}
          y1={yScale(model.posteriorMean[0] + model.posteriorMean[1] * xDomain[0])}
          x2={xScale(xDomain[1])}
          y2={yScale(model.posteriorMean[0] + model.posteriorMean[1] * xDomain[1])}
          stroke={COLORS.posterior}
          strokeWidth={2}
        />
      )}

      {/* OLS line */}
      {showOLS && ols && (
        <line
          x1={xScale(xDomain[0])}
          y1={yScale(ols.intercept + ols.slope * xDomain[0])}
          x2={xScale(xDomain[1])}
          y2={yScale(ols.intercept + ols.slope * xDomain[1])}
          stroke={COLORS.ols}
          strokeWidth={2}
          strokeDasharray="6,3"
        />
      )}

      {/* Data points */}
      {dataX.map((x, i) => (
        <circle
          key={i}
          cx={xScale(x)}
          cy={yScale(dataY[i])}
          r={3.5}
          fill={COLORS.data}
          fillOpacity={0.8}
          stroke={COLORS.data}
          strokeWidth={0.5}
          strokeOpacity={0.3}
        />
      ))}

      {/* Hover line */}
      {hoverX !== null && (
        <line
          x1={xScale(hoverX)}
          y1={pad.top}
          x2={xScale(hoverX)}
          y2={pad.top + h}
          stroke={COLORS.predictive}
          strokeWidth={1}
          strokeDasharray="4,3"
          strokeOpacity={0.6}
        />
      )}

      {/* Axes */}
      <text x={pad.left + w / 2} y={height - 5} textAnchor="middle" className="text-[10px] fill-text-tertiary font-mono">x</text>
      <text x={12} y={pad.top + h / 2} textAnchor="middle" transform={`rotate(-90, 12, ${pad.top + h / 2})`} className="text-[10px] fill-text-tertiary font-mono">y</text>

      {/* Title */}
      <text x={pad.left + w / 2} y={16} textAnchor="middle" className="text-[11px] fill-text-secondary font-medium">Data Space</text>

      {/* Axis ticks */}
      {[-2, 0, 2, 4, 6].map((v) => (
        <text key={`tx-${v}`} x={xScale(v)} y={pad.top + h + 14} textAnchor="middle" className="text-[9px] fill-text-tertiary font-mono">{v}</text>
      ))}
      {[-5, 0, 5, 10, 15].map((v) => (
        <text key={`ty-${v}`} x={pad.left - 6} y={yScale(v) + 3} textAnchor="end" className="text-[9px] fill-text-tertiary font-mono">{v}</text>
      ))}

      {/* Click hint */}
      {dataX.length === 0 && (
        <text x={pad.left + w / 2} y={pad.top + h / 2} textAnchor="middle" className="text-[11px] fill-text-tertiary">
          Click to add data points
        </text>
      )}
    </svg>
  )
}

// ── Predictive distribution panel ───────────────────────────────────
function PredictivePanel({
  model,
  hoverX,
  width,
  height,
}: {
  model: BayesianLinearRegression
  hoverX: number | null
  width: number
  height: number
}) {
  const pad = { top: 30, right: 15, bottom: 35, left: 45 }
  const w = width - pad.left - pad.right
  const h = height - pad.top - pad.bottom

  const prediction = useMemo(() => {
    if (hoverX === null || !model.hasData) return null
    return model.predict(hoverX)
  }, [hoverX, model.posteriorMean, model.posteriorCov, model.noiseVariance, model.hasData])

  // Gaussian bell curve
  const curve = useMemo(() => {
    if (!prediction) return null
    const { mean, variance } = prediction
    const std = Math.sqrt(variance)
    const yMin = mean - 4 * std
    const yMax = mean + 4 * std
    const nPts = 80
    const pts: { y: number; density: number }[] = []
    let maxD = 0
    for (let i = 0; i <= nPts; i++) {
      const y = yMin + (yMax - yMin) * (i / nPts)
      const z = (y - mean) / std
      const density = Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI))
      if (density > maxD) maxD = density
      pts.push({ y, density })
    }
    return { pts, maxD, yMin, yMax, mean, std }
  }, [prediction])

  if (!curve) {
    return (
      <svg width={width} height={height} className="block">
        <text x={width / 2} y={16} textAnchor="middle" className="text-[11px] fill-text-secondary font-medium">Predictive Distribution</text>
        <text x={width / 2} y={height / 2} textAnchor="middle" className="text-[11px] fill-text-tertiary">
          {model.hasData ? 'Hover on data space' : 'Add data first'}
        </text>
      </svg>
    )
  }

  const yScale = (v: number) => pad.top + h - ((v - curve.yMin) / (curve.yMax - curve.yMin)) * h
  const dScale = (d: number) => pad.left + (d / curve.maxD) * w

  // Build path
  const linePath = curve.pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${dScale(p.density)},${yScale(p.y)}`).join(' ')
  const fillPath = `M${dScale(0)},${yScale(curve.pts[0].y)} ` + curve.pts.map((p) => `L${dScale(p.density)},${yScale(p.y)}`).join(' ') + ` L${dScale(0)},${yScale(curve.pts[curve.pts.length - 1].y)} Z`

  return (
    <svg width={width} height={height} className="block">
      <text x={width / 2} y={16} textAnchor="middle" className="text-[11px] fill-text-secondary font-medium">Predictive at x = {hoverX?.toFixed(1)}</text>

      {/* Fill */}
      <path d={fillPath} fill={COLORS.predictive} fillOpacity={0.12} />
      {/* Stroke */}
      <path d={linePath} fill="none" stroke={COLORS.predictive} strokeWidth={2} />

      {/* Mean line */}
      <line x1={pad.left} y1={yScale(curve.mean)} x2={pad.left + w} y2={yScale(curve.mean)} stroke={COLORS.predictive} strokeWidth={1} strokeDasharray="4,3" />

      {/* ±1σ markers */}
      {[-1, 1].map((s) => (
        <line
          key={s}
          x1={pad.left}
          y1={yScale(curve.mean + s * curve.std)}
          x2={pad.left + w}
          y2={yScale(curve.mean + s * curve.std)}
          stroke={COLORS.predictive}
          strokeWidth={0.5}
          strokeDasharray="2,3"
          strokeOpacity={0.4}
        />
      ))}

      {/* Labels */}
      <text x={pad.left + w + 2} y={yScale(curve.mean) + 3} className="text-[8px] fill-text-tertiary font-mono">μ={curve.mean.toFixed(1)}</text>
      <text x={pad.left + w + 2} y={yScale(curve.mean + curve.std) + 3} className="text-[8px] fill-text-tertiary font-mono">+σ</text>
      <text x={pad.left + w + 2} y={yScale(curve.mean - curve.std) + 3} className="text-[8px] fill-text-tertiary font-mono">-σ</text>

      {/* Y-axis ticks */}
      {[curve.yMin, curve.mean, curve.yMax].map((v) => (
        <text key={v} x={pad.left - 6} y={yScale(v) + 3} textAnchor="end" className="text-[9px] fill-text-tertiary font-mono">{v.toFixed(0)}</text>
      ))}
      <text x={12} y={pad.top + h / 2} textAnchor="middle" transform={`rotate(-90, 12, ${pad.top + h / 2})`} className="text-[10px] fill-text-tertiary font-mono">y</text>
    </svg>
  )
}

// ── Main Component ──────────────────────────────────────────────────
export function BayesianRegressionViz() {
  const [dataX, setDataX] = useState<number[]>([])
  const [dataY, setDataY] = useState<number[]>([])
  const [nSamples, setNSamples] = useState(30)
  const [priorType, setPriorType] = useState('uninformed')
  const [noiseVar, setNoiseVar] = useState(2.0)
  const [showOLS, setShowOLS] = useState(true)
  const [showBand, setShowBand] = useState(true)
  const [hoverX, setHoverX] = useState<number | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(900)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) setContainerWidth(entry.contentRect.width)
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const prior = getPriorParams(priorType)

  const model = useMemo(() => {
    const m = new BayesianLinearRegression(prior.mean, prior.cov, noiseVar)
    m.fit(dataX, dataY)
    return m
  }, [dataX, dataY, prior.mean, prior.cov, noiseVar])

  const handleAddPoint = useCallback((x: number, y: number) => {
    setDataX((prev) => [...prev, x])
    setDataY((prev) => [...prev, y])
  }, [])

  const handleAddBatch = useCallback(() => {
    const batch = makeBayesianLinearData(10, TRUE_SLOPE, TRUE_INTERCEPT, Math.sqrt(noiseVar), Date.now())
    setDataX((prev) => [...prev, ...batch.x])
    setDataY((prev) => [...prev, ...batch.y])
  }, [noiseVar])

  const handleClear = useCallback(() => {
    setDataX([])
    setDataY([])
    setHoverX(null)
  }, [])

  // Responsive panel sizing
  const isCompact = containerWidth < 700
  const panelWidth = isCompact ? containerWidth - 32 : Math.floor((containerWidth - 48) / 3)
  const panelHeight = isCompact ? 280 : 320

  return (
    <div className="space-y-4" ref={containerRef}>
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Select label="Prior" value={priorType} options={PRIOR_TYPES} onChange={setPriorType} className="w-44" />
          <Slider label="Noise σ²" value={noiseVar} min={0.5} max={5} step={0.5} onChange={setNoiseVar} formatValue={(v) => v.toFixed(1)} className="w-32" />
          <Slider label="Samples" value={nSamples} min={5} max={100} step={5} onChange={setNSamples} formatValue={(v) => String(v)} className="w-32" />

          <div className="h-6 w-px bg-obsidian-border" />

          <Toggle label="OLS" checked={showOLS} onChange={setShowOLS} />
          <Toggle label="95% Band" checked={showBand} onChange={setShowBand} />

          <div className="h-6 w-px bg-obsidian-border" />

          <Button variant="secondary" size="sm" onClick={handleClear}>Clear</Button>
          <Button variant="primary" size="sm" onClick={handleAddBatch}>Add 10 Points</Button>
        </div>
      </GlassCard>

      {/* Triple Panel */}
      <div className={`grid gap-4 ${isCompact ? 'grid-cols-1' : 'grid-cols-3'}`}>
        <GlassCard className="p-2 overflow-hidden">
          <ContourPanel model={model} width={panelWidth} height={panelHeight} />
        </GlassCard>

        <GlassCard className="p-2 overflow-hidden">
          <DataSpacePanel
            model={model}
            dataX={dataX}
            dataY={dataY}
            nSamples={nSamples}
            showOLS={showOLS}
            showBand={showBand}
            width={panelWidth}
            height={panelHeight}
            onAddPoint={handleAddPoint}
            hoverX={hoverX}
            onHoverX={setHoverX}
          />
        </GlassCard>

        <GlassCard className="p-2 overflow-hidden">
          <PredictivePanel model={model} hoverX={hoverX} width={panelWidth} height={panelHeight} />
        </GlassCard>
      </div>

      {/* Callout */}
      <div className="text-[11px] text-text-tertiary max-w-2xl leading-relaxed px-1">
        OLS gives you one line (the MAP when the prior is flat). Bayesian regression gives you the MAP plus every other plausible line, weighted by probability. Try the "Wrong prior" to see how data overwhelms a bad belief.
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 px-1">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5" style={{ backgroundColor: COLORS.prior, opacity: 0.6 }} />
          <span className="text-[10px] text-text-tertiary">Prior</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5" style={{ backgroundColor: COLORS.posterior }} />
          <span className="text-[10px] text-text-tertiary">Posterior / Samples</span>
        </div>
        {showOLS && (
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5" style={{ backgroundColor: COLORS.ols, borderTop: '1px dashed' }} />
            <span className="text-[10px] text-text-tertiary">OLS</span>
          </div>
        )}
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.data, opacity: 0.6 }} />
          <span className="text-[10px] text-text-tertiary">Data ({dataX.length} pts)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5" style={{ backgroundColor: COLORS.predictive }} />
          <span className="text-[10px] text-text-tertiary">Predictive</span>
        </div>
      </div>
    </div>
  )
}
