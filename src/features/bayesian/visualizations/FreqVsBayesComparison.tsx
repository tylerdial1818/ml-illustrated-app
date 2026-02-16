import { useState, useMemo, useRef, useEffect } from 'react'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Toggle } from '../../../components/ui/Toggle'
import { BayesianLinearRegression, makeBayesianLinearData } from '../../../lib/algorithms/bayesian/bayesianLinearRegression'
import { GaussianProcess } from '../../../lib/algorithms/bayesian/gaussianProcess'
import { RBFKernel } from '../../../lib/algorithms/bayesian/kernels'

// ── Colors ──────────────────────────────────────────────────────────
const COLORS = {
  ols: '#FBBF24',
  ridge: '#F472B6',
  bayesianLR: '#6366F1',
  gp: '#34D399',
  data: '#F4F4F5',
  band: (c: string) => `${c}22`,
  gridLine: 'rgba(255,255,255,0.06)',
}

const METHODS = ['ols', 'ridge', 'bayesianLR', 'gp'] as const
type Method = typeof METHODS[number]

const METHOD_LABELS: Record<Method, string> = {
  ols: 'OLS',
  ridge: 'Ridge',
  bayesianLR: 'Bayesian LR',
  gp: 'Gaussian Process',
}

const METHOD_COLORS: Record<Method, string> = {
  ols: COLORS.ols,
  ridge: COLORS.ridge,
  bayesianLR: COLORS.bayesianLR,
  gp: COLORS.gp,
}

function ridgeEstimate(x: number[], y: number[], lambda: number): { slope: number; intercept: number } {
  const n = x.length
  if (n === 0) return { slope: 0, intercept: 0 }
  const mx = x.reduce((a, b) => a + b, 0) / n
  const my = y.reduce((a, b) => a + b, 0) / n
  let num = 0, den = 0
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my)
    den += (x[i] - mx) * (x[i] - mx)
  }
  const slope = (den + lambda) === 0 ? 0 : num / (den + lambda)
  const intercept = my - slope * mx
  return { slope, intercept }
}

// ── Single method panel ─────────────────────────────────────────────
function MethodPanel({
  method,
  dataX,
  dataY,
  width,
  height,
}: {
  method: Method
  dataX: number[]
  dataY: number[]
  width: number
  height: number
}) {
  const pad = { top: 22, right: 10, bottom: 25, left: 35 }
  const w = width - pad.left - pad.right
  const h = height - pad.top - pad.bottom
  const xDomain: [number, number] = [-4, 8]
  const yDomain: [number, number] = [-6, 16]

  const xScale = (v: number) => pad.left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * w
  const yScale = (v: number) => pad.top + h - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * h

  const xRange = useMemo(() => {
    const pts: number[] = []
    for (let i = 0; i <= 60; i++) pts.push(xDomain[0] + (xDomain[1] - xDomain[0]) * (i / 60))
    return pts
  }, [])

  const color = METHOD_COLORS[method]

  // Compute fit
  const fit = useMemo(() => {
    if (dataX.length < 2) return null

    if (method === 'ols') {
      const est = BayesianLinearRegression.olsEstimate(dataX, dataY)
      return { meanPath: xRange.map((x) => est.intercept + est.slope * x), band: null }
    }

    if (method === 'ridge') {
      const est = ridgeEstimate(dataX, dataY, 2.0)
      return { meanPath: xRange.map((x) => est.intercept + est.slope * x), band: null }
    }

    if (method === 'bayesianLR') {
      const blr = new BayesianLinearRegression([0, 0], [[10, 0], [0, 10]], 2.0)
      blr.fit(dataX, dataY)
      const band = blr.getCredibleBand(xRange)
      return { meanPath: band.mean, band }
    }

    if (method === 'gp') {
      const kernel = new RBFKernel(1.5, 1.0)
      const gp = new GaussianProcess(kernel, 0.3)
      gp.fit(dataX, dataY)
      const band = gp.getCredibleBand(xRange)
      return { meanPath: band.mean, band }
    }

    return null
  }, [method, dataX, dataY, xRange])

  const meanLinePath = fit
    ? xRange.map((x, i) => `${i === 0 ? 'M' : 'L'}${xScale(x)},${yScale(fit.meanPath[i])}`).join(' ')
    : ''

  const bandPath = fit?.band
    ? xRange.map((x, i) => `${i === 0 ? 'M' : 'L'}${xScale(x)},${yScale(fit.band!.upper[i])}`)
        .join(' ') + ' ' +
      [...xRange].reverse().map((x, i) => `L${xScale(x)},${yScale(fit.band!.lower[xRange.length - 1 - i])}`)
        .join(' ') + ' Z'
    : null

  return (
    <svg width={width} height={height} className="block">
      {/* Title */}
      <text x={width / 2} y={14} textAnchor="middle" className="text-[10px] fill-text-secondary font-medium" style={{ fill: color }}>{METHOD_LABELS[method]}</text>

      {/* Grid */}
      <line x1={pad.left} y1={yScale(0)} x2={pad.left + w} y2={yScale(0)} stroke={COLORS.gridLine} />

      {/* Band */}
      {bandPath && <path d={bandPath} fill={COLORS.band(color)} />}

      {/* Mean line */}
      {fit && <path d={meanLinePath} fill="none" stroke={color} strokeWidth={2} />}

      {/* Data */}
      {dataX.map((x, i) => (
        <circle key={i} cx={xScale(x)} cy={yScale(dataY[i])} r={2.5} fill={COLORS.data} fillOpacity={0.7} />
      ))}

      {/* No data */}
      {dataX.length < 2 && (
        <text x={width / 2} y={height / 2} textAnchor="middle" className="text-[10px] fill-text-tertiary">Need data</text>
      )}
    </svg>
  )
}

// ── Main Component ──────────────────────────────────────────────────
export function FreqVsBayesComparison() {
  const [nPoints, setNPoints] = useState(15)
  const [showAll, setShowAll] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(800)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) setContainerWidth(entry.contentRect.width)
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const data = useMemo(
    () => makeBayesianLinearData(nPoints, 1.5, 2.0, 1.5, 42),
    [nPoints]
  )

  const methods = showAll ? METHODS : (['ols', 'gp'] as const)
  const panelW = Math.floor((containerWidth - 32 - (methods.length - 1) * 8) / methods.length)
  const panelH = 220

  return (
    <div className="space-y-4" ref={containerRef}>
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Slider label="Data points" value={nPoints} min={5} max={200} step={5} onChange={setNPoints} formatValue={(v) => String(v)} className="w-40" />
          <Toggle label="All methods" checked={showAll} onChange={setShowAll} />
        </div>
      </GlassCard>

      <div className="flex gap-2 overflow-x-auto">
        {methods.map((method) => (
          <GlassCard key={method} className="p-1 flex-shrink-0">
            <MethodPanel method={method} dataX={data.x} dataY={data.y} width={panelW} height={panelH} />
          </GlassCard>
        ))}
      </div>

      <div className="text-[11px] text-text-tertiary max-w-2xl leading-relaxed px-1">
        {nPoints <= 10 && 'With few data points, Bayesian methods show wide bands reflecting genuine uncertainty. OLS shows the same confidence regardless of sample size.'}
        {nPoints > 10 && nPoints <= 50 && 'At moderate sample sizes, the Bayesian and frequentist estimates converge, but the GP still captures non-linear patterns the linear models miss.'}
        {nPoints > 50 && 'With abundant data, all methods converge. The GP uncertainty band tightens to near zero. The prior becomes irrelevant.'}
      </div>
    </div>
  )
}
