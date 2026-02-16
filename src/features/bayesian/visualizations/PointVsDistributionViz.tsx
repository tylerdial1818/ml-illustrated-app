import { useState, useMemo, useEffect } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'

// ── Seeded random ─────────────────────────────────────────────────────
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

function gaussianRandom(rng: () => number): number {
  const u1 = rng()
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// ── Data generation ──────────────────────────────────────────────────
const TRUE_SLOPE = 0.8
const TRUE_INTERCEPT = 1.0
const NOISE_STD = 1.2
const X_RANGE: [number, number] = [-3, 5]

function generateData(n: number, seed = 42) {
  const rng = seededRandom(seed)
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const xi = X_RANGE[0] + rng() * (X_RANGE[1] - X_RANGE[0])
    const yi = TRUE_SLOPE * xi + TRUE_INTERCEPT + gaussianRandom(rng) * NOISE_STD
    x.push(xi)
    y.push(yi)
  }
  return { x, y }
}

// OLS fit
function olsFit(x: number[], y: number[]) {
  const n = x.length
  const mx = x.reduce((a, b) => a + b, 0) / n
  const my = y.reduce((a, b) => a + b, 0) / n
  let num = 0, den = 0
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my)
    den += (x[i] - mx) * (x[i] - mx)
  }
  const slope = den === 0 ? 0 : num / den
  const intercept = my - slope * mx
  return { slope, intercept }
}

// Sample posterior lines (simplified Bayesian linear regression)
function samplePosteriorLines(
  x: number[],
  y: number[],
  nSamples: number,
  seed = 123
) {
  const ols = olsFit(x, y)
  const n = x.length
  const rng = seededRandom(seed)

  // Estimate parameter uncertainty from data
  const residuals = y.map((yi, i) => yi - (ols.slope * x[i] + ols.intercept))
  const sigmaResid = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / Math.max(n - 2, 1))

  const mx = x.reduce((a, b) => a + b, 0) / n
  const sxx = x.reduce((s, xi) => s + (xi - mx) ** 2, 0)
  const seSlopeBase = sigmaResid / Math.sqrt(Math.max(sxx, 0.01))
  const seInterceptBase = sigmaResid * Math.sqrt(1 / n + (mx * mx) / Math.max(sxx, 0.01))

  // Scale up uncertainty slightly for visual clarity
  const seSlope = seSlopeBase * 1.5
  const seIntercept = seInterceptBase * 1.5

  const lines: { slope: number; intercept: number }[] = []
  for (let i = 0; i < nSamples; i++) {
    lines.push({
      slope: ols.slope + gaussianRandom(rng) * seSlope,
      intercept: ols.intercept + gaussianRandom(rng) * seIntercept,
    })
  }

  return { lines, ols, seSlope, seIntercept, sigmaResid }
}

// Compute credible band
function credibleBand(
  xGrid: number[],
  lines: { slope: number; intercept: number }[],
  level = 0.95
): { lower: number[]; upper: number[] } {
  const tail = (1 - level) / 2
  const lower: number[] = []
  const upper: number[] = []

  for (const xv of xGrid) {
    const preds = lines.map((l) => l.slope * xv + l.intercept).sort((a, b) => a - b)
    lower.push(preds[Math.floor(preds.length * tail)])
    upper.push(preds[Math.floor(preds.length * (1 - tail))])
  }

  return { lower, upper }
}

// ── Bayesian colors ──────────────────────────────────────────────────
const BAY = {
  prior: '#A1A1AA',
  likelihood: '#FBBF24',
  posterior: '#6366F1',
  posteriorBand: 'rgba(99, 102, 241, 0.15)',
  predictive: '#34D399',
  data: '#F4F4F5',
}

// ── Frequentist Panel ────────────────────────────────────────────────
function FrequentistPanel({
  innerWidth,
  innerHeight,
  data,
  ols,
  animPhase,
}: {
  innerWidth: number
  innerHeight: number
  data: { x: number[]; y: number[] }
  ols: { slope: number; intercept: number }
  animPhase: number
}) {
  const xScale = d3.scaleLinear().domain([X_RANGE[0] - 0.5, X_RANGE[1] + 0.5]).range([0, innerWidth])
  const allY = data.y
  const yExtent = d3.extent(allY) as [number, number]
  const yPad = (yExtent[1] - yExtent[0]) * 0.3
  const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([innerHeight, 0])

  const lineX1 = X_RANGE[0] - 1
  const lineX2 = X_RANGE[1] + 1

  return (
    <>
      {/* Data points */}
      {data.x.map((xi, i) => (
        <circle
          key={`fpt-${i}`}
          cx={xScale(xi)}
          cy={yScale(data.y[i])}
          r={3.5}
          fill={BAY.data}
          fillOpacity={0.8}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={0.5}
        />
      ))}

      {/* OLS line (appears after phase 1) */}
      {animPhase >= 1 && (
        <motion.line
          x1={xScale(lineX1)}
          y1={yScale(ols.slope * lineX1 + ols.intercept)}
          x2={xScale(lineX2)}
          y2={yScale(ols.slope * lineX2 + ols.intercept)}
          stroke={BAY.likelihood}
          strokeWidth={2.5}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
        />
      )}

      {/* Label */}
      <rect x={0} y={0} width={innerWidth} height={22} rx={0} fill="rgba(0,0,0,0.3)" />
      <text x={innerWidth / 2} y={15} textAnchor="middle" className="text-[10px] font-mono font-medium" fill="#E4E4E7">
        Frequentist: one best line
      </text>
    </>
  )
}

// ── Bayesian Panel ───────────────────────────────────────────────────
function BayesianPanel({
  innerWidth,
  innerHeight,
  data,
  lines,
  ols,
  animPhase,
}: {
  innerWidth: number
  innerHeight: number
  data: { x: number[]; y: number[] }
  lines: { slope: number; intercept: number }[]
  ols: { slope: number; intercept: number }
  animPhase: number
}) {
  const xScale = d3.scaleLinear().domain([X_RANGE[0] - 0.5, X_RANGE[1] + 0.5]).range([0, innerWidth])
  const allY = data.y
  const yExtent = d3.extent(allY) as [number, number]
  const yPad = (yExtent[1] - yExtent[0]) * 0.3
  const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([innerHeight, 0])

  const lineX1 = X_RANGE[0] - 1
  const lineX2 = X_RANGE[1] + 1

  // X grid for credible band
  const xGrid = useMemo(() => {
    const xs: number[] = []
    for (let i = 0; i < 100; i++) {
      xs.push(X_RANGE[0] - 0.5 + (X_RANGE[1] - X_RANGE[0] + 1) * (i / 99))
    }
    return xs
  }, [])

  const band = useMemo(
    () => credibleBand(xGrid, lines, 0.95),
    [xGrid, lines]
  )

  // Band area path
  const areaPath = useMemo(() => {
    const upper = xGrid.map((xv, i) => `${xScale(xv)},${yScale(band.upper[i])}`).join(' L')
    const lower = xGrid
      .map((xv, i) => `${xScale(xv)},${yScale(band.lower[i])}`)
      .reverse()
      .join(' L')
    return `M${upper} L${lower} Z`
  }, [xGrid, band, xScale, yScale])

  // How many sample lines to show based on animation phase
  const nLinesToShow = animPhase >= 2 ? Math.min(lines.length, 25) : 0
  const showBand = animPhase >= 3

  return (
    <>
      {/* Credible band */}
      {showBand && (
        <motion.path
          d={areaPath}
          fill={BAY.posteriorBand}
          stroke={BAY.posterior}
          strokeWidth={0.5}
          strokeOpacity={0.3}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        />
      )}

      {/* Sample lines */}
      <AnimatePresence>
        {lines.slice(0, nLinesToShow).map((line, i) => (
          <motion.line
            key={`bline-${i}`}
            x1={xScale(lineX1)}
            y1={yScale(line.slope * lineX1 + line.intercept)}
            x2={xScale(lineX2)}
            y2={yScale(line.slope * lineX2 + line.intercept)}
            stroke={BAY.posterior}
            strokeWidth={1}
            strokeOpacity={0.15}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.15 }}
            transition={{ delay: i * 0.04, duration: 0.3 }}
          />
        ))}
      </AnimatePresence>

      {/* Data points */}
      {data.x.map((xi, i) => (
        <circle
          key={`bpt-${i}`}
          cx={xScale(xi)}
          cy={yScale(data.y[i])}
          r={3.5}
          fill={BAY.data}
          fillOpacity={0.8}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={0.5}
        />
      ))}

      {/* Mean line */}
      {animPhase >= 2 && (
        <motion.line
          x1={xScale(lineX1)}
          y1={yScale(ols.slope * lineX1 + ols.intercept)}
          x2={xScale(lineX2)}
          y2={yScale(ols.slope * lineX2 + ols.intercept)}
          stroke={BAY.posterior}
          strokeWidth={2}
          strokeOpacity={0.8}
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.8 }}
          transition={{ duration: 0.4 }}
        />
      )}

      {/* Label */}
      <rect x={0} y={0} width={innerWidth} height={22} rx={0} fill="rgba(0,0,0,0.3)" />
      <text x={innerWidth / 2} y={15} textAnchor="middle" className="text-[10px] font-mono font-medium" fill="#E4E4E7">
        Bayesian: a distribution over lines
      </text>
    </>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function PointVsDistributionViz() {
  const [nPoints, setNPoints] = useState(30)
  const [animPhase, setAnimPhase] = useState(0)
  const data = useMemo(() => generateData(nPoints, 42), [nPoints])
  const { lines, ols } = useMemo(
    () => samplePosteriorLines(data.x, data.y, 50, 123),
    [data]
  )

  // Auto-animate phases on mount
  useEffect(() => {
    setAnimPhase(0)
    const timers: ReturnType<typeof setTimeout>[] = []
    timers.push(setTimeout(() => setAnimPhase(1), 600))
    timers.push(setTimeout(() => setAnimPhase(2), 1200))
    timers.push(setTimeout(() => setAnimPhase(3), 2200))
    return () => timers.forEach(clearTimeout)
  }, [nPoints])

  return (
    <div className="space-y-4">
      <GlassCard className="p-4">
        <div className="flex items-end gap-4">
          <Slider
            label="Data Points"
            value={nPoints}
            min={5}
            max={80}
            step={1}
            onChange={setNPoints}
            formatValue={(v) => String(v)}
            className="w-48"
          />
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <GlassCard className="overflow-hidden">
          <SVGContainer
            aspectRatio={4 / 3}
            minHeight={250}
            maxHeight={380}
            padding={{ top: 26, right: 16, bottom: 16, left: 24 }}
          >
            {({ innerWidth, innerHeight }) => (
              <FrequentistPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                data={data}
                ols={ols}
                animPhase={animPhase}
              />
            )}
          </SVGContainer>
        </GlassCard>

        <GlassCard className="overflow-hidden">
          <SVGContainer
            aspectRatio={4 / 3}
            minHeight={250}
            maxHeight={380}
            padding={{ top: 26, right: 16, bottom: 16, left: 24 }}
          >
            {({ innerWidth, innerHeight }) => (
              <BayesianPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                data={data}
                lines={lines}
                ols={ols}
                animPhase={animPhase}
              />
            )}
          </SVGContainer>
        </GlassCard>
      </div>
    </div>
  )
}
