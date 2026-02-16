import { useState, useMemo, useCallback, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'
import {
  makeLinearData,
  computeLossSurface,
  olsSolution,
} from '../../../lib/algorithms/basics/linearModel'
import { runGradientDescent, type GDVariant, type GDStep } from '../../../lib/algorithms/basics/gradientDescent'

// ── Constants ─────────────────────────────────────────────────────────
const SLOPE_RANGE: [number, number] = [-2, 4]
const INTERCEPT_RANGE: [number, number] = [-5, 10]
const TRUE_SLOPE = 1.8
const TRUE_INTERCEPT = 2.0
const NOISE_STD = 2.5
const N_POINTS = 20
const SURFACE_RES = 50
const MAX_STEPS = 150

const VARIANT_LABELS: Record<GDVariant, string> = {
  batch: 'Batch GD',
  sgd: 'SGD',
  'mini-batch': 'Mini-Batch',
}

const VARIANT_COLORS: Record<GDVariant, string> = {
  batch: '#6366F1',
  sgd: '#F472B6',
  'mini-batch': '#FBBF24',
}

// ── Contour plot with GD path ────────────────────────────────────────
function ContourPanel({
  innerWidth,
  innerHeight,
  surface,
  history,
  currentStep,
  optSlope,
  optIntercept,
  showGradientArrows,
  showPathTrace,
  variant,
}: {
  innerWidth: number
  innerHeight: number
  surface: { slopes: number[]; intercepts: number[]; losses: number[][] }
  history: GDStep[]
  currentStep: number
  optSlope: number
  optIntercept: number
  showGradientArrows: boolean
  showPathTrace: boolean
  variant: GDVariant
}) {
  const { slopes, intercepts, losses } = surface
  const res = slopes.length

  const xScale = d3.scaleLinear().domain([slopes[0], slopes[res - 1]]).range([0, innerWidth])
  const yScale = d3.scaleLinear().domain([intercepts[0], intercepts[res - 1]]).range([innerHeight, 0])

  // Color scale
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

  const step = history[currentStep]
  const pathToShow = showPathTrace ? history.slice(0, currentStep + 1) : []

  // Gradient arrow scaling
  const gradScale = Math.min(innerWidth, innerHeight) * 0.08

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

      {/* Contour lines (log-spaced levels) */}
      {useMemo(() => {
        const logMin = Math.log10(Math.max(minLoss, 0.01))
        const logMax = Math.log10(maxLoss)
        const numLevels = 12
        const levels: number[] = []
        for (let i = 0; i < numLevels; i++) {
          levels.push(Math.pow(10, logMin + (logMax - logMin) * (i / (numLevels - 1))))
        }

        // Build contour data as a flat array (column-major for d3-contour)
        const values: number[] = []
        for (let j = res - 1; j >= 0; j--) {
          for (let i = 0; i < res; i++) {
            values.push(losses[j][i])
          }
        }

        const contours = d3.contours().size([res, res]).thresholds(levels)(values)

        const pathGen = d3.geoPath(
          d3.geoTransform({
            point(px, py) {
              this.stream.point((px / (res - 1)) * innerWidth, (py / (res - 1)) * innerHeight)
            },
          })
        )

        return contours.map((contour, idx) => (
          <path
            key={`contour-${idx}`}
            d={pathGen(contour) || ''}
            fill="none"
            stroke="rgba(255,255,255,0.12)"
            strokeWidth={0.5}
          />
        ))
      }, [losses, minLoss, maxLoss, res, innerWidth, innerHeight])}

      {/* Path trace */}
      {pathToShow.length > 1 && (
        <path
          d={pathToShow
            .map((s, i) => `${i === 0 ? 'M' : 'L'}${xScale(s.slope)},${yScale(s.intercept)}`)
            .join(' ')}
          fill="none"
          stroke={VARIANT_COLORS[variant]}
          strokeWidth={1.5}
          strokeOpacity={0.7}
          strokeLinejoin="round"
        />
      )}

      {/* Path dots */}
      {pathToShow.map((s, i) => (
        <circle
          key={`dot-${i}`}
          cx={xScale(s.slope)}
          cy={yScale(s.intercept)}
          r={i === currentStep ? 0 : 2}
          fill={VARIANT_COLORS[variant]}
          fillOpacity={0.5}
        />
      ))}

      {/* Gradient arrow */}
      {showGradientArrows && step && (
        <line
          x1={xScale(step.slope)}
          y1={yScale(step.intercept)}
          x2={xScale(step.slope) - step.gradient.dSlope * gradScale}
          y2={yScale(step.intercept) + step.gradient.dIntercept * gradScale}
          stroke="#4ADE80"
          strokeWidth={2}
          markerEnd="url(#arrowhead)"
          strokeOpacity={0.8}
        />
      )}

      {/* Arrow marker def */}
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#4ADE80" />
        </marker>
      </defs>

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

      {/* Current position */}
      {step && (
        <motion.circle
          cx={xScale(step.slope)}
          cy={yScale(step.intercept)}
          r={6}
          fill={VARIANT_COLORS[variant]}
          stroke="white"
          strokeWidth={2}
          animate={{
            cx: xScale(step.slope),
            cy: yScale(step.intercept),
          }}
          transition={{ duration: 0.1 }}
        />
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

      {/* Step counter */}
      <rect x={0} y={0} width={80} height={24} rx={4} fill="rgba(0,0,0,0.5)" />
      <text x={40} y={16} textAnchor="middle" className="text-[10px] font-mono" fill="#A1A1AA">
        Step {currentStep}/{history.length - 1}
      </text>
    </>
  )
}

// ── Linked scatter showing current fit ───────────────────────────────
function LinkedScatterPanel({
  innerWidth,
  innerHeight,
  data,
  history,
  currentStep,
}: {
  innerWidth: number
  innerHeight: number
  data: { x: number[]; y: number[] }
  history: GDStep[]
  currentStep: number
}) {
  const step = history[currentStep]
  if (!step) return null

  const { slope, intercept, loss } = step

  const allY = [...data.y, ...data.x.map((xi) => slope * xi + intercept)]
  const xExtent = d3.extent(data.x) as [number, number]
  const yExtent = d3.extent(allY) as [number, number]
  const xPad = (xExtent[1] - xExtent[0]) * 0.1
  const yPad = (yExtent[1] - yExtent[0]) * 0.15

  const xScale = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([0, innerWidth])
  const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([innerHeight, 0])

  const lineX1 = xExtent[0] - xPad
  const lineX2 = xExtent[1] + xPad
  const lineY1 = slope * lineX1 + intercept
  const lineY2 = slope * lineX2 + intercept

  return (
    <>
      {/* Regression line */}
      <motion.line
        x1={xScale(lineX1)}
        y1={yScale(lineY1)}
        x2={xScale(lineX2)}
        y2={yScale(lineY2)}
        stroke="#818CF8"
        strokeWidth={2}
        strokeOpacity={0.9}
        animate={{ y1: yScale(lineY1), y2: yScale(lineY2) }}
        transition={{ duration: 0.05 }}
      />

      {/* Data points */}
      {data.x.map((xi, i) => (
        <circle
          key={`pt-${i}`}
          cx={xScale(xi)}
          cy={yScale(data.y[i])}
          r={3.5}
          fill="#E4E4E7"
          fillOpacity={0.8}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={0.5}
        />
      ))}

      {/* Loss display */}
      <rect x={innerWidth - 120} y={0} width={120} height={44} rx={6} fill="rgba(0,0,0,0.5)" stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} />
      <text x={innerWidth - 60} y={14} textAnchor="middle" className="text-[8px] font-mono uppercase tracking-wider" fill="#71717A">
        MSE Loss
      </text>
      <text
        x={innerWidth - 60}
        y={34}
        textAnchor="middle"
        className="text-[14px] font-mono font-bold"
        fill={loss < 10 ? '#4ADE80' : loss < 30 ? '#FBBF24' : '#F87171'}
      >
        {loss.toFixed(2)}
      </text>
    </>
  )
}

// ── Loss curve over steps ────────────────────────────────────────────
function LossCurvePanel({
  innerWidth,
  innerHeight,
  history,
  currentStep,
  variant,
}: {
  innerWidth: number
  innerHeight: number
  history: GDStep[]
  currentStep: number
  variant: GDVariant
}) {
  const steps = history.slice(0, currentStep + 1)
  if (steps.length < 2) return null

  const xScale = d3.scaleLinear().domain([0, history.length - 1]).range([0, innerWidth])
  const maxLoss = Math.max(...history.map((s) => s.loss))
  const minLoss = Math.min(...history.map((s) => s.loss))
  const yPad = (maxLoss - minLoss) * 0.1 || 1
  const yScale = d3.scaleLinear().domain([Math.max(0, minLoss - yPad), maxLoss + yPad]).range([innerHeight, 0])

  const line = d3
    .line<GDStep>()
    .x((_, i) => xScale(i))
    .y((d) => yScale(d.loss))
    .curve(d3.curveMonotoneX)

  return (
    <>
      {/* Grid lines */}
      {yScale.ticks(4).map((tick) => (
        <line
          key={`grid-${tick}`}
          x1={0}
          y1={yScale(tick)}
          x2={innerWidth}
          y2={yScale(tick)}
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={0.5}
        />
      ))}

      {/* Loss curve */}
      <path
        d={line(steps) || ''}
        fill="none"
        stroke={VARIANT_COLORS[variant]}
        strokeWidth={2}
        strokeOpacity={0.8}
      />

      {/* Current point */}
      {steps.length > 0 && (
        <circle
          cx={xScale(currentStep)}
          cy={yScale(steps[steps.length - 1].loss)}
          r={4}
          fill={VARIANT_COLORS[variant]}
          stroke="white"
          strokeWidth={1.5}
        />
      )}

      {/* Labels */}
      <text x={innerWidth / 2} y={innerHeight + 14} textAnchor="middle" className="text-[9px] font-mono" fill="#71717A">
        Step
      </text>
      <text x={-innerHeight / 2} y={-8} textAnchor="middle" className="text-[9px] font-mono" fill="#71717A" transform="rotate(-90)">
        Loss
      </text>
    </>
  )
}

// ── Learning Rate Comparison (3 side-by-side) ────────────────────────
function LRComparisonPanel({
  data,
  surface,
  optSlope,
  optIntercept,
}: {
  data: { x: number[]; y: number[] }
  surface: { slopes: number[]; intercepts: number[]; losses: number[][] }
  optSlope: number
  optIntercept: number
}) {
  const configs = [
    { lr: 0.001, label: 'Too Small (0.001)', color: '#38BDF8' },
    { lr: 0.03, label: 'Just Right (0.03)', color: '#4ADE80' },
    { lr: 0.12, label: 'Too Large (0.12)', color: '#F87171' },
  ]

  const histories = useMemo(
    () =>
      configs.map((cfg) =>
        runGradientDescent(data.x, data.y, -1, -3, cfg.lr, 60, 'batch')
      ),
    [data]
  )

  return (
    <div className="grid grid-cols-3 gap-3">
      {configs.map((cfg, idx) => (
        <GlassCard key={cfg.lr} className="overflow-hidden">
          <div className="px-3 pt-2 pb-1">
            <p className="text-[10px] font-mono font-medium" style={{ color: cfg.color }}>
              {cfg.label}
            </p>
          </div>
          <SVGContainer
            aspectRatio={1}
            minHeight={140}
            maxHeight={200}
            padding={{ top: 6, right: 6, bottom: 6, left: 6 }}
          >
            {({ innerWidth, innerHeight }) => (
              <MiniContourPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                surface={surface}
                history={histories[idx]}
                color={cfg.color}
                optSlope={optSlope}
                optIntercept={optIntercept}
              />
            )}
          </SVGContainer>
        </GlassCard>
      ))}
    </div>
  )
}

function MiniContourPanel({
  innerWidth,
  innerHeight,
  surface,
  history,
  color,
  optSlope,
  optIntercept,
}: {
  innerWidth: number
  innerHeight: number
  surface: { slopes: number[]; intercepts: number[]; losses: number[][] }
  history: GDStep[]
  color: string
  optSlope: number
  optIntercept: number
}) {
  const { slopes, intercepts, losses } = surface
  const res = slopes.length

  const xScale = d3.scaleLinear().domain([slopes[0], slopes[res - 1]]).range([0, innerWidth])
  const yScale = d3.scaleLinear().domain([intercepts[0], intercepts[res - 1]]).range([innerHeight, 0])

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

  // Clip history to visible range
  const visibleHistory = history.filter(
    (s) =>
      s.slope >= slopes[0] &&
      s.slope <= slopes[res - 1] &&
      s.intercept >= intercepts[0] &&
      s.intercept <= intercepts[res - 1]
  )

  return (
    <>
      {/* Heatmap */}
      {losses.map((row, j) =>
        row.map((loss, i) => (
          <rect
            key={`mc-${j}-${i}`}
            x={i * cellW}
            y={(res - 1 - j) * cellH}
            width={cellW + 0.5}
            height={cellH + 0.5}
            fill={colorScale(loss)}
          />
        ))
      )}

      {/* Optimum */}
      <circle
        cx={xScale(optSlope)}
        cy={yScale(optIntercept)}
        r={3}
        fill="none"
        stroke="#4ADE80"
        strokeWidth={1}
        strokeDasharray="2 2"
      />

      {/* Path */}
      {visibleHistory.length > 1 && (
        <path
          d={visibleHistory
            .map((s, i) => `${i === 0 ? 'M' : 'L'}${xScale(s.slope)},${yScale(s.intercept)}`)
            .join(' ')}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeOpacity={0.8}
          strokeLinejoin="round"
        />
      )}

      {/* End dot */}
      {visibleHistory.length > 0 && (
        <circle
          cx={xScale(visibleHistory[visibleHistory.length - 1].slope)}
          cy={yScale(visibleHistory[visibleHistory.length - 1].intercept)}
          r={4}
          fill={color}
          stroke="white"
          strokeWidth={1.5}
        />
      )}
    </>
  )
}

// ── Variant comparison cards ─────────────────────────────────────────
function VariantComparisonCards({
  data,
}: {
  data: { x: number[]; y: number[] }
}) {
  const variants: { key: GDVariant; desc: string }[] = [
    { key: 'batch', desc: 'Smooth, stable path. Uses all data every step.' },
    { key: 'sgd', desc: 'Noisy, jittery path. Uses 1 random point per step.' },
    { key: 'mini-batch', desc: 'Moderate noise. Uses a small random subset per step.' },
  ]

  const histories = useMemo(
    () =>
      variants.map((v) =>
        runGradientDescent(data.x, data.y, -1, -3, 0.03, 80, v.key, 4, 42)
      ),
    [data]
  )

  return (
    <div className="grid grid-cols-3 gap-3">
      {variants.map((v, idx) => {
        const h = histories[idx]
        const finalLoss = h[h.length - 1].loss
        return (
          <div
            key={v.key}
            className="rounded-lg border p-3 text-center"
            style={{
              backgroundColor: `${VARIANT_COLORS[v.key]}08`,
              borderColor: `${VARIANT_COLORS[v.key]}25`,
            }}
          >
            <p className="text-[10px] font-mono font-medium" style={{ color: VARIANT_COLORS[v.key] }}>
              {VARIANT_LABELS[v.key]}
            </p>
            <p className="text-lg font-mono font-bold mt-1" style={{ color: VARIANT_COLORS[v.key] }}>
              {finalLoss.toFixed(2)}
            </p>
            <p className="text-[9px] text-text-tertiary mt-1">{v.desc}</p>
            <p className="text-[9px] text-text-tertiary mt-0.5">{h.length - 1} steps</p>
          </div>
        )
      })}
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function GradientDescentViz() {
  const [learningRate, setLearningRate] = useState(0.03)
  const [variant, setVariant] = useState<GDVariant>('batch')
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [showGradientArrows, setShowGradientArrows] = useState(true)
  const [showPathTrace, setShowPathTrace] = useState(true)
  const [startSlope, setStartSlope] = useState(-1)
  const [startIntercept, setStartIntercept] = useState(-3)

  const playRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const data = useMemo(
    () => makeLinearData(N_POINTS, TRUE_SLOPE, TRUE_INTERCEPT, NOISE_STD, 42),
    []
  )

  const ols = useMemo(() => olsSolution(data.x, data.y), [data])

  const surface = useMemo(
    () => computeLossSurface(data.x, data.y, SLOPE_RANGE, INTERCEPT_RANGE, SURFACE_RES, 'mse'),
    [data]
  )

  const history = useMemo(
    () => runGradientDescent(data.x, data.y, startSlope, startIntercept, learningRate, MAX_STEPS, variant, 4, 42),
    [data, startSlope, startIntercept, learningRate, variant]
  )

  // Clamp currentStep when history changes
  useEffect(() => {
    if (currentStep >= history.length) {
      setCurrentStep(history.length - 1)
    }
  }, [history, currentStep])

  // Playback
  useEffect(() => {
    if (isPlaying) {
      playRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= history.length - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, 120)
    }
    return () => {
      if (playRef.current) clearInterval(playRef.current)
    }
  }, [isPlaying, history.length])

  const handleReset = useCallback(() => {
    setCurrentStep(0)
    setIsPlaying(false)
  }, [])

  const handleStepForward = useCallback(() => {
    setCurrentStep((prev) => Math.min(prev + 1, history.length - 1))
  }, [history.length])

  const handleStepBack = useCallback(() => {
    setCurrentStep((prev) => Math.max(prev - 1, 0))
  }, [])

  const handleRandomStart = useCallback(() => {
    setStartSlope(Math.random() * 4 - 1)
    setStartIntercept(Math.random() * 12 - 3)
    setCurrentStep(0)
    setIsPlaying(false)
  }, [])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          {/* Playback controls */}
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={handleReset}>
              Reset
            </Button>
            <Button variant="ghost" size="sm" onClick={handleStepBack} disabled={currentStep === 0}>
              ◀
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setIsPlaying(!isPlaying)}
              active={isPlaying}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </Button>
            <Button variant="ghost" size="sm" onClick={handleStepForward} disabled={currentStep >= history.length - 1}>
              ▶
            </Button>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <Slider
            label="Learning Rate"
            value={learningRate}
            min={0.001}
            max={0.15}
            step={0.001}
            onChange={(v) => {
              setLearningRate(v)
              setCurrentStep(0)
              setIsPlaying(false)
            }}
            formatValue={(v) => v.toFixed(3)}
            className="w-44"
          />

          <div className="h-6 w-px bg-obsidian-border" />

          <div className="flex items-center gap-1">
            {(['batch', 'sgd', 'mini-batch'] as GDVariant[]).map((v) => (
              <Button
                key={v}
                variant="secondary"
                size="sm"
                active={variant === v}
                onClick={() => {
                  setVariant(v)
                  setCurrentStep(0)
                  setIsPlaying(false)
                }}
              >
                {VARIANT_LABELS[v]}
              </Button>
            ))}
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <Button variant="ghost" size="sm" onClick={handleRandomStart}>
            Random Start
          </Button>

          <div className="flex items-center gap-3 ml-auto">
            <label className="flex items-center gap-1.5 text-xs text-text-tertiary cursor-pointer">
              <input
                type="checkbox"
                checked={showGradientArrows}
                onChange={(e) => setShowGradientArrows(e.target.checked)}
                className="accent-accent"
              />
              Gradient
            </label>
            <label className="flex items-center gap-1.5 text-xs text-text-tertiary cursor-pointer">
              <input
                type="checkbox"
                checked={showPathTrace}
                onChange={(e) => setShowPathTrace(e.target.checked)}
                className="accent-accent"
              />
              Path
            </label>
          </div>
        </div>
      </GlassCard>

      {/* Main viz: Contour + Linked Scatter side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Contour plot */}
        <GlassCard className="overflow-hidden">
          <div className="px-4 pt-3 pb-1">
            <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary">
              Loss Surface
            </h4>
            <p className="text-[9px] text-text-tertiary mt-0.5">
              Watch gradient descent navigate the parameter space toward the minimum.
            </p>
          </div>
          <SVGContainer
            aspectRatio={1}
            minHeight={280}
            maxHeight={400}
            padding={{ top: 10, right: 16, bottom: 24, left: 24 }}
          >
            {({ innerWidth, innerHeight }) => (
              <ContourPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                surface={surface}
                history={history}
                currentStep={currentStep}
                optSlope={ols.slope}
                optIntercept={ols.intercept}
                showGradientArrows={showGradientArrows}
                showPathTrace={showPathTrace}
                variant={variant}
              />
            )}
          </SVGContainer>
        </GlassCard>

        {/* Linked scatter + loss curve */}
        <div className="space-y-4">
          <GlassCard className="overflow-hidden">
            <div className="px-4 pt-3 pb-1">
              <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary">
                Current Fit
              </h4>
              <p className="text-[9px] text-text-tertiary mt-0.5">
                The regression line updates as gradient descent adjusts the parameters.
              </p>
            </div>
            <SVGContainer
              aspectRatio={16 / 8}
              minHeight={160}
              maxHeight={220}
              padding={{ top: 10, right: 16, bottom: 16, left: 24 }}
            >
              {({ innerWidth, innerHeight }) => (
                <LinkedScatterPanel
                  innerWidth={innerWidth}
                  innerHeight={innerHeight}
                  data={data}
                  history={history}
                  currentStep={currentStep}
                />
              )}
            </SVGContainer>
          </GlassCard>

          <GlassCard className="overflow-hidden">
            <div className="px-4 pt-3 pb-1">
              <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary">
                Loss Over Steps
              </h4>
            </div>
            <SVGContainer
              aspectRatio={16 / 6}
              minHeight={100}
              maxHeight={160}
              padding={{ top: 8, right: 16, bottom: 20, left: 32 }}
            >
              {({ innerWidth, innerHeight }) => (
                <LossCurvePanel
                  innerWidth={innerWidth}
                  innerHeight={innerHeight}
                  history={history}
                  currentStep={currentStep}
                  variant={variant}
                />
              )}
            </SVGContainer>
          </GlassCard>
        </div>
      </div>

      {/* Learning Rate Comparison */}
      <div>
        <p className="text-xs font-medium text-text-secondary mb-2">
          Learning Rate Comparison
        </p>
        <p className="text-[10px] text-text-tertiary mb-3">
          Same starting point, same data. Only the learning rate changes. Too small: barely moves.
          Too large: overshoots and may diverge.
        </p>
        <LRComparisonPanel data={data} surface={surface} optSlope={ols.slope} optIntercept={ols.intercept} />
      </div>

      {/* Variant Comparison */}
      <div>
        <p className="text-xs font-medium text-text-secondary mb-2">
          Variant Comparison
        </p>
        <p className="text-[10px] text-text-tertiary mb-3">
          Same learning rate, same start. Different amounts of data per step change the path character.
        </p>
        <VariantComparisonCards data={data} />
      </div>
    </div>
  )
}
