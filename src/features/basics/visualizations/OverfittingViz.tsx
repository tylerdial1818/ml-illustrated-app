import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'
import { Toggle } from '../../../components/ui/Toggle'
import {
  makeNoisyCurve,
  fitPolynomial,
  predictPolynomial,
  computeMSE,
  computeLossCurves,
  trainTestSplit,
} from '../../../lib/algorithms/basics/polynomialFit'

// ── Constants ─────────────────────────────────────────────────────────
const N_POINTS = 30
const X_RANGE: [number, number] = [-3, 3]
const TRUE_FN = (x: number) => Math.sin(x * 1.2) * 2 + 0.3 * x
const MAX_DEGREE = 15
const CURVE_SAMPLES = 200

const COLORS = {
  train: '#6366F1',
  test: '#FBBF24',
  curve: '#818CF8',
  trueFn: '#4ADE80',
}

// ── Scatter + Curve Panel ────────────────────────────────────────────
function ScatterCurvePanel({
  innerWidth,
  innerHeight,
  trainX,
  trainY,
  testX,
  testY,
  degree,
  showTestPoints,
}: {
  innerWidth: number
  innerHeight: number
  trainX: number[]
  trainY: number[]
  testX: number[]
  testY: number[]
  degree: number
  showTestPoints: boolean
}) {
  const coeffs = useMemo(() => fitPolynomial(trainX, trainY, degree), [trainX, trainY, degree])

  // Sample the fitted curve
  const curvePoints = useMemo(() => {
    const xs: number[] = []
    for (let i = 0; i < CURVE_SAMPLES; i++) {
      xs.push(X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * (i / (CURVE_SAMPLES - 1)))
    }
    const ys = predictPolynomial(xs, coeffs)
    return xs.map((x, i) => ({ x, y: ys[i] }))
  }, [coeffs])

  // True function curve
  const trueCurve = useMemo(() => {
    const xs: number[] = []
    for (let i = 0; i < CURVE_SAMPLES; i++) {
      xs.push(X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * (i / (CURVE_SAMPLES - 1)))
    }
    return xs.map((x) => ({ x, y: TRUE_FN(x) }))
  }, [])

  // Compute losses
  const trainPred = predictPolynomial(trainX, coeffs)
  const testPred = predictPolynomial(testX, coeffs)
  const trainLoss = computeMSE(trainY, trainPred)
  const testLoss = computeMSE(testY, testPred)

  // Scales
  const allY = [
    ...trainY,
    ...testY,
    ...curvePoints.map((p) => p.y).filter((y) => Math.abs(y) < 20),
  ]
  const xPad = 0.3
  const yExtent = d3.extent(allY) as [number, number]
  const yPad = (yExtent[1] - yExtent[0]) * 0.15

  const xScale = d3.scaleLinear().domain([X_RANGE[0] - xPad, X_RANGE[1] + xPad]).range([0, innerWidth])
  const yScale = d3
    .scaleLinear()
    .domain([yExtent[0] - yPad, yExtent[1] + yPad])
    .range([innerHeight, 0])

  // Clip curve to reasonable y range
  const yMin = yExtent[0] - yPad * 2
  const yMax = yExtent[1] + yPad * 2

  const line = d3
    .line<{ x: number; y: number }>()
    .x((d) => xScale(d.x))
    .y((d) => yScale(Math.max(yMin, Math.min(yMax, d.y))))
    .curve(d3.curveMonotoneX)

  return (
    <>
      {/* True function */}
      <path
        d={line(trueCurve) || ''}
        fill="none"
        stroke={COLORS.trueFn}
        strokeWidth={1}
        strokeDasharray="4 3"
        strokeOpacity={0.4}
      />

      {/* Fitted curve */}
      <path
        d={line(curvePoints) || ''}
        fill="none"
        stroke={COLORS.curve}
        strokeWidth={2}
        strokeOpacity={0.9}
      />

      {/* Test points */}
      {showTestPoints &&
        testX.map((xi, i) => (
          <circle
            key={`test-${i}`}
            cx={xScale(xi)}
            cy={yScale(testY[i])}
            r={4}
            fill={COLORS.test}
            fillOpacity={0.8}
            stroke="rgba(255,255,255,0.2)"
            strokeWidth={0.5}
          />
        ))}

      {/* Train points */}
      {trainX.map((xi, i) => (
        <circle
          key={`train-${i}`}
          cx={xScale(xi)}
          cy={yScale(trainY[i])}
          r={4}
          fill={COLORS.train}
          fillOpacity={0.9}
          stroke="rgba(255,255,255,0.2)"
          strokeWidth={0.5}
        />
      ))}

      {/* Loss display */}
      <rect x={0} y={0} width={160} height={50} rx={6} fill="rgba(0,0,0,0.5)" stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} />
      <circle cx={12} cy={14} r={4} fill={COLORS.train} />
      <text x={22} y={18} className="text-[10px] font-mono" fill="#A1A1AA">
        Train: <tspan fill={COLORS.train} fontWeight="bold">{trainLoss.toFixed(2)}</tspan>
      </text>
      {showTestPoints && (
        <>
          <circle cx={12} cy={34} r={4} fill={COLORS.test} />
          <text x={22} y={38} className="text-[10px] font-mono" fill="#A1A1AA">
            Test: <tspan fill={COLORS.test} fontWeight="bold">{testLoss.toFixed(2)}</tspan>
          </text>
        </>
      )}
    </>
  )
}

// ── Dual Loss Curve Panel ────────────────────────────────────────────
function DualLossCurvePanel({
  innerWidth,
  innerHeight,
  lossCurves,
  currentDegree,
}: {
  innerWidth: number
  innerHeight: number
  lossCurves: { degree: number; trainLoss: number; testLoss: number }[]
  currentDegree: number
}) {
  // Find optimal degree (min test loss)
  let optimalDegree = 1
  let minTestLoss = Infinity
  for (const lc of lossCurves) {
    if (lc.testLoss < minTestLoss) {
      minTestLoss = lc.testLoss
      optimalDegree = lc.degree
    }
  }

  // Scales
  const xScale = d3.scaleLinear().domain([1, MAX_DEGREE]).range([0, innerWidth])

  // Cap extreme loss values for display
  const cappedCurves = lossCurves.map((lc) => ({
    ...lc,
    trainLoss: Math.min(lc.trainLoss, 100),
    testLoss: Math.min(lc.testLoss, 100),
  }))

  const maxLoss = Math.max(
    ...cappedCurves.map((lc) => Math.max(lc.trainLoss, lc.testLoss))
  )
  const yScale = d3.scaleLinear().domain([0, maxLoss * 1.1]).range([innerHeight, 0])

  const trainLine = d3
    .line<{ degree: number; trainLoss: number }>()
    .x((d) => xScale(d.degree))
    .y((d) => yScale(d.trainLoss))
    .curve(d3.curveMonotoneX)

  const testLine = d3
    .line<{ degree: number; testLoss: number }>()
    .x((d) => xScale(d.degree))
    .y((d) => yScale(d.testLoss))
    .curve(d3.curveMonotoneX)

  // Current degree data
  const currentData = cappedCurves.find((lc) => lc.degree === currentDegree)

  // Region annotations
  const regions = [
    { label: 'Underfitting', x1: 1, x2: optimalDegree - 0.5, color: '#38BDF8' },
    { label: 'Sweet spot', x1: optimalDegree - 0.5, x2: optimalDegree + 1.5, color: '#4ADE80' },
    { label: 'Overfitting', x1: optimalDegree + 1.5, x2: MAX_DEGREE, color: '#F87171' },
  ]

  return (
    <>
      {/* Region backgrounds */}
      {regions.map((r) => (
        <rect
          key={r.label}
          x={xScale(Math.max(r.x1, 1))}
          y={0}
          width={xScale(Math.min(r.x2, MAX_DEGREE)) - xScale(Math.max(r.x1, 1))}
          height={innerHeight}
          fill={r.color}
          fillOpacity={0.04}
        />
      ))}

      {/* Region labels */}
      {regions.map((r) => (
        <text
          key={`label-${r.label}`}
          x={(xScale(Math.max(r.x1, 1)) + xScale(Math.min(r.x2, MAX_DEGREE))) / 2}
          y={12}
          textAnchor="middle"
          className="text-[8px] font-mono uppercase tracking-wider"
          fill={r.color}
          fillOpacity={0.6}
        >
          {r.label}
        </text>
      ))}

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

      {/* Optimal degree line */}
      <line
        x1={xScale(optimalDegree)}
        y1={0}
        x2={xScale(optimalDegree)}
        y2={innerHeight}
        stroke="#4ADE80"
        strokeWidth={1}
        strokeDasharray="4 3"
        strokeOpacity={0.5}
      />

      {/* Train loss curve */}
      <path d={trainLine(cappedCurves) || ''} fill="none" stroke={COLORS.train} strokeWidth={2} strokeOpacity={0.8} />

      {/* Test loss curve */}
      <path d={testLine(cappedCurves) || ''} fill="none" stroke={COLORS.test} strokeWidth={2} strokeOpacity={0.8} />

      {/* Current degree marker */}
      {currentData && (
        <>
          <line
            x1={xScale(currentDegree)}
            y1={0}
            x2={xScale(currentDegree)}
            y2={innerHeight}
            stroke="rgba(255,255,255,0.15)"
            strokeWidth={1}
            strokeDasharray="2 2"
          />
          <circle
            cx={xScale(currentDegree)}
            cy={yScale(currentData.trainLoss)}
            r={5}
            fill={COLORS.train}
            stroke="white"
            strokeWidth={1.5}
          />
          <circle
            cx={xScale(currentDegree)}
            cy={yScale(currentData.testLoss)}
            r={5}
            fill={COLORS.test}
            stroke="white"
            strokeWidth={1.5}
          />
        </>
      )}

      {/* Axis labels */}
      <text x={innerWidth / 2} y={innerHeight + 16} textAnchor="middle" className="text-[9px] font-mono" fill="#71717A">
        Polynomial Degree
      </text>
      <text x={-innerHeight / 2} y={-10} textAnchor="middle" className="text-[9px] font-mono" fill="#71717A" transform="rotate(-90)">
        Loss (MSE)
      </text>

      {/* Legend */}
      <rect x={innerWidth - 110} y={innerHeight - 40} width={110} height={36} rx={4} fill="rgba(0,0,0,0.4)" />
      <circle cx={innerWidth - 96} cy={innerHeight - 28} r={3} fill={COLORS.train} />
      <text x={innerWidth - 88} y={innerHeight - 24} className="text-[9px] font-mono" fill="#A1A1AA">
        Training loss
      </text>
      <circle cx={innerWidth - 96} cy={innerHeight - 14} r={3} fill={COLORS.test} />
      <text x={innerWidth - 88} y={innerHeight - 10} className="text-[9px] font-mono" fill="#A1A1AA">
        Test loss
      </text>

      {/* Degree tick marks */}
      {[1, 3, 5, 7, 10, 15].map((d) => (
        <text
          key={`tick-${d}`}
          x={xScale(d)}
          y={innerHeight + 10}
          textAnchor="middle"
          className="text-[8px] font-mono"
          fill="#52525B"
        >
          {d}
        </text>
      ))}
    </>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function OverfittingViz() {
  const [degree, setDegree] = useState(3)
  const [noiseStd, setNoiseStd] = useState(1.0)
  const [trainRatio, setTrainRatio] = useState(0.67)
  const [showTestPoints, setShowTestPoints] = useState(true)
  const [showLossCurve, setShowLossCurve] = useState(true)
  const [seed, setSeed] = useState(42)

  const rawData = useMemo(
    () => makeNoisyCurve(N_POINTS, TRUE_FN, noiseStd, X_RANGE, seed),
    [noiseStd, seed]
  )

  const { trainX, trainY, testX, testY } = useMemo(
    () => trainTestSplit(rawData.x, rawData.y, trainRatio, seed + 1),
    [rawData, trainRatio, seed]
  )

  const lossCurves = useMemo(
    () => computeLossCurves(trainX, trainY, testX, testY, MAX_DEGREE),
    [trainX, trainY, testX, testY]
  )

  const handleRandomize = useCallback(() => {
    setSeed((prev) => prev + 7)
  }, [])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Slider
            label="Degree"
            value={degree}
            min={1}
            max={MAX_DEGREE}
            step={1}
            onChange={setDegree}
            formatValue={(v) => String(v)}
            className="w-40"
          />
          <Slider
            label="Noise"
            value={noiseStd}
            min={0.2}
            max={2.5}
            step={0.1}
            onChange={setNoiseStd}
            formatValue={(v) => v.toFixed(1)}
            className="w-32"
          />
          <Slider
            label="Train %"
            value={trainRatio}
            min={0.5}
            max={0.9}
            step={0.05}
            onChange={setTrainRatio}
            formatValue={(v) => `${Math.round(v * 100)}%`}
            className="w-32"
          />

          <div className="h-6 w-px bg-obsidian-border" />

          <Button variant="ghost" size="sm" onClick={handleRandomize}>
            Randomize Data
          </Button>

          <div className="h-6 w-px bg-obsidian-border" />

          <Toggle label="Test Points" checked={showTestPoints} onChange={setShowTestPoints} />
          <Toggle label="Loss Curves" checked={showLossCurve} onChange={setShowLossCurve} />
        </div>
      </GlassCard>

      {/* Main scatter + curve */}
      <GlassCard className="overflow-hidden">
        <div className="px-4 pt-3 pb-1">
          <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary">
            Polynomial Fit (Degree {degree})
          </h4>
          <p className="text-[9px] text-text-tertiary mt-0.5">
            {degree <= 2
              ? 'Too simple. The model misses the underlying pattern.'
              : degree <= 5
                ? 'Good fit. The curve follows the trend without chasing noise.'
                : degree <= 9
                  ? 'Getting wiggly. The curve starts fitting noise in the training data.'
                  : 'Overfitting. The curve passes through training points but swings wildly between them.'}
          </p>
        </div>
        <SVGContainer
          aspectRatio={16 / 7}
          minHeight={240}
          maxHeight={360}
          padding={{ top: 15, right: 20, bottom: 20, left: 30 }}
        >
          {({ innerWidth, innerHeight }) => (
            <ScatterCurvePanel
              innerWidth={innerWidth}
              innerHeight={innerHeight}
              trainX={trainX}
              trainY={trainY}
              testX={testX}
              testY={testY}
              degree={degree}
              showTestPoints={showTestPoints}
            />
          )}
        </SVGContainer>
      </GlassCard>

      {/* Dual loss curve */}
      {showLossCurve && (
        <GlassCard className="overflow-hidden">
          <div className="px-4 pt-3 pb-1">
            <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary">
              Train vs. Test Loss
            </h4>
            <p className="text-[9px] text-text-tertiary mt-0.5">
              Training loss always decreases with more complexity. Test loss reveals when the model starts memorizing noise.
            </p>
          </div>
          <SVGContainer
            aspectRatio={16 / 7}
            minHeight={200}
            maxHeight={300}
            padding={{ top: 20, right: 20, bottom: 24, left: 36 }}
          >
            {({ innerWidth, innerHeight }) => (
              <DualLossCurvePanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                lossCurves={lossCurves}
                currentDegree={degree}
              />
            )}
          </SVGContainer>
        </GlassCard>
      )}

      {/* Bias-Variance callout */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div
          className="rounded-lg border p-3"
          style={{ backgroundColor: '#38BDF808', borderColor: '#38BDF825' }}
        >
          <p className="text-[10px] font-mono font-medium text-[#38BDF8]">Underfitting</p>
          <p className="text-[10px] font-mono font-bold text-[#38BDF8] mt-1">High Bias</p>
          <p className="text-[9px] text-text-tertiary mt-1">
            The model is too simple to capture the pattern. Both train and test error are high.
          </p>
        </div>
        <div
          className="rounded-lg border p-3"
          style={{ backgroundColor: '#4ADE8008', borderColor: '#4ADE8025' }}
        >
          <p className="text-[10px] font-mono font-medium text-[#4ADE80]">Sweet Spot</p>
          <p className="text-[10px] font-mono font-bold text-[#4ADE80] mt-1">Balanced</p>
          <p className="text-[9px] text-text-tertiary mt-1">
            The model captures the real pattern without fitting noise. Test error is minimized.
          </p>
        </div>
        <div
          className="rounded-lg border p-3"
          style={{ backgroundColor: '#F8717108', borderColor: '#F8717125' }}
        >
          <p className="text-[10px] font-mono font-medium text-[#F87171]">Overfitting</p>
          <p className="text-[10px] font-mono font-bold text-[#F87171] mt-1">High Variance</p>
          <p className="text-[9px] text-text-tertiary mt-1">
            The model memorized the training data. Train error is low but test error is high.
          </p>
        </div>
      </div>
    </div>
  )
}
