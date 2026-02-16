import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'
import { BetaBinomial, simulateCoinFlips } from '../../../lib/algorithms/bayesian/betaBinomial'

// ── Colors ────────────────────────────────────────────────────────────
const BAY = {
  prior: '#A1A1AA',
  likelihood: '#FBBF24',
  posterior: '#6366F1',
  posteriorFill: 'rgba(99, 102, 241, 0.2)',
  credible: 'rgba(99, 102, 241, 0.35)',
}

const PRIOR_PRESETS: { label: string; alpha: number; beta: number }[] = [
  { label: 'Uninformed', alpha: 1, beta: 1 },
  { label: 'Weakly Fair', alpha: 5, beta: 5 },
  { label: 'Strongly Fair', alpha: 50, beta: 50 },
]

// ── Distribution curve panel ─────────────────────────────────────────
function DistributionPanel({
  innerWidth,
  innerHeight,
  curves,
  credibleInterval,
  label,
}: {
  innerWidth: number
  innerHeight: number
  curves: { x: number[]; y: number[]; color: string; dashed?: boolean; filled?: boolean; label: string }[]
  credibleInterval?: [number, number]
  label: string
}) {
  const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerWidth])

  // Find max y across all curves for consistent scale
  let maxY = 0
  for (const curve of curves) {
    for (const y of curve.y) {
      if (y > maxY && isFinite(y)) maxY = y
    }
  }
  maxY = Math.max(maxY * 1.1, 0.5)
  const yScale = d3.scaleLinear().domain([0, maxY]).range([innerHeight, 0])

  // Build area path for filled curves
  function areaPath(curve: { x: number[]; y: number[] }) {
    const top = curve.x
      .map((xi, i) => `${i === 0 ? 'M' : 'L'}${xScale(xi)},${yScale(Math.min(curve.y[i], maxY))}`)
      .join(' ')
    return `${top} L${xScale(curve.x[curve.x.length - 1])},${yScale(0)} L${xScale(curve.x[0])},${yScale(0)} Z`
  }

  function linePath(curve: { x: number[]; y: number[] }) {
    return curve.x
      .map((xi, i) => `${i === 0 ? 'M' : 'L'}${xScale(xi)},${yScale(Math.min(curve.y[i], maxY))}`)
      .join(' ')
  }

  return (
    <>
      {/* X axis ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={`tick-${tick}`}>
          <line
            x1={xScale(tick)}
            y1={innerHeight}
            x2={xScale(tick)}
            y2={innerHeight + 4}
            stroke="rgba(255,255,255,0.2)"
            strokeWidth={0.5}
          />
          <text x={xScale(tick)} y={innerHeight + 14} textAnchor="middle" className="text-[8px] font-mono" fill="#52525B">
            {tick}
          </text>
        </g>
      ))}

      {/* Credible interval */}
      {credibleInterval && (
        <rect
          x={xScale(credibleInterval[0])}
          y={0}
          width={xScale(credibleInterval[1]) - xScale(credibleInterval[0])}
          height={innerHeight}
          fill={BAY.credible}
          rx={2}
        />
      )}

      {/* Curves */}
      {curves.map((curve, idx) => (
        <g key={`curve-${idx}`}>
          {/* Fill */}
          {curve.filled && (
            <motion.path
              d={areaPath(curve)}
              fill={`${curve.color}20`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            />
          )}
          {/* Line */}
          <motion.path
            d={linePath(curve)}
            fill="none"
            stroke={curve.color}
            strokeWidth={curve.dashed ? 1.5 : 2}
            strokeDasharray={curve.dashed ? '4 3' : undefined}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          />
        </g>
      ))}

      {/* Credible interval labels */}
      {credibleInterval && (
        <>
          <text x={xScale(credibleInterval[0])} y={-4} textAnchor="middle" className="text-[7px] font-mono" fill={BAY.posterior}>
            {credibleInterval[0].toFixed(2)}
          </text>
          <text x={xScale(credibleInterval[1])} y={-4} textAnchor="middle" className="text-[7px] font-mono" fill={BAY.posterior}>
            {credibleInterval[1].toFixed(2)}
          </text>
        </>
      )}

      {/* Panel label */}
      <text x={innerWidth / 2} y={innerHeight + 26} textAnchor="middle" className="text-[9px] font-mono" fill="#71717A">
        {label}
      </text>
    </>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function CoinFlipUpdater() {
  const [trueBias, setTrueBias] = useState(0.6)
  const [priorIdx, setPriorIdx] = useState(0)
  const [flipIndex, setFlipIndex] = useState(0)

  const prior = PRIOR_PRESETS[priorIdx]

  // Pre-generate a batch of flips
  const allFlips = useMemo(
    () => simulateCoinFlips(200, trueBias, 42),
    [trueBias]
  )

  // Model that tracks current state
  const model = useMemo(() => {
    const m = new BetaBinomial(prior.alpha, prior.beta)
    // Replay observed flips
    for (let i = 0; i < flipIndex; i++) {
      if (allFlips.outcomes[i] === 'H') {
        m.observe(1, 0)
      } else {
        m.observe(0, 1)
      }
    }
    return m
  }, [prior, flipIndex, allFlips])

  const priorPDF = useMemo(() => model.getPriorPDF(200), [model])
  const posteriorPDF = useMemo(() => model.getPosteriorPDF(200), [model])
  const likelihoodPDF = useMemo(() => model.getLikelihoodPDF(200), [model])
  const credibleInterval = useMemo(() => model.getCredibleInterval(0.95), [model])

  const heads = flipIndex > 0 ? allFlips.cumulativeHeads[flipIndex - 1] : 0
  const tails = flipIndex > 0 ? allFlips.cumulativeTails[flipIndex - 1] : 0

  const handleFlip1 = useCallback(() => {
    setFlipIndex((prev) => Math.min(prev + 1, 200))
  }, [])

  const handleFlip10 = useCallback(() => {
    setFlipIndex((prev) => Math.min(prev + 10, 200))
  }, [])

  const handleReset = useCallback(() => {
    setFlipIndex(0)
  }, [])

  return (
    <div className="space-y-4">
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Button variant="primary" size="sm" onClick={handleFlip1} disabled={flipIndex >= 200}>
            Flip
          </Button>
          <Button variant="secondary" size="sm" onClick={handleFlip10} disabled={flipIndex >= 200}>
            Flip 10
          </Button>
          <Button variant="ghost" size="sm" onClick={handleReset}>
            Reset
          </Button>

          <div className="h-6 w-px bg-obsidian-border" />

          <Slider
            label="True Bias"
            value={trueBias}
            min={0.05}
            max={0.95}
            step={0.05}
            onChange={(v) => {
              setTrueBias(v)
              setFlipIndex(0)
            }}
            formatValue={(v) => v.toFixed(2)}
            className="w-36"
          />

          <div className="h-6 w-px bg-obsidian-border" />

          <div className="flex items-center gap-1">
            {PRIOR_PRESETS.map((p, i) => (
              <Button
                key={p.label}
                variant="secondary"
                size="sm"
                active={priorIdx === i}
                onClick={() => {
                  setPriorIdx(i)
                  setFlipIndex(0)
                }}
              >
                {p.label}
              </Button>
            ))}
          </div>
        </div>
      </GlassCard>

      {/* Flip history and counts */}
      <div className="flex items-center gap-4 px-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-tertiary">Flips: {flipIndex}</span>
          <span className="text-xs font-mono text-text-tertiary">|</span>
          <span className="text-xs font-mono" style={{ color: '#F87171' }}>
            H: {heads}
          </span>
          <span className="text-xs font-mono" style={{ color: '#38BDF8' }}>
            T: {tails}
          </span>
        </div>

        {/* Recent flips */}
        <div className="flex items-center gap-0.5 overflow-hidden max-w-[300px]">
          {allFlips.outcomes.slice(Math.max(0, flipIndex - 20), flipIndex).map((o, i) => (
            <motion.span
              key={`flip-${flipIndex - Math.min(20, flipIndex) + i}`}
              className="text-[10px] font-mono font-bold"
              style={{ color: o === 'H' ? '#F87171' : '#38BDF8' }}
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.15 }}
            >
              {o}
            </motion.span>
          ))}
        </div>
      </div>

      {/* Three-panel distribution display */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Prior */}
        <GlassCard className="overflow-hidden">
          <div className="px-3 pt-2 pb-0">
            <p className="text-[10px] font-mono font-medium" style={{ color: BAY.prior }}>
              Prior: Beta({prior.alpha}, {prior.beta})
            </p>
          </div>
          <SVGContainer
            aspectRatio={4 / 3}
            minHeight={160}
            maxHeight={220}
            padding={{ top: 10, right: 10, bottom: 32, left: 10 }}
          >
            {({ innerWidth, innerHeight }) => (
              <DistributionPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                curves={[
                  { x: priorPDF.x, y: priorPDF.y, color: BAY.prior, dashed: true, label: 'Prior' },
                ]}
                label="θ (coin bias)"
              />
            )}
          </SVGContainer>
        </GlassCard>

        {/* Likelihood */}
        <GlassCard className="overflow-hidden">
          <div className="px-3 pt-2 pb-0">
            <p className="text-[10px] font-mono font-medium" style={{ color: BAY.likelihood }}>
              Likelihood: {heads}H, {tails}T
            </p>
          </div>
          <SVGContainer
            aspectRatio={4 / 3}
            minHeight={160}
            maxHeight={220}
            padding={{ top: 10, right: 10, bottom: 32, left: 10 }}
          >
            {({ innerWidth, innerHeight }) => (
              <DistributionPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                curves={[
                  { x: likelihoodPDF.x, y: likelihoodPDF.y, color: BAY.likelihood, label: 'Likelihood' },
                ]}
                label="θ (coin bias)"
              />
            )}
          </SVGContainer>
        </GlassCard>

        {/* Posterior */}
        <GlassCard className="overflow-hidden">
          <div className="px-3 pt-2 pb-0">
            <p className="text-[10px] font-mono font-medium" style={{ color: BAY.posterior }}>
              Posterior: Beta({model.alpha.toFixed(0)}, {model.beta.toFixed(0)})
            </p>
          </div>
          <SVGContainer
            aspectRatio={4 / 3}
            minHeight={160}
            maxHeight={220}
            padding={{ top: 10, right: 10, bottom: 32, left: 10 }}
          >
            {({ innerWidth, innerHeight }) => (
              <DistributionPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                curves={[
                  { x: priorPDF.x, y: priorPDF.y, color: BAY.prior, dashed: true, label: 'Prior' },
                  { x: posteriorPDF.x, y: posteriorPDF.y, color: BAY.posterior, filled: true, label: 'Posterior' },
                ]}
                credibleInterval={flipIndex > 0 ? credibleInterval : undefined}
                label="θ (coin bias)"
              />
            )}
          </SVGContainer>
        </GlassCard>
      </div>

      {/* Info cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="rounded-lg border p-3" style={{ borderColor: `${BAY.posterior}25`, backgroundColor: `${BAY.posterior}08` }}>
          <p className="text-[10px] font-mono font-medium" style={{ color: BAY.posterior }}>
            Posterior Mean
          </p>
          <p className="text-lg font-mono font-bold mt-1" style={{ color: BAY.posterior }}>
            {model.getPosteriorMean().toFixed(3)}
          </p>
        </div>
        <div className="rounded-lg border p-3" style={{ borderColor: `${BAY.posterior}25`, backgroundColor: `${BAY.posterior}08` }}>
          <p className="text-[10px] font-mono font-medium" style={{ color: BAY.posterior }}>
            95% Credible Interval
          </p>
          <p className="text-sm font-mono font-bold mt-1" style={{ color: BAY.posterior }}>
            [{credibleInterval[0].toFixed(3)}, {credibleInterval[1].toFixed(3)}]
          </p>
        </div>
        <div className="rounded-lg border p-3" style={{ borderColor: 'rgba(255,255,255,0.08)', backgroundColor: 'rgba(255,255,255,0.02)' }}>
          <p className="text-[10px] font-mono font-medium text-text-tertiary">
            True Bias (hidden)
          </p>
          <p className="text-lg font-mono font-bold mt-1 text-text-secondary">
            {trueBias.toFixed(2)}
          </p>
        </div>
      </div>
    </div>
  )
}
