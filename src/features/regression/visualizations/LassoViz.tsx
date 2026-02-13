import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { makeMultiFeature } from '../../../lib/data/regressionGenerators'
import { ridgePath } from '../../../lib/algorithms/regression/ridge'
import { lassoPath } from '../../../lib/algorithms/regression/lasso'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function LassoViz() {
  const [alpha, setAlpha] = useState(0.1)

  const { X, y } = useMemo(
    () => makeMultiFeature(100, 8, 4, 0.5, 42),
    []
  )

  const Xaug = useMemo(() => X.map((row) => [1, ...row]), [X])

  const alphas = useMemo(() => {
    const result: number[] = []
    for (let i = 0; i <= 40; i++) {
      result.push(Math.pow(10, -3 + (3 * i) / 40))
    }
    return result
  }, [])

  const ridge = useMemo(() => ridgePath(Xaug, y, alphas), [Xaug, y, alphas])
  const lasso = useMemo(() => lassoPath(Xaug, y, alphas), [Xaug, y, alphas])

  // Find closest alpha index
  const alphaIdx = useMemo(() => {
    let best = 0
    let bestDist = Infinity
    alphas.forEach((a, i) => {
      const dist = Math.abs(Math.log10(a) - Math.log10(alpha))
      if (dist < bestDist) { bestDist = dist; best = i }
    })
    return best
  }, [alpha, alphas])

  const activeFeatures = lasso.activeFeatureCounts[alphaIdx] ?? 0

  return (
    <GlassCard className="p-6">
      <div className="flex flex-wrap gap-6 mb-6 items-end">
        <Slider
          label="α (regularization)"
          value={Math.log10(alpha)}
          min={-3}
          max={0}
          step={0.05}
          onChange={(v) => setAlpha(Math.pow(10, v))}
          formatValue={() => alpha.toFixed(4)}
          className="w-64"
        />
        <span className="text-xs text-text-tertiary font-mono pb-1">
          Active features: {activeFeatures} / 8
        </span>
      </div>

      <p className="text-xs text-text-tertiary uppercase tracking-wider mb-3">
        Ridge vs Lasso coefficient paths (side by side)
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Ridge trace */}
        <CoeffTracePlot
          title="Ridge (L2)"
          path={ridge}
          alphas={alphas}
          currentAlpha={alpha}
          subtitle="Coefficients shrink but never reach zero"
        />

        {/* Lasso trace */}
        <CoeffTracePlot
          title="Lasso (L1)"
          path={lasso}
          alphas={alphas}
          currentAlpha={alpha}
          subtitle="Coefficients are driven to zero one by one"
        />
      </div>
    </GlassCard>
  )
}

function CoeffTracePlot({
  title,
  path,
  alphas,
  currentAlpha,
  subtitle,
}: {
  title: string
  path: { coeffPaths: number[][] }
  alphas: number[]
  currentAlpha: number
  subtitle: string
}) {
  return (
    <div>
      <p className="text-sm font-medium text-text-primary mb-0.5">{title}</p>
      <p className="text-[10px] text-text-tertiary mb-2">{subtitle}</p>
      <SVGContainer aspectRatio={16 / 10} minHeight={220} maxHeight={350} padding={{ top: 15, right: 15, bottom: 25, left: 40 }}>
        {({ innerWidth, innerHeight }) => {
          const nFeatures = path.coeffPaths[0].length - 1
          const xScale = d3.scaleLog().domain([alphas[0], alphas[alphas.length - 1]]).range([0, innerWidth])
          const allCoeffs = path.coeffPaths.flatMap((c) => c.slice(1))
          const yMax = Math.max(d3.max(allCoeffs.map(Math.abs)) ?? 1, 0.1)
          const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerHeight, 0])

          return (
            <>
              <line x1={0} x2={innerWidth} y1={yScale(0)} y2={yScale(0)} stroke="rgba(255,255,255,0.1)" strokeDasharray="4,4" />

              {Array.from({ length: nFeatures }, (_, fi) => {
                const lineData = path.coeffPaths.map((c, ai) => ({
                  x: xScale(alphas[ai]),
                  y: yScale(c[fi + 1]),
                }))
                const d = lineData.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ')
                return (
                  <path
                    key={fi}
                    d={d}
                    fill="none"
                    stroke={COLORS.clusters[fi % COLORS.clusters.length]}
                    strokeWidth={1.5}
                    strokeOpacity={0.7}
                  />
                )
              })}

              <line
                x1={xScale(currentAlpha)}
                x2={xScale(currentAlpha)}
                y1={0}
                y2={innerHeight}
                stroke="#fff"
                strokeWidth={1}
                strokeOpacity={0.3}
                strokeDasharray="4,4"
              />

              <text x={innerWidth / 2} y={innerHeight + 20} textAnchor="middle" className="text-[9px] fill-text-tertiary">
                α (log scale)
              </text>
            </>
          )
        }}
      </SVGContainer>
    </div>
  )
}
