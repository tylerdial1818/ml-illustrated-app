import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { makeMultiFeature } from '../../../lib/data/regressionGenerators'
import { ridgePath } from '../../../lib/algorithms/regression/ridge'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function RidgeViz() {
  const [alpha, setAlpha] = useState(1.0)

  const { X, y, trueCoeffs } = useMemo(
    () => makeMultiFeature(100, 8, 4, 0.5, 42),
    []
  )

  // Add intercept column
  const Xaug = useMemo(() => X.map((row) => [1, ...row]), [X])

  const alphas = useMemo(() => {
    const result: number[] = []
    for (let i = 0; i <= 50; i++) {
      result.push(Math.pow(10, -2 + (4 * i) / 50))
    }
    return result
  }, [])

  const path = useMemo(() => ridgePath(Xaug, y, alphas), [Xaug, y, alphas])

  // Find closest alpha index for current slider value
  const alphaIdx = useMemo(() => {
    let best = 0
    let bestDist = Infinity
    alphas.forEach((a, i) => {
      const dist = Math.abs(Math.log10(a) - Math.log10(alpha))
      if (dist < bestDist) { bestDist = dist; best = i }
    })
    return best
  }, [alpha, alphas])

  const currentCoeffs = path.coeffPaths[alphaIdx]

  return (
    <GlassCard className="p-6">
      <div className="flex flex-wrap gap-6 mb-6">
        <Slider
          label="α (regularization)"
          value={Math.log10(alpha)}
          min={-2}
          max={2}
          step={0.05}
          onChange={(v) => setAlpha(Math.pow(10, v))}
          formatValue={() => alpha.toFixed(3)}
          className="w-64"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Coefficient trace plot */}
        <div>
          <p className="text-xs text-text-tertiary uppercase tracking-wider mb-2">Coefficient paths</p>
          <SVGContainer aspectRatio={16 / 10} minHeight={250} maxHeight={400} padding={{ top: 20, right: 20, bottom: 30, left: 50 }}>
            {({ innerWidth, innerHeight }) => {
              const nFeatures = path.coeffPaths[0].length - 1 // exclude intercept
              const xScale = d3.scaleLog().domain([alphas[0], alphas[alphas.length - 1]]).range([0, innerWidth])
              const allCoeffs = path.coeffPaths.flatMap((c) => c.slice(1))
              const yMax = Math.max(d3.max(allCoeffs) ?? 1, Math.abs(d3.min(allCoeffs) ?? -1))
              const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerHeight, 0])

              return (
                <>
                  {/* Zero line */}
                  <line x1={0} x2={innerWidth} y1={yScale(0)} y2={yScale(0)} stroke="rgba(255,255,255,0.1)" strokeDasharray="4,4" />

                  {/* Coefficient lines */}
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

                  {/* Current alpha line */}
                  <line
                    x1={xScale(alpha)}
                    x2={xScale(alpha)}
                    y1={0}
                    y2={innerHeight}
                    stroke="#fff"
                    strokeWidth={1}
                    strokeOpacity={0.3}
                    strokeDasharray="4,4"
                  />

                  {/* Axis labels */}
                  <text x={innerWidth / 2} y={innerHeight + 24} textAnchor="middle" className="text-[10px] fill-text-tertiary">
                    α (log scale)
                  </text>
                </>
              )
            }}
          </SVGContainer>
        </div>

        {/* Coefficient bar chart at current alpha */}
        <div>
          <p className="text-xs text-text-tertiary uppercase tracking-wider mb-2">
            Coefficients at α={alpha.toFixed(3)}
          </p>
          <SVGContainer aspectRatio={16 / 10} minHeight={250} maxHeight={400} padding={{ top: 20, right: 20, bottom: 30, left: 40 }}>
            {({ innerWidth, innerHeight }) => {
              if (!currentCoeffs) return null
              const coeffs = currentCoeffs.slice(1) // exclude intercept
              const nFeatures = coeffs.length
              const barWidth = innerWidth / nFeatures - 4
              const yMax = Math.max(...coeffs.map(Math.abs), ...trueCoeffs.map(Math.abs), 0.1)
              const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerHeight, 0])

              return (
                <>
                  {/* Zero line */}
                  <line x1={0} x2={innerWidth} y1={yScale(0)} y2={yScale(0)} stroke="rgba(255,255,255,0.15)" />

                  {/* True coefficients */}
                  {trueCoeffs.map((c, i) => (
                    <rect
                      key={`true-${i}`}
                      x={i * (barWidth + 4) + 1}
                      y={c >= 0 ? yScale(c) : yScale(0)}
                      width={barWidth / 2 - 1}
                      height={Math.abs(yScale(c) - yScale(0))}
                      fill="#fff"
                      fillOpacity={0.1}
                    />
                  ))}

                  {/* Estimated coefficients */}
                  {coeffs.map((c, i) => (
                    <rect
                      key={`est-${i}`}
                      x={i * (barWidth + 4) + barWidth / 2}
                      y={c >= 0 ? yScale(c) : yScale(0)}
                      width={barWidth / 2 - 1}
                      height={Math.abs(yScale(c) - yScale(0))}
                      fill={COLORS.clusters[i % COLORS.clusters.length]}
                      fillOpacity={0.7}
                    />
                  ))}

                  {/* Feature labels */}
                  {coeffs.map((_, i) => (
                    <text
                      key={`label-${i}`}
                      x={i * (barWidth + 4) + barWidth / 2}
                      y={innerHeight + 16}
                      textAnchor="middle"
                      className="text-[9px] fill-text-tertiary"
                    >
                      β{i + 1}
                    </text>
                  ))}
                </>
              )
            }}
          </SVGContainer>
          <div className="flex items-center gap-4 text-[10px] text-text-tertiary mt-1">
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 bg-white/10 inline-block" /> True
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 bg-cluster-1 inline-block" /> Estimated
            </span>
          </div>
        </div>
      </div>
    </GlassCard>
  )
}
