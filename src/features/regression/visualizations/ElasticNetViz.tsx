import { useState, useMemo } from 'react'
import * as d3 from 'd3'
import { makeMultiFeature } from '../../../lib/data/regressionGenerators'
import { elasticNetPath } from '../../../lib/algorithms/regression/elasticnet'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function ElasticNetViz() {
  const [alpha, setAlpha] = useState(0.1)
  const [l1Ratio, setL1Ratio] = useState(0.5)

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

  const path = useMemo(
    () => elasticNetPath(Xaug, y, alphas, l1Ratio),
    [Xaug, y, alphas, l1Ratio]
  )

  const alphaIdx = useMemo(() => {
    let best = 0
    let bestDist = Infinity
    alphas.forEach((a, i) => {
      const dist = Math.abs(Math.log10(a) - Math.log10(alpha))
      if (dist < bestDist) { bestDist = dist; best = i }
    })
    return best
  }, [alpha, alphas])

  const currentCoeffs = path.coeffPaths[alphaIdx]?.slice(1) ?? []
  const activeFeatures = currentCoeffs.filter((c) => Math.abs(c) > 1e-10).length

  return (
    <GlassCard className="p-8">
      <div className="flex flex-wrap gap-6 mb-4">
        <Slider
          label="α (regularization)"
          value={Math.log10(alpha)}
          min={-3}
          max={0}
          step={0.05}
          onChange={(v) => setAlpha(Math.pow(10, v))}
          formatValue={() => alpha.toFixed(4)}
          className="w-56"
        />
        <Slider
          label="l1_ratio (L1 vs L2 mix)"
          value={l1Ratio}
          min={0}
          max={1}
          step={0.05}
          onChange={setL1Ratio}
          formatValue={(v) => v.toFixed(2)}
          className="w-56"
        />
      </div>

      <div className="flex flex-wrap gap-2 mb-6">
        <Button
          variant="secondary"
          size="sm"
          active={l1Ratio === 0}
          onClick={() => setL1Ratio(0)}
        >
          Pure Ridge
        </Button>
        <Button
          variant="secondary"
          size="sm"
          active={l1Ratio === 0.5}
          onClick={() => setL1Ratio(0.5)}
        >
          Balanced
        </Button>
        <Button
          variant="secondary"
          size="sm"
          active={l1Ratio === 1}
          onClick={() => setL1Ratio(1)}
        >
          Pure Lasso
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Constraint shape visualization */}
        <div>
          <p className="text-xs text-text-tertiary uppercase tracking-wider mb-2">
            Constraint shape (l1_ratio = {l1Ratio.toFixed(2)})
          </p>
          <SVGContainer aspectRatio={1} minHeight={200} maxHeight={300} padding={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            {({ innerWidth, innerHeight }) => {
              const cx = innerWidth / 2
              const cy = innerHeight / 2
              const r = Math.min(innerWidth, innerHeight) / 2 - 10

              // Generate constraint boundary shape (mix of diamond and circle)
              const points: string[] = []
              for (let theta = 0; theta <= 2 * Math.PI; theta += 0.02) {
                const cosT = Math.cos(theta)
                const sinT = Math.sin(theta)

                // L1 norm radius at this angle: 1 / (|cos| + |sin|)
                const l1r = 1 / (Math.abs(cosT) + Math.abs(sinT))
                // L2 norm radius at this angle: 1 (circle)
                const l2r = 1

                const mixedR = r * (l1Ratio * l1r + (1 - l1Ratio) * l2r)

                const x = cx + mixedR * cosT
                const y = cy + mixedR * sinT
                points.push(`${x},${y}`)
              }

              return (
                <>
                  {/* Axes */}
                  <line x1={0} x2={innerWidth} y1={cy} y2={cy} stroke="rgba(255,255,255,0.08)" />
                  <line x1={cx} x2={cx} y1={0} y2={innerHeight} stroke="rgba(255,255,255,0.08)" />

                  {/* Constraint shape */}
                  <polygon
                    points={points.join(' ')}
                    fill={COLORS.accent}
                    fillOpacity={0.1}
                    stroke={COLORS.accent}
                    strokeWidth={2}
                    strokeOpacity={0.6}
                  />

                  {/* Labels */}
                  <text x={innerWidth - 5} y={cy - 5} textAnchor="end" className="text-[10px] fill-text-tertiary">
                    β₁
                  </text>
                  <text x={cx + 5} y={12} className="text-[10px] fill-text-tertiary">
                    β₂
                  </text>
                </>
              )
            }}
          </SVGContainer>
        </div>

        {/* Coefficient trace */}
        <div>
          <p className="text-xs text-text-tertiary uppercase tracking-wider mb-2">
            Coefficient paths ({activeFeatures} active)
          </p>
          <SVGContainer aspectRatio={16 / 10} minHeight={200} maxHeight={300} padding={{ top: 15, right: 15, bottom: 25, left: 40 }}>
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
                    x1={xScale(alpha)}
                    x2={xScale(alpha)}
                    y1={0}
                    y2={innerHeight}
                    stroke="#fff"
                    strokeWidth={1}
                    strokeOpacity={0.3}
                    strokeDasharray="4,4"
                  />
                </>
              )
            }}
          </SVGContainer>
        </div>
      </div>
    </GlassCard>
  )
}
