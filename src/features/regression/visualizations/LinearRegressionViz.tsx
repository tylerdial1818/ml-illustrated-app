import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { makeLinear } from '../../../lib/data/regressionGenerators'
import { solveSimpleOLS } from '../../../lib/algorithms/regression/ols'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'
import { GlassCard } from '../../../components/ui/GlassCard'
import { createRng, normalRandom } from '../../../lib/math/random'
import type { RegressionPoint } from '../../../lib/data/regressionGenerators'

export function LinearRegressionViz() {
  const [noise, setNoise] = useState(1.5)
  const [seed, setSeed] = useState(42)
  const [extraPoints, setExtraPoints] = useState<RegressionPoint[]>([])

  const baseData = useMemo(() => makeLinear(80, 2, 1, noise, seed), [noise, seed])
  const data = useMemo(() => [...baseData, ...extraPoints], [baseData, extraPoints])

  const ols = useMemo(() => solveSimpleOLS(data), [data])

  const addOutlier = useCallback(() => {
    const rng = createRng(Date.now())
    const x = normalRandom(rng, 0, 2)
    const y = normalRandom(rng, 10, 2) // Far from the main trend
    setExtraPoints((prev) => [...prev, { x, y }])
  }, [])

  const resetData = useCallback(() => {
    setExtraPoints([])
    setSeed((s) => s + 1)
  }, [])

  return (
    <GlassCard className="p-6">
      <div className="flex flex-wrap gap-4 mb-6">
        <Slider
          label="Noise"
          value={noise}
          min={0.2}
          max={4}
          step={0.1}
          onChange={(v) => { setNoise(v); setExtraPoints([]) }}
          formatValue={(v) => v.toFixed(1)}
          className="w-48"
        />
        <div className="flex items-end gap-2">
          <Button variant="secondary" size="sm" onClick={addOutlier}>
            Add outlier
          </Button>
          <Button variant="ghost" size="sm" onClick={resetData}>
            Reset
          </Button>
        </div>
      </div>

      <SVGContainer aspectRatio={16 / 10} minHeight={350} maxHeight={550}>
        {({ innerWidth, innerHeight }) => {
          const xExtent = d3.extent(data, (d) => d.x) as [number, number]
          const yExtent = d3.extent(data, (d) => d.y) as [number, number]
          const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice()
          const yScale = d3.scaleLinear().domain([yExtent[0] - 1, yExtent[1] + 1]).range([innerHeight, 0]).nice()

          const [b0, b1] = ols.coefficients

          return (
            <>
              {/* Residual squares */}
              {data.map((point, i) => {
                const predicted = b0 + b1 * point.x
                const residual = point.y - predicted
                const side = Math.abs(yScale(point.y) - yScale(predicted))
                if (side < 2) return null

                return (
                  <rect
                    key={`sq-${i}`}
                    x={xScale(point.x)}
                    y={Math.min(yScale(point.y), yScale(predicted))}
                    width={side}
                    height={side}
                    fill="#F87171"
                    fillOpacity={0.06}
                    stroke="#F87171"
                    strokeWidth={0.5}
                    strokeOpacity={0.15}
                  />
                )
              })}

              {/* Residual lines */}
              {data.map((point, i) => {
                const predicted = b0 + b1 * point.x
                return (
                  <line
                    key={`res-${i}`}
                    x1={xScale(point.x)}
                    y1={yScale(point.y)}
                    x2={xScale(point.x)}
                    y2={yScale(predicted)}
                    stroke="#F87171"
                    strokeWidth={1}
                    strokeOpacity={0.4}
                  />
                )
              })}

              {/* Regression line */}
              <line
                x1={xScale(xExtent[0])}
                y1={yScale(b0 + b1 * xExtent[0])}
                x2={xScale(xExtent[1])}
                y2={yScale(b0 + b1 * xExtent[1])}
                stroke="#818CF8"
                strokeWidth={2.5}
              />

              {/* Equation label */}
              <text x={10} y={20} className="text-xs fill-accent font-mono">
                y = {b1.toFixed(2)}x + {b0.toFixed(2)}
              </text>

              {/* Data points */}
              {data.map((point, i) => (
                <circle
                  key={i}
                  cx={xScale(point.x)}
                  cy={yScale(point.y)}
                  r={3.5}
                  fill={i >= baseData.length ? '#FBBF24' : '#6366F1'}
                  fillOpacity={0.8}
                />
              ))}
            </>
          )
        }}
      </SVGContainer>

      {/* Stats */}
      <div className="mt-4 flex flex-wrap gap-6 text-xs text-text-tertiary font-mono">
        <span>R² = {ols.rSquared.toFixed(4)}</span>
        <span>SSE = {ols.sse.toFixed(2)}</span>
        <span>slope = {ols.coefficients[1].toFixed(4)}</span>
        <span>intercept = {ols.coefficients[0].toFixed(4)}</span>
      </div>
    </GlassCard>
  )
}
