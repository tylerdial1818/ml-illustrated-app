import { useMemo, useState, useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import * as d3 from 'd3'
import { makeLinear } from '../../../lib/data/regressionGenerators'
import { solveSimpleOLS } from '../../../lib/algorithms/regression/ols'
import { GlassCard } from '../../../components/ui/GlassCard'

export function RegressionOverview() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.2, once: true })
  const [slope, setSlope] = useState(1.5)
  const [intercept, setIntercept] = useState(0)
  const [showBestFit, setShowBestFit] = useState(false)

  const data = useMemo(() => makeLinear(80, 2, 1, 1.5, 42), [])
  const ols = useMemo(() => solveSimpleOLS(data), [data])

  const width = 600
  const height = 400
  const padding = { top: 20, right: 20, bottom: 40, left: 50 }
  const innerW = width - padding.left - padding.right
  const innerH = height - padding.top - padding.bottom

  const xExtent = d3.extent(data, (d) => d.x) as [number, number]
  const yExtent = d3.extent(data, (d) => d.y) as [number, number]
  const xScale = d3.scaleLinear().domain(xExtent).range([0, innerW]).nice()
  const yScale = d3.scaleLinear().domain([yExtent[0] - 2, yExtent[1] + 2]).range([innerH, 0]).nice()

  const activeSlope = showBestFit ? ols.coefficients[1] : slope
  const activeIntercept = showBestFit ? ols.coefficients[0] : intercept

  // Compute SSE for current line
  const sse = data.reduce((sum, p) => {
    const predicted = activeSlope * p.x + activeIntercept
    return sum + (p.y - predicted) ** 2
  }, 0)

  return (
    <section id="regression-overview" ref={ref} className="py-16 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h4 className="text-xs font-semibold uppercase tracking-[0.1em] text-text-tertiary mb-6">
          The Problem
        </h4>
        <h2 className="text-2xl font-semibold text-text-primary mb-4">
          What is Regression?
        </h2>
        <p className="text-text-secondary max-w-2xl leading-relaxed">
          Regression is about drawing the best line (or curve) through data to predict a continuous outcome.
          The goal: find the line that makes the residual lines (the vertical distances from each point to
          the line) as short as possible.
        </p>

        <GlassCard className="mt-8 p-8">
          <div className="flex flex-col items-center">
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="max-w-full h-auto">
              <g transform={`translate(${padding.left}, ${padding.top})`}>
                {/* Residual lines */}
                {data.map((point, i) => {
                  const predicted = activeSlope * point.x + activeIntercept
                  return (
                    <line
                      key={`residual-${i}`}
                      x1={xScale(point.x)}
                      y1={yScale(point.y)}
                      x2={xScale(point.x)}
                      y2={yScale(predicted)}
                      stroke="#F87171"
                      strokeWidth={1}
                      strokeOpacity={0.3}
                    />
                  )
                })}

                {/* Regression line */}
                <line
                  x1={xScale(xExtent[0])}
                  y1={yScale(activeSlope * xExtent[0] + activeIntercept)}
                  x2={xScale(xExtent[1])}
                  y2={yScale(activeSlope * xExtent[1] + activeIntercept)}
                  stroke="#818CF8"
                  strokeWidth={2}
                />

                {/* Data points */}
                {data.map((point, i) => (
                  <circle
                    key={i}
                    cx={xScale(point.x)}
                    cy={yScale(point.y)}
                    r={3.5}
                    fill="#6366F1"
                    fillOpacity={0.7}
                  />
                ))}

                {/* SSE display */}
                <text x={innerW} y={20} textAnchor="end" className="text-xs fill-text-tertiary font-mono">
                  SSE = {sse.toFixed(1)}
                </text>
              </g>
            </svg>

            <div className="mt-4 flex flex-wrap items-center gap-4">
              {!showBestFit && (
                <>
                  <label className="flex items-center gap-2 text-sm text-text-secondary">
                    Slope
                    <input
                      type="range"
                      min={-1}
                      max={4}
                      step={0.1}
                      value={slope}
                      onChange={(e) => setSlope(parseFloat(e.target.value))}
                      className="w-24"
                    />
                    <span className="font-mono text-xs text-text-tertiary w-8">{slope.toFixed(1)}</span>
                  </label>
                  <label className="flex items-center gap-2 text-sm text-text-secondary">
                    Intercept
                    <input
                      type="range"
                      min={-3}
                      max={5}
                      step={0.1}
                      value={intercept}
                      onChange={(e) => setIntercept(parseFloat(e.target.value))}
                      className="w-24"
                    />
                    <span className="font-mono text-xs text-text-tertiary w-8">{intercept.toFixed(1)}</span>
                  </label>
                </>
              )}
              <button
                onClick={() => setShowBestFit(!showBestFit)}
                className="px-4 py-2 text-sm bg-obsidian-hover border border-obsidian-border rounded-lg text-text-secondary hover:text-text-primary transition-colors"
              >
                {showBestFit ? 'Manual mode' : 'Find best fit'}
              </button>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </section>
  )
}
