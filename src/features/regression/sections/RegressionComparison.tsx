import { useMemo, useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import * as d3 from 'd3'
import { makeMultiFeature } from '../../../lib/data/regressionGenerators'
import { solveOLS } from '../../../lib/algorithms/regression/ols'
import { solveRidge } from '../../../lib/algorithms/regression/ridge'
import { solveLasso } from '../../../lib/algorithms/regression/lasso'
import { solveElasticNet } from '../../../lib/algorithms/regression/elasticnet'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function RegressionComparison() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  const { X, y, trueCoeffs } = useMemo(
    () => makeMultiFeature(100, 8, 4, 0.5, 42),
    []
  )

  const Xaug = useMemo(() => X.map((row) => [1, ...row]), [X])

  const results = useMemo(() => {
    const ols = solveOLS(Xaug, y)
    const ridge = solveRidge(Xaug, y, 1.0)
    const lasso = solveLasso(Xaug, y, 0.1)
    const elasticnet = solveElasticNet(Xaug, y, 0.1, 0.5)
    return { ols, ridge, lasso, elasticnet }
  }, [Xaug, y])

  const methods = [
    { name: 'OLS', coeffs: results.ols.coefficients.slice(1), r2: results.ols.rSquared, color: COLORS.clusters[0] },
    { name: 'Ridge', coeffs: results.ridge.coefficients.slice(1), r2: results.ridge.rSquared, color: COLORS.clusters[1] },
    { name: 'Lasso', coeffs: results.lasso.coefficients.slice(1), r2: results.lasso.rSquared, color: COLORS.clusters[2] },
    { name: 'ElasticNet', coeffs: results.elasticnet.coefficients.slice(1), r2: results.elasticnet.rSquared, color: COLORS.clusters[3] },
  ]

  const nFeatures = trueCoeffs.length
  const allCoeffs = methods.flatMap((m) => m.coeffs)
  const yMax = Math.max(...allCoeffs.map(Math.abs), ...trueCoeffs.map(Math.abs), 0.1)

  return (
    <section id="regression-comparison" ref={ref} className="py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-3xl font-semibold text-text-primary">Comparison</h2>
        <p className="mt-2 text-lg text-text-secondary max-w-2xl">
          The same dataset (8 features, only 4 relevant), four different regression methods.
        </p>

        <GlassCard className="mt-8 p-8">
          <p className="text-xs text-text-tertiary uppercase tracking-wider mb-4">Coefficient comparison</p>

          {/* Coefficient comparison chart */}
          <div className="overflow-x-auto">
            <svg width={700} height={300} viewBox="0 0 700 300" className="max-w-full h-auto">
              <g transform="translate(50, 20)">
                {(() => {
                  const innerW = 620
                  const innerH = 240
                  const groupWidth = innerW / nFeatures
                  const barWidth = groupWidth / (methods.length + 1) - 2
                  const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerH, 0])

                  return (
                    <>
                      {/* Zero line */}
                      <line x1={0} x2={innerW} y1={yScale(0)} y2={yScale(0)} stroke="rgba(255,255,255,0.15)" />

                      {/* True coefficients (background) */}
                      {trueCoeffs.map((c, i) => (
                        <rect
                          key={`true-${i}`}
                          x={i * groupWidth + 2}
                          y={c >= 0 ? yScale(c) : yScale(0)}
                          width={barWidth}
                          height={Math.abs(yScale(c) - yScale(0))}
                          fill="#fff"
                          fillOpacity={0.08}
                        />
                      ))}

                      {/* Method coefficients */}
                      {methods.map((method, mi) =>
                        method.coeffs.map((c, fi) => (
                          <rect
                            key={`${method.name}-${fi}`}
                            x={fi * groupWidth + (mi + 1) * barWidth + 4}
                            y={c >= 0 ? yScale(c) : yScale(0)}
                            width={barWidth}
                            height={Math.max(1, Math.abs(yScale(c) - yScale(0)))}
                            fill={method.color}
                            fillOpacity={0.7}
                          />
                        ))
                      )}

                      {/* Feature labels */}
                      {Array.from({ length: nFeatures }, (_, i) => (
                        <text
                          key={`label-${i}`}
                          x={i * groupWidth + groupWidth / 2}
                          y={innerH + 20}
                          textAnchor="middle"
                          className="text-[10px] fill-text-tertiary"
                        >
                          β{i + 1} {i < 4 ? '' : '(noise)'}
                        </text>
                      ))}
                    </>
                  )
                })()}
              </g>
            </svg>
          </div>

          {/* Legend + stats */}
          <div className="flex flex-wrap gap-6 mt-4">
            <span className="flex items-center gap-1.5 text-xs text-text-tertiary">
              <span className="w-3 h-2 bg-white/10 inline-block rounded-sm" /> True
            </span>
            {methods.map((m) => (
              <span key={m.name} className="flex items-center gap-1.5 text-xs text-text-secondary">
                <span className="w-3 h-2 rounded-sm inline-block" style={{ backgroundColor: m.color }} />
                {m.name} (R²={m.r2.toFixed(3)})
              </span>
            ))}
          </div>
        </GlassCard>

        {/* Selection guide */}
        <GlassCard className="mt-6 p-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Which regression method?</h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-1 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Just need a baseline?</p>
                <p className="text-text-secondary">Use OLS. Simple, fast, interpretable.</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-2 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Correlated features?</p>
                <p className="text-text-secondary">Ridge or ElasticNet. They handle multicollinearity gracefully.</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-3 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Many features, some irrelevant?</p>
                <p className="text-text-secondary">Lasso. It'll zero out the noise features automatically.</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-4 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Want feature selection + stability?</p>
                <p className="text-text-secondary">ElasticNet. The best of both worlds.</p>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </section>
  )
}
