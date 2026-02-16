import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { FreqVsBayesComparison } from '../visualizations/FreqVsBayesComparison'
import { BayesianFlowchart } from '../visualizations/BayesianFlowchart'

export function BayesianComparisonSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="comparison" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">Comparison &amp; When to Go Bayesian</h2>
        <p className="text-base lg:text-lg text-text-secondary mt-2 max-w-2xl leading-relaxed">
          Frequentist vs. Bayesian on the same problem. Same data, different philosophies, different outputs.
        </p>

        <div className="mt-10 space-y-14">
          {/* Regression comparison */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-2">
              Same Data, Four Methods
            </h4>
            <p className="text-sm text-text-secondary mb-4 max-w-2xl">
              Slide the data points from 5 to 200 and watch. With few points, Bayesian methods show
              wide honest uncertainty bands. OLS shows the same confidence regardless. At 200 points,
              all methods converge.
            </p>
            <FreqVsBayesComparison />
          </div>

          {/* Flowchart + table */}
          <BayesianFlowchart />
        </div>
      </motion.div>
    </section>
  )
}
