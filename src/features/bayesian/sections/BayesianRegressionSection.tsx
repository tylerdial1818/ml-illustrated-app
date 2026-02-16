import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { CollapsibleSection } from '../../../components/ui/CollapsibleSection'
import { BayesianRegressionViz } from '../visualizations/BayesianRegressionViz'
import { BayesianRegressionMath } from '../content/bayesianRegressionMath'

export function BayesianRegressionSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="bayesian-regression" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">Bayesian Linear Regression</h2>
        <p className="text-base lg:text-lg text-text-secondary mt-2 max-w-2xl leading-relaxed">
          Linear regression, but instead of one best-fit line, you get a probability distribution over all plausible lines.
        </p>

        <div className="mt-10 space-y-14">
          {/* Intuition */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
              The Intuition
            </h4>
            <div className="max-w-2xl space-y-3">
              <p className="text-text-secondary leading-relaxed">
                In ordinary regression you get one slope and one intercept. That is it. No way
                to say "I am confident about predictions near the data but unsure far away."
              </p>
              <p className="text-text-secondary leading-relaxed">
                Bayesian regression treats the slope and intercept as random variables with their own
                distributions. Start with a prior belief about what they might be. Then observe data
                and the posterior tightens. Regions with lots of data get narrow prediction bands.
                Sparse regions stay wide, reflecting honest uncertainty.
              </p>
              <p className="text-text-secondary leading-relaxed">
                The result is not one line but a whole family of plausible lines, each weighted by
                how likely it is given the data. Every prediction comes with a built-in uncertainty
                estimate, for free.
              </p>
            </div>
          </div>

          {/* Mechanism */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-2">
              The Mechanism
            </h4>
            <p className="text-sm text-text-secondary mb-4 max-w-2xl">
              Click to place data points. Watch the parameter-space posterior tighten and the
              regression lines converge. Hover in the data space to see the predictive distribution
              widen in data-sparse regions.
            </p>
            <BayesianRegressionViz />
          </div>

          {/* Math */}
          <CollapsibleSection title="The Math">
            <BayesianRegressionMath />
          </CollapsibleSection>

          {/* When to Use */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
              When to Use
            </h4>
            <div className="max-w-2xl">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
                  <ul className="space-y-2 text-sm text-text-secondary">
                    <li>Calibrated uncertainty: predictions are wide where data is sparse</li>
                    <li>Small datasets: the prior acts as regularization</li>
                    <li>Encode domain knowledge via informative priors</li>
                    <li>The MAP with a Gaussian prior is Ridge regression (free connection)</li>
                  </ul>
                </div>
                <div>
                  <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
                  <ul className="space-y-2 text-sm text-text-secondary">
                    <li>More expensive than OLS (matrix inversions)</li>
                    <li>Prior choice can feel arbitrary without domain knowledge</li>
                    <li>Scales poorly to very high dimensions (need approximate inference)</li>
                    <li>Assumes Gaussian noise, which may not hold</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  )
}
