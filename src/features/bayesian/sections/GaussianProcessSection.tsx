import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { CollapsibleSection } from '../../../components/ui/CollapsibleSection'
import { GaussianProcessViz } from '../visualizations/GaussianProcessViz'
import { GaussianProcessMath } from '../content/gaussianProcessMath'

export function GaussianProcessSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="gaussian-processes" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">Gaussian Processes</h2>
        <p className="text-base lg:text-lg text-text-secondary mt-2 max-w-2xl leading-relaxed">
          A distribution over functions, not parameters. Make predictions with honest uncertainty bands that widen where you have less data.
        </p>

        <div className="mt-10 space-y-14">
          {/* Intuition */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
              The Intuition
            </h4>
            <div className="max-w-2xl space-y-3">
              <p className="text-text-secondary leading-relaxed">
                Forget about finding the best line or curve. A Gaussian Process does not commit to any
                specific function. Instead, it defines a distribution over ALL possible smooth functions
                that could explain your data.
              </p>
              <p className="text-text-secondary leading-relaxed">
                Before seeing any data, the GP considers an infinite number of plausible functions. After
                seeing data, it narrows to only the functions that pass through (or near) the observed
                points. The result: a mean prediction plus a confidence band that is tight near data and
                wide where you are uncertain.
              </p>
              <p className="text-text-secondary leading-relaxed">
                You control the GP's behavior through the kernel function. The kernel defines what
                "similar" means. A squared exponential kernel says "nearby inputs should produce similar
                outputs" and gives smooth predictions. A periodic kernel says "the function repeats."
                The kernel is the GP's built-in assumption about the shape of the world.
              </p>
            </div>
          </div>

          {/* Mechanism */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-2">
              The Mechanism
            </h4>
            <p className="text-sm text-text-secondary mb-4 max-w-2xl">
              Click to place data points and watch the uncertainty band contract. Try the "Gap in
              middle" preset to see uncertainty balloon where there is no data. Sweep the length
              scale from very small (wiggly) to very large (smooth) to see the effect on predictions.
            </p>
            <GaussianProcessViz />
          </div>

          {/* Math */}
          <CollapsibleSection title="The Math">
            <GaussianProcessMath />
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
                    <li>Calibrated uncertainty that widens in data-sparse regions</li>
                    <li>Non-parametric: model complexity grows with data</li>
                    <li>Kernel encodes prior assumptions about function smoothness</li>
                    <li>Foundation for Bayesian optimization and active learning</li>
                    <li>Works well for spatial and temporal data</li>
                  </ul>
                </div>
                <div>
                  <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
                  <ul className="space-y-2 text-sm text-text-secondary">
                    <li>O(n³) training cost limits to ~10,000 data points</li>
                    <li>Struggles with high-dimensional inputs</li>
                    <li>Kernel choice is critical and requires domain knowledge</li>
                    <li>Not suitable for discrete or categorical inputs</li>
                    <li>Exact inference intractable for large datasets</li>
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
