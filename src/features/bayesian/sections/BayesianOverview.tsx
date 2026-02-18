import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { PointVsDistributionViz } from '../visualizations/PointVsDistributionViz'

const BAYESIAN_COLORS = {
  prior: '#A1A1AA',
  likelihood: '#FBBF24',
  posterior: '#6366F1',
}

export function BayesianOverview() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="overview" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">
          A Different Way to Learn
        </h2>

        {/* Prominent opening text */}
        <div className="mt-6 max-w-3xl">
          <p className="text-text-secondary leading-relaxed">
            Every model you have seen so far on this site finds a single best answer. Linear
            regression finds the one best-fit line. A decision tree finds the one best set of
            splits. These are called point estimates. Bayesian machine learning takes a
            completely different approach. Instead of finding one best answer, it maintains a
            full probability distribution over all possible answers and updates that distribution
            as it sees more data.
          </p>
          <GlassCard className="mt-4 p-4">
            <p className="text-text-primary font-medium leading-relaxed">
              Bayesian ML is not a model family so much as an alternative philosophy for how you
              estimate parameters and make predictions. The payoff is that every prediction comes
              with a built-in measure of uncertainty. The model tells you not just "what" but
              "how sure."
            </p>
          </GlassCard>
        </div>

        {/* Visualization */}
        <div className="mt-10">
          <PointVsDistributionViz />
        </div>

        {/* Three ingredients */}
        <div className="mt-10 max-w-3xl">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
            Three Ingredients
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div
              className="rounded-lg border p-4"
              style={{ borderColor: `${BAYESIAN_COLORS.prior}40`, backgroundColor: `${BAYESIAN_COLORS.prior}08` }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-0.5 border-t-2 border-dashed" style={{ borderColor: BAYESIAN_COLORS.prior }} />
                <p className="text-xs font-mono font-medium" style={{ color: BAYESIAN_COLORS.prior }}>
                  Prior
                </p>
              </div>
              <p className="text-[11px] text-text-secondary leading-relaxed">
                What you believe before seeing any data. Could be uninformed (flat) or encode
                domain knowledge (e.g. "slopes are probably small").
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: `${BAYESIAN_COLORS.likelihood}40`, backgroundColor: `${BAYESIAN_COLORS.likelihood}08` }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-0.5" style={{ backgroundColor: BAYESIAN_COLORS.likelihood }} />
                <p className="text-xs font-mono font-medium" style={{ color: BAYESIAN_COLORS.likelihood }}>
                  Likelihood
                </p>
              </div>
              <p className="text-[11px] text-text-secondary leading-relaxed">
                How probable is the observed data for each possible parameter value? This is
                where your data speaks.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: `${BAYESIAN_COLORS.posterior}40`, backgroundColor: `${BAYESIAN_COLORS.posterior}08` }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-1 rounded" style={{ backgroundColor: BAYESIAN_COLORS.posterior }} />
                <p className="text-xs font-mono font-medium" style={{ color: BAYESIAN_COLORS.posterior }}>
                  Posterior
                </p>
              </div>
              <p className="text-[11px] text-text-secondary leading-relaxed">
                Your updated belief after combining prior and data. This is the full answer:
                a distribution, not a single number.
              </p>
            </div>
          </div>
        </div>

        {/* Update cycle */}
        <div className="mt-8 max-w-3xl">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-4">
            The Bayesian Update Cycle
          </h3>
          <div className="flex items-center gap-3 flex-wrap">
            <span
              className="px-3 py-1.5 rounded-lg text-xs font-mono font-medium border border-dashed"
              style={{ color: BAYESIAN_COLORS.prior, borderColor: `${BAYESIAN_COLORS.prior}50` }}
            >
              Prior
            </span>
            <span className="text-text-tertiary text-xs">+</span>
            <span
              className="px-3 py-1.5 rounded-lg text-xs font-mono font-medium border"
              style={{ color: BAYESIAN_COLORS.likelihood, borderColor: `${BAYESIAN_COLORS.likelihood}50` }}
            >
              Data
            </span>
            <span className="text-text-tertiary text-xs">→</span>
            <span
              className="px-3 py-1.5 rounded-lg text-xs font-mono font-medium border"
              style={{ color: BAYESIAN_COLORS.posterior, borderColor: `${BAYESIAN_COLORS.posterior}50`, backgroundColor: `${BAYESIAN_COLORS.posterior}15` }}
            >
              Posterior
            </span>
            <span className="text-text-tertiary text-xs mx-2">⟲</span>
            <span className="text-[10px] text-text-tertiary italic">
              Tomorrow's posterior becomes next experiment's prior
            </span>
          </div>
        </div>

        {/* Why it matters */}
        <div className="mt-8 max-w-3xl">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-4">
            Why Go Bayesian?
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="text-[11px] text-text-secondary leading-relaxed">
              <p className="text-text-primary font-medium mb-1">Small data</p>
              When you have few observations, priors prevent the model from making wild predictions.
            </div>
            <div className="text-[11px] text-text-secondary leading-relaxed">
              <p className="text-text-primary font-medium mb-1">Safety-critical decisions</p>
              Medical diagnosis, autonomous driving. You need to know how confident the model is.
            </div>
            <div className="text-[11px] text-text-secondary leading-relaxed">
              <p className="text-text-primary font-medium mb-1">Active learning</p>
              Uncertainty tells you where to collect the next data point for maximum information gain.
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  )
}
