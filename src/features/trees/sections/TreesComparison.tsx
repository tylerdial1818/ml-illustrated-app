import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'

export function TreesComparison() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="trees-comparison" ref={ref} className="py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-3xl font-semibold text-text-primary">Comparison</h2>
        <p className="mt-2 text-lg text-text-secondary max-w-2xl">
          Three approaches to the same idea — splitting data with yes/no questions.
          Each makes different trade-offs between interpretability, accuracy, and robustness.
        </p>

        {/* Three-column comparison grid */}
        <GlassCard className="mt-8 p-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div>
              <h3 className="text-sm font-semibold text-text-primary mb-3">Decision Tree</h3>
              <div className="space-y-3 text-sm text-text-secondary">
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Approach</p>
                  <p>Single tree, greedy splits</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Interpretability</p>
                  <p>High — full decision path is visible</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Variance</p>
                  <p>High — sensitive to data changes</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Bias</p>
                  <p>Low (deep trees) to High (shallow)</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Training</p>
                  <p>Fast, single pass</p>
                </div>
              </div>
            </div>

            <div className="border-t lg:border-t-0 lg:border-l border-obsidian-border pt-6 lg:pt-0 lg:pl-6">
              <h3 className="text-sm font-semibold text-text-primary mb-3">Random Forest</h3>
              <div className="space-y-3 text-sm text-text-secondary">
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Approach</p>
                  <p>Many trees, bagging + feature subsampling</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Interpretability</p>
                  <p>Medium — feature importances available</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Variance</p>
                  <p>Low — averaging smooths out noise</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Bias</p>
                  <p>Similar to individual trees</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Training</p>
                  <p>Parallelizable — trees are independent</p>
                </div>
              </div>
            </div>

            <div className="border-t lg:border-t-0 lg:border-l border-obsidian-border pt-6 lg:pt-0 lg:pl-6">
              <h3 className="text-sm font-semibold text-text-primary mb-3">Gradient Boosted Trees</h3>
              <div className="space-y-3 text-sm text-text-secondary">
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Approach</p>
                  <p>Sequential trees, each fixes errors</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Interpretability</p>
                  <p>Low — complex ensemble of weak learners</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Variance</p>
                  <p>Low — regularized by learning rate</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Bias</p>
                  <p>Low — iteratively reduces residuals</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-text-tertiary mb-1">Training</p>
                  <p>Sequential — each tree depends on previous</p>
                </div>
              </div>
            </div>
          </div>
        </GlassCard>

        {/* Selection guide */}
        <GlassCard className="mt-6 p-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Which model should you use?</h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-1 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Do you need to explain every decision?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> Decision Tree. You can show the exact path of rules that led to each prediction.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-2 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Do you need a strong baseline with minimal tuning?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> Random Forest. It works well out of the box and is hard to badly misconfigure.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-3 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Do you need maximum predictive accuracy on tabular data?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> Gradient Boosted Trees. With proper tuning, GBTs consistently achieve
                  state-of-the-art results on structured data.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="text-cluster-4 font-bold mt-0.5">?</span>
              <div>
                <p className="text-text-primary font-medium">Is training speed a concern?</p>
                <p className="text-text-secondary">
                  <strong>Decision Tree</strong> is fastest. <strong>Random Forest</strong> parallelizes
                  well. <strong>GBT</strong> is sequential but modern implementations (XGBoost, LightGBM)
                  are heavily optimized.
                </p>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </section>
  )
}
