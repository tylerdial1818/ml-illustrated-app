import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { CollapsibleSection } from '../../../components/ui/CollapsibleSection'
import { NaiveBayesPipeline } from '../visualizations/NaiveBayesPipeline'
import { NaiveBayesLiveClassifier } from '../visualizations/NaiveBayesLiveClassifier'
import { NaiveBayesMath } from '../content/naiveBayesMath'

export function NaiveBayesSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="naive-bayes" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">Naive Bayes Classifier</h2>
        <p className="text-base lg:text-lg text-text-secondary mt-2 max-w-2xl leading-relaxed">
          Classify by computing which class makes the observed features most probable, assuming features are independent.
        </p>

        <div className="mt-10 space-y-14">
          {/* Intuition */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
              The Intuition
            </h4>
            <div className="max-w-2xl space-y-3">
              <p className="text-text-secondary leading-relaxed">
                An email arrives. Spam or not? Naive Bayes checks each word independently.
                "Congratulations" is common in spam, rare in real email. "Meeting" is the opposite.
                Multiply all per-word probabilities for each class, pick the class with the highest total.
              </p>
              <p className="text-text-secondary leading-relaxed">
                The "naive" part: it assumes every word is independent of every other word. That is
                clearly wrong ("free" and "money" tend to appear together in spam). But the simplification
                makes the math tractable and the classifier surprisingly effective in practice.
              </p>
            </div>
          </div>

          {/* Mechanism Part A: Pipeline */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-2">
              The Mechanism: Part A
            </h4>
            <p className="text-sm text-text-secondary mb-4 max-w-2xl">
              Walk through the classification of an email step by step. Tokenize, compute per-word
              likelihoods, multiply, and apply the prior.
            </p>
            <NaiveBayesPipeline />
          </div>

          {/* Mechanism Part B: Live classifier */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-2">
              The Mechanism: Part B
            </h4>
            <p className="text-sm text-text-secondary mb-4 max-w-2xl">
              Type anything and watch the classifier respond in real time. Each word pushes
              the tug-of-war bar toward spam or not spam. Try adjusting the prior and smoothing
              to see their effect.
            </p>
            <NaiveBayesLiveClassifier />
          </div>

          {/* Math */}
          <CollapsibleSection title="The Math">
            <NaiveBayesMath />
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
                    <li>Extremely fast to train and predict</li>
                    <li>Works well with small datasets and sparse features</li>
                    <li>Natural baseline for text classification tasks</li>
                    <li>Handles high-dimensional feature spaces gracefully</li>
                  </ul>
                </div>
                <div>
                  <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
                  <ul className="space-y-2 text-sm text-text-secondary">
                    <li>Independence assumption is almost always violated</li>
                    <li>Probabilities are poorly calibrated (overconfident)</li>
                    <li>Cannot capture feature interactions or context</li>
                    <li>Sensitive to the zero-frequency problem without smoothing</li>
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
