import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function TreesOverview() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.2, once: true })

  return (
    <section id="trees-overview" ref={ref} className="py-16 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h4 className="text-xs font-semibold uppercase tracking-[0.1em] text-text-tertiary mb-6">
          The Idea
        </h4>
        <h2 className="text-2xl font-semibold text-text-primary mb-4">
          The Decision Game
        </h2>
        <p className="text-text-secondary max-w-2xl leading-relaxed">
          Tree-based models learn a sequence of yes/no questions that carve feature space into regions.
          Each question splits the data along one feature, and the process repeats until
          each region is "pure enough" to make a confident prediction. Think of it as playing
          20 Questions with your data — each answer narrows down the possibilities.
        </p>
        <p className="mt-3 text-text-secondary max-w-2xl leading-relaxed">
          A single tree is intuitive but fragile. The real power comes from combining many trees
          into ensembles — either by training them independently and voting (bagging) or by training
          them sequentially so each one fixes the previous mistakes (boosting).
        </p>

        {/* Taxonomy */}
        <div className="mt-10 grid grid-cols-1 sm:grid-cols-3 gap-3">
          <GlassCard className="p-5">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.clusters[0] }} />
              <span className="text-sm font-medium text-text-primary">Single Tree</span>
            </div>
            <p className="text-xs text-text-secondary">
              A flowchart of if/else rules. Highly interpretable but prone to overfitting (Decision Tree / CART).
            </p>
          </GlassCard>
          <GlassCard className="p-5">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.clusters[1] }} />
              <span className="text-sm font-medium text-text-primary">Bagging (Random Forest)</span>
            </div>
            <p className="text-xs text-text-secondary">
              Train many trees on random subsets of data and features. Let them vote. Reduces variance without increasing bias.
            </p>
          </GlassCard>
          <GlassCard className="p-5">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.clusters[2] }} />
              <span className="text-sm font-medium text-text-primary">Boosting (GBT)</span>
            </div>
            <p className="text-xs text-text-secondary">
              Build trees one at a time, each correcting the mistakes of the ensemble so far. State-of-the-art for tabular data.
            </p>
          </GlassCard>
        </div>

        {/* Key concept callout */}
        <GlassCard className="mt-6 p-6">
          <p className="text-sm text-text-secondary leading-relaxed">
            <strong className="text-text-primary">The core trade-off:</strong> a single deep tree memorizes
            the training data (high variance, low bias). Ensembles fix this — Random Forests average out
            the noise, while Gradient Boosted Trees carefully reduce bias one step at a time.
          </p>
        </GlassCard>
      </motion.div>
    </section>
  )
}
