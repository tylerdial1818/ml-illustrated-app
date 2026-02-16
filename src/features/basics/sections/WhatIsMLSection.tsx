import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { SupervisedVsUnsupervisedViz } from '../visualizations/SupervisedVsUnsupervisedViz'

// ── Animated flow panel (traditional vs ML) ───────────────────────────
function TraditionalPanel({ isInView }: { isInView: boolean }) {
  const steps = [
    { label: 'Input', sub: '(email)', color: '#A1A1AA' },
    { label: 'Hand-Written Rules', sub: 'if subject="FREE"...', color: '#FBBF24' },
    { label: 'Output', sub: '(spam/not spam)', color: '#A1A1AA' },
  ]

  return (
    <div className="flex flex-col items-center">
      <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-4">
        Traditional Programming
      </p>

      <div className="flex items-center gap-3">
        {steps.map((step, i) => (
          <motion.div key={step.label} className="flex items-center gap-3">
            <motion.div
              className="px-3 py-2 rounded-lg border text-center min-w-[80px]"
              style={{
                backgroundColor: `${step.color}10`,
                borderColor: `${step.color}30`,
              }}
              initial={{ opacity: 0, y: 10 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: i * 0.15, duration: 0.4 }}
            >
              <p className="text-[10px] font-mono font-medium" style={{ color: step.color }}>
                {step.label}
              </p>
              <p className="text-[8px] text-text-tertiary mt-0.5">{step.sub}</p>
            </motion.div>
            {i < steps.length - 1 && (
              <motion.span
                className="text-text-tertiary text-xs"
                initial={{ opacity: 0 }}
                animate={isInView ? { opacity: 0.6 } : {}}
                transition={{ delay: i * 0.15 + 0.1 }}
              >
                →
              </motion.span>
            )}
          </motion.div>
        ))}
      </div>

      {/* Failure case */}
      <motion.div
        className="mt-3 px-3 py-1.5 rounded-md border border-red-500/20 bg-red-500/5"
        initial={{ opacity: 0 }}
        animate={isInView ? { opacity: 1 } : {}}
        transition={{ delay: 0.6 }}
      >
        <p className="text-[9px] text-red-400 text-center">
          New spam pattern? Rules break. You rewrite them by hand.
        </p>
      </motion.div>
    </div>
  )
}

function MLPanel({ isInView }: { isInView: boolean }) {
  const steps = [
    { label: 'Input + Labels', sub: '(10,000 emails)', color: '#6366F1' },
    { label: 'Learning Algorithm', sub: 'finds patterns', color: '#6366F1' },
    { label: 'Model', sub: '(learned rules)', color: '#34D399' },
    { label: 'Output', sub: '(prediction)', color: '#34D399' },
  ]

  return (
    <div className="flex flex-col items-center">
      <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-4">
        Machine Learning
      </p>

      <div className="flex items-center gap-2">
        {steps.map((step, i) => (
          <motion.div key={step.label} className="flex items-center gap-2">
            <motion.div
              className="px-3 py-2 rounded-lg border text-center min-w-[75px]"
              style={{
                backgroundColor: `${step.color}10`,
                borderColor: `${step.color}30`,
              }}
              initial={{ opacity: 0, y: 10 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.3 + i * 0.15, duration: 0.4 }}
            >
              <p className="text-[10px] font-mono font-medium" style={{ color: step.color }}>
                {step.label}
              </p>
              <p className="text-[8px] text-text-tertiary mt-0.5">{step.sub}</p>
            </motion.div>
            {i < steps.length - 1 && (
              <motion.span
                className="text-text-tertiary text-xs"
                initial={{ opacity: 0 }}
                animate={isInView ? { opacity: 0.6 } : {}}
                transition={{ delay: 0.3 + i * 0.15 + 0.1 }}
              >
                →
              </motion.span>
            )}
          </motion.div>
        ))}
      </div>

      {/* Success case */}
      <motion.div
        className="mt-3 px-3 py-1.5 rounded-md border border-emerald-500/20 bg-emerald-500/5"
        initial={{ opacity: 0 }}
        animate={isInView ? { opacity: 1 } : {}}
        transition={{ delay: 0.9 }}
      >
        <p className="text-[9px] text-emerald-400 text-center">
          New pattern? Feed new examples. The model adapts on its own.
        </p>
      </motion.div>
    </div>
  )
}

// ── Main Section ──────────────────────────────────────────────────────
export function WhatIsMLSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="what-is-ml" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">
          What Is Machine Learning?
        </h2>
        <p className="text-base lg:text-lg text-text-secondary mt-2 max-w-2xl leading-relaxed">
          Teaching computers to find patterns in data instead of programming rules by hand.
        </p>

        {/* Intuition text */}
        <div className="mt-10">
          <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
            The Intuition
          </h4>
          <div className="max-w-2xl space-y-3">
            <p className="text-text-secondary leading-relaxed">
              Imagine you want to build a program that identifies whether an email is spam. The
              traditional approach: you sit down and write hundreds of if-then rules.
              &quot;If the subject line contains FREE, mark as spam. If the sender is unknown and there
              are more than 3 links, mark as spam.&quot; You would be writing rules forever, and spammers
              would adapt faster than you can keep up.
            </p>
            <p className="text-text-secondary leading-relaxed">
              The machine learning approach: you give the computer 10,000 emails that humans have
              already labeled as spam or not spam. The computer finds the patterns itself. It figures out
              which words, which senders, which structures tend to appear in spam. And when spammers
              change tactics, you just feed in new labeled examples and the model adapts.
            </p>
            <p className="text-text-secondary leading-relaxed">
              That is the core idea. Instead of programming rules, you provide data and let the
              algorithm learn the rules on its own.
            </p>
          </div>
        </div>

        {/* Rules vs Learning side-by-side */}
        <div className="mt-10">
          <GlassCard className="p-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <TraditionalPanel isInView={isInView} />
              <MLPanel isInView={isInView} />
            </div>
          </GlassCard>
        </div>

        {/* Supervised vs Unsupervised */}
        <div className="mt-14">
          <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
            Two Ways to Learn
          </h4>
          <div className="max-w-2xl space-y-3 mb-8">
            <p className="text-text-secondary leading-relaxed">
              There are two major paradigms in machine learning.
              In <strong className="text-text-primary">supervised learning</strong>, you give the model
              both the data and the correct answers. The model learns to predict answers for new data.
            </p>
            <p className="text-text-secondary leading-relaxed">
              In <strong className="text-text-primary">unsupervised learning</strong>, you give the
              model just the data. No answers. The model finds structure on its own, discovering
              patterns and groupings you did not explicitly tell it about.
            </p>
          </div>

          <SupervisedVsUnsupervisedViz />
        </div>

        {/* Site navigation context */}
        <div className="mt-10">
          <GlassCard className="p-5">
            <p className="text-sm text-text-secondary leading-relaxed max-w-2xl">
              The <strong className="text-text-primary">Clustering</strong> page covers unsupervised
              learning. The <strong className="text-text-primary">Regression</strong>,{' '}
              <strong className="text-text-primary">Trees</strong>,{' '}
              <strong className="text-text-primary">Neural Networks</strong>, and{' '}
              <strong className="text-text-primary">Transformers</strong> pages cover supervised
              learning. This Basics page covers the shared foundations that all of them depend on.
            </p>
          </GlassCard>
        </div>
      </motion.div>
    </section>
  )
}
