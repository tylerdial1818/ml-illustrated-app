import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'
import { RNNvsAttentionComparison } from '../visualizations/RNNvsAttentionComparison'

export function TransformersOverview() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.2, once: true })

  return (
    <section id="transformers-overview" ref={ref} className="py-16 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h4 className="text-xs font-semibold uppercase tracking-[0.1em] text-text-tertiary mb-6">
          The Idea
        </h4>
        <h2 className="text-2xl font-semibold text-text-primary mb-4">
          From Sequential to Parallel
        </h2>
        <p className="text-text-secondary max-w-2xl leading-relaxed">
          Before Transformers, we processed language with recurrent neural networks (RNNs). An RNN
          reads one word at a time, left to right, passing a hidden state forward like a game of
          telephone. This creates a bottleneck: all information about the beginning of a sentence
          has to survive compression into a single vector. Long-range dependencies get lost.
        </p>
        <p className="mt-3 text-text-secondary max-w-2xl leading-relaxed">
          The Transformer takes a radically different approach. Instead of reading sequentially, it
          lets every word look at every other word directly. Want to know what "it" refers to in a
          long paragraph? Just look. No need to pass a message through dozens of intermediate steps.
          This is the attention mechanism, and it changed everything.
        </p>
        <p className="mt-3 text-text-secondary max-w-2xl leading-relaxed">
          The original paper, "Attention Is All You Need" (Vaswani et al., 2017), proposed an
          architecture built entirely on attention. No recurrence, no convolutions. It was faster
          to train, better at capturing long-range patterns, and fully parallelizable on modern
          hardware. Every major language model since, from BERT to GPT-4, is built on this foundation.
        </p>

        {/* RNN vs Attention visualization */}
        <div className="mt-8">
          <RNNvsAttentionComparison />
        </div>

        {/* Key text */}
        <div className="mt-8 border-l-2 border-accent pl-5 max-w-2xl">
          <p className="text-base text-text-primary leading-relaxed font-medium">
            RNNs process words one at a time and compress everything into a fixed-size hidden state.
            Transformers let every word look at every other word directly. No bottleneck. No forgetting.
          </p>
        </div>

        {/* Building Blocks */}
        <div className="mt-10 grid grid-cols-1 sm:grid-cols-3 gap-3">
          <GlassCard className="p-5">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.clusters[0] }} />
              <span className="text-sm font-medium text-text-primary">Tokenization &amp; Embedding</span>
            </div>
            <p className="text-xs text-text-secondary">
              Convert raw text into numerical vectors. Break words into subword tokens, then map each
              token to a dense vector that captures meaning. This is where text becomes math.
            </p>
          </GlassCard>
          <GlassCard className="p-5">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.clusters[1] }} />
              <span className="text-sm font-medium text-text-primary">Attention</span>
            </div>
            <p className="text-xs text-text-secondary">
              The core mechanism. Each token computes a relevance score with every other token,
              then builds a weighted mix of their information. This is how the model learns context.
            </p>
          </GlassCard>
          <GlassCard className="p-5">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.clusters[2] }} />
              <span className="text-sm font-medium text-text-primary">Architecture</span>
            </div>
            <p className="text-xs text-text-secondary">
              How it all connects: residual connections, layer normalization, feed-forward networks,
              and the encoder/decoder structure. The wiring that makes deep stacking work.
            </p>
          </GlassCard>
        </div>

        {/* Key concept callout */}
        <GlassCard className="mt-6 p-6">
          <p className="text-sm text-text-secondary leading-relaxed">
            <strong className="text-text-primary">Why parallelization matters:</strong> an RNN must
            process token 5 before it can process token 6, because token 6 depends on the hidden
            state from token 5. A Transformer computes all tokens simultaneously. This means
            training on a GPU goes from sequential (slow) to massively parallel (fast). It is the
            reason we can train models on trillions of tokens in a reasonable timeframe.
          </p>
        </GlassCard>
      </motion.div>
    </section>
  )
}
