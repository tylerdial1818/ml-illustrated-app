import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

export function NNOverview() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.2, once: true })

  return (
    <section id="nn-overview" ref={ref} className="py-16 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h4 className="text-xs font-semibold uppercase tracking-[0.1em] text-text-tertiary mb-6">
          From Neuron to Network
        </h4>
        <h2 className="text-2xl font-semibold text-text-primary mb-4">
          What is a Neural Network?
        </h2>
        <p className="text-text-secondary max-w-2xl leading-relaxed">
          A neural network is a function built from layers of simple units called neurons.
          Each neuron takes in numbers, multiplies them by weights, adds them up, and passes
          the result through an activation function. Stack enough of these together and you get
          a system that can learn remarkably complex patterns from data.
        </p>

        {/* Single neuron diagram */}
        <GlassCard className="mt-8 p-8">
          <h3 className="text-sm font-semibold text-text-primary mb-4">A Single Neuron</h3>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 text-sm text-text-secondary">
            <div className="flex flex-col items-center gap-1">
              <div className="px-3 py-2 bg-obsidian-surface border border-obsidian-border rounded-lg">
                <span className="text-text-primary font-mono">x&#8321;, x&#8322;, ...</span>
              </div>
              <span className="text-xs text-text-tertiary">Inputs</span>
            </div>
            <span className="text-text-tertiary text-lg">&rarr;</span>
            <div className="flex flex-col items-center gap-1">
              <div className="px-3 py-2 bg-obsidian-surface border border-obsidian-border rounded-lg">
                <span className="text-text-primary font-mono">&times; w&#8321;, w&#8322;, ...</span>
              </div>
              <span className="text-xs text-text-tertiary">Weights</span>
            </div>
            <span className="text-text-tertiary text-lg">&rarr;</span>
            <div className="flex flex-col items-center gap-1">
              <div className="px-3 py-2 bg-obsidian-surface border border-obsidian-border rounded-lg">
                <span className="text-text-primary font-mono">&Sigma; + b</span>
              </div>
              <span className="text-xs text-text-tertiary">Sum + Bias</span>
            </div>
            <span className="text-text-tertiary text-lg">&rarr;</span>
            <div className="flex flex-col items-center gap-1">
              <div className="px-3 py-2 bg-obsidian-surface border border-obsidian-border rounded-lg">
                <span className="text-text-primary font-mono">&sigma;(z)</span>
              </div>
              <span className="text-xs text-text-tertiary">Activation</span>
            </div>
            <span className="text-text-tertiary text-lg">&rarr;</span>
            <div className="flex flex-col items-center gap-1">
              <div className="px-3 py-2 bg-obsidian-surface border border-obsidian-border rounded-lg">
                <span className="text-text-primary font-mono">&ycirc;</span>
              </div>
              <span className="text-xs text-text-tertiary">Output</span>
            </div>
          </div>
          <p className="mt-4 text-xs text-text-tertiary text-center">
            Every neural network, no matter how deep, is built from copies of this basic unit.
          </p>
        </GlassCard>

        {/* Layers compose into networks */}
        <GlassCard className="mt-4 p-8">
          <h3 className="text-sm font-semibold text-text-primary mb-3">Layers Compose into Networks</h3>
          <p className="text-sm text-text-secondary leading-relaxed max-w-2xl">
            A single neuron can only draw a straight line. But when you arrange neurons into layers
            and connect them, each layer feeding into the next, the network can represent
            increasingly complex functions. The first layer might detect simple features, and deeper
            layers combine them into higher-level abstractions.
          </p>
        </GlassCard>

        {/* Taxonomy */}
        <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {[
            {
              name: 'Perceptron',
              desc: 'One neuron, one decision boundary. The building block of it all.',
              color: COLORS.clusters[0],
            },
            {
              name: 'Multi-Layer Perceptron',
              desc: 'Stack layers to learn non-linear boundaries. The classic feedforward network.',
              color: COLORS.clusters[1],
            },
            {
              name: 'CNN',
              desc: 'Slide filters across spatial data to detect local patterns like edges and textures.',
              color: COLORS.clusters[2],
            },
            {
              name: 'RNN & LSTM',
              desc: 'Process sequences step by step, carrying memory forward through time.',
              color: COLORS.clusters[3],
            },
            {
              name: 'Autoencoder',
              desc: 'Compress data into a bottleneck and reconstruct it, learning efficient representations.',
              color: COLORS.clusters[4],
            },
            {
              name: 'GAN',
              desc: 'Two networks compete: a generator creates fakes, a discriminator spots them.',
              color: COLORS.clusters[5],
            },
          ].map((item) => (
            <GlassCard key={item.name} className="p-5">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-sm font-medium text-text-primary">{item.name}</span>
              </div>
              <p className="text-xs text-text-secondary">{item.desc}</p>
            </GlassCard>
          ))}
        </div>
      </motion.div>
    </section>
  )
}
