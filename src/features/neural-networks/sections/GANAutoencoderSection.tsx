import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import katex from 'katex'
import { GlassCard } from '../../../components/ui/GlassCard'
import { GANAutoencoderViz } from '../visualizations/GANAutoencoderViz'
import { COLORS } from '../../../types'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function GANAutoencoderSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="gan-autoencoder" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">
          GANs & Autoencoders
        </h2>
        <p className="text-base lg:text-lg text-text-secondary mt-2 max-w-2xl leading-relaxed">
          Generative architectures that learn to create new data or discover compressed representations.
        </p>

        <div className="mt-10 space-y-8">
          {/* Autoencoder */}
          <GlassCard className="p-8">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS.clusters[4] }} />
              <h3 className="text-lg font-semibold text-text-primary">Autoencoder</h3>
            </div>
            <p className="text-text-secondary leading-relaxed max-w-2xl mb-6">
              An autoencoder learns to compress data into a small bottleneck and then reconstruct it.
              The encoder squeezes high-dimensional input into a compact latent representation; the
              decoder expands it back. If the reconstruction is good, the bottleneck has captured the
              essential structure of the data.
            </p>

            {/* Architecture diagram */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 text-sm text-text-secondary mb-6">
              <div className="flex flex-col items-center gap-1">
                <div className="px-4 py-3 bg-obsidian-surface border border-obsidian-border rounded-lg">
                  <span className="text-text-primary font-medium">Input</span>
                </div>
                <span className="text-xs text-text-tertiary">High-dim</span>
              </div>
              <span className="text-text-tertiary text-lg">&rarr;</span>
              <div className="flex flex-col items-center gap-1">
                <div className="px-4 py-3 bg-obsidian-surface border border-obsidian-border rounded-lg" style={{ borderColor: COLORS.clusters[4] + '40' }}>
                  <span className="text-text-primary font-medium">Encoder</span>
                </div>
                <span className="text-xs text-text-tertiary">Compress</span>
              </div>
              <span className="text-text-tertiary text-lg">&rarr;</span>
              <div className="flex flex-col items-center gap-1">
                <div className="px-3 py-3 bg-obsidian-surface border-2 rounded-lg" style={{ borderColor: COLORS.clusters[4] }}>
                  <span className="font-medium" style={{ color: COLORS.clusters[4] }}>z</span>
                </div>
                <span className="text-xs text-text-tertiary">Bottleneck</span>
              </div>
              <span className="text-text-tertiary text-lg">&rarr;</span>
              <div className="flex flex-col items-center gap-1">
                <div className="px-4 py-3 bg-obsidian-surface border border-obsidian-border rounded-lg" style={{ borderColor: COLORS.clusters[4] + '40' }}>
                  <span className="text-text-primary font-medium">Decoder</span>
                </div>
                <span className="text-xs text-text-tertiary">Reconstruct</span>
              </div>
              <span className="text-text-tertiary text-lg">&rarr;</span>
              <div className="flex flex-col items-center gap-1">
                <div className="px-4 py-3 bg-obsidian-surface border border-obsidian-border rounded-lg">
                  <span className="text-text-primary font-medium">Output</span>
                </div>
                <span className="text-xs text-text-tertiary">High-dim</span>
              </div>
            </div>

            <div className="text-sm text-text-tertiary max-w-xl">
              <p>
                <strong className="text-text-secondary">Use cases:</strong> dimensionality reduction, denoising,
                anomaly detection, learning latent representations for downstream tasks.
              </p>
            </div>
          </GlassCard>

          {/* GAN */}
          <GlassCard className="p-8">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS.clusters[5] }} />
              <h3 className="text-lg font-semibold text-text-primary">Generative Adversarial Network (GAN)</h3>
            </div>
            <p className="text-text-secondary leading-relaxed max-w-2xl mb-6">
              A GAN pits two networks against each other. The <strong className="text-text-primary">generator</strong>{' '}
              creates fake data from random noise, trying to fool the discriminator. The{' '}
              <strong className="text-text-primary">discriminator</strong> tries to tell real data from fakes.
              As they train together, the generator gets better at producing realistic data, and the
              discriminator gets better at detecting fakes, until the generator's output is indistinguishable
              from real data.
            </p>

            {/* Architecture diagram */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 text-sm text-text-secondary mb-6">
              <div className="flex flex-col items-center gap-1">
                <div className="px-4 py-3 bg-obsidian-surface border border-obsidian-border rounded-lg">
                  <span className="text-text-primary font-medium">Noise z</span>
                </div>
                <span className="text-xs text-text-tertiary">Random</span>
              </div>
              <span className="text-text-tertiary text-lg">&rarr;</span>
              <div className="flex flex-col items-center gap-1">
                <div className="px-4 py-3 bg-obsidian-surface border border-obsidian-border rounded-lg" style={{ borderColor: COLORS.clusters[5] + '40' }}>
                  <span className="text-text-primary font-medium">Generator</span>
                </div>
                <span className="text-xs text-text-tertiary">Creates fakes</span>
              </div>
              <span className="text-text-tertiary text-lg">&rarr;</span>
              <div className="flex flex-col items-center gap-1">
                <div className="px-4 py-3 bg-obsidian-surface border border-obsidian-border rounded-lg">
                  <span className="text-text-primary font-medium">Fake data</span>
                </div>
                <span className="text-xs text-text-tertiary">Generated</span>
              </div>
              <span className="text-text-tertiary text-lg">&rarr;</span>
              <div className="flex flex-col items-center gap-1">
                <div className="px-4 py-3 bg-obsidian-surface border-2 rounded-lg" style={{ borderColor: COLORS.clusters[5] }}>
                  <span className="font-medium" style={{ color: COLORS.clusters[5] }}>Discriminator</span>
                </div>
                <span className="text-xs text-text-tertiary">Real or fake?</span>
              </div>
            </div>

            <div className="text-sm text-text-tertiary max-w-xl">
              <p>
                <strong className="text-text-secondary">Use cases:</strong> image generation, style transfer,
                data augmentation, super-resolution, synthetic data for privacy-sensitive domains.
              </p>
            </div>
          </GlassCard>

          {/* Math section */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
              The Math
            </h4>
            <div className="space-y-6 max-w-2xl">
              <div>
                <p className="text-sm font-medium text-text-primary mb-2">Autoencoder reconstruction loss</p>
                <p className="text-sm text-text-secondary mb-3">
                  The autoencoder minimizes the difference between input and reconstruction:
                </p>
                <div className="bg-obsidian-surface rounded-lg p-4 text-center">
                  <Eq tex="\mathcal{L}_{\text{AE}} = \frac{1}{N}\sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2" display />
                </div>
              </div>

              <div>
                <p className="text-sm font-medium text-text-primary mb-2">GAN minimax objective</p>
                <p className="text-sm text-text-secondary mb-3">
                  The generator and discriminator play a two-player game:
                </p>
                <div className="bg-obsidian-surface rounded-lg p-4 text-center">
                  <Eq tex="\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\ln D(x)] + \mathbb{E}_{z \sim p_z}[\ln(1 - D(G(z)))]" display />
                </div>
              </div>

              <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
                <p>
                  <Eq tex="D(x)" /> is the discriminator's estimate that <Eq tex="x" /> is real.{' '}
                  <Eq tex="G(z)" /> is the generator's output from noise <Eq tex="z" />.
                  The discriminator wants to maximize its accuracy; the generator wants to minimize it.
                  At equilibrium, <Eq tex="D(x) = 0.5" /> everywhere. The discriminator can't tell
                  real from fake.
                </p>
              </div>
            </div>
          </div>

          {/* Interactive Visualization */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
              Interactive Exploration
            </h4>
            <GANAutoencoderViz />
          </div>
        </div>
      </motion.div>
    </section>
  )
}
