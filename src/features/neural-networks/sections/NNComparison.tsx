import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

const ARCHITECTURES = [
  {
    name: 'Perceptron',
    color: COLORS.clusters[0],
    dataType: 'Tabular (linearly separable)',
    innovation: 'Weighted sum + activation = decision boundary',
    useCases: 'Binary classification, logic gates (AND, OR)',
    strengths: 'Simple, fast, interpretable',
    limitations: 'Linear boundaries only',
  },
  {
    name: 'MLP',
    color: COLORS.clusters[1],
    dataType: 'Tabular, structured',
    innovation: 'Hidden layers + backpropagation',
    useCases: 'Classification, regression, function approximation',
    strengths: 'Universal approximator, flexible',
    limitations: 'No spatial/temporal structure awareness',
  },
  {
    name: 'CNN',
    color: COLORS.clusters[2],
    dataType: 'Images, spatial data',
    innovation: 'Shared filters, local connectivity',
    useCases: 'Image classification, object detection, segmentation',
    strengths: 'Translation invariant, parameter efficient for spatial data',
    limitations: 'Needs lots of data, fixed input size',
  },
  {
    name: 'RNN / LSTM',
    color: COLORS.clusters[3],
    dataType: 'Sequences, time series',
    innovation: 'Recurrent connections, gated memory (LSTM)',
    useCases: 'Language modeling, speech recognition, time series',
    strengths: 'Variable-length input, temporal memory',
    limitations: 'Hard to parallelize, superseded by Transformers',
  },
  {
    name: 'Autoencoder',
    color: COLORS.clusters[4],
    dataType: 'Any (unsupervised)',
    innovation: 'Bottleneck forces compressed representation',
    useCases: 'Dimensionality reduction, denoising, anomaly detection',
    strengths: 'Learns latent structure, no labels needed',
    limitations: 'Reconstruction quality depends on bottleneck size',
  },
  {
    name: 'GAN',
    color: COLORS.clusters[5],
    dataType: 'Any (generative)',
    innovation: 'Adversarial training: generator vs discriminator',
    useCases: 'Image generation, style transfer, data augmentation',
    strengths: 'Produces realistic synthetic data',
    limitations: 'Hard to train, mode collapse',
  },
]

const TABLE_ROWS = [
  { key: 'dataType' as const, label: 'Data Type' },
  { key: 'innovation' as const, label: 'Key Innovation' },
  { key: 'useCases' as const, label: 'Typical Use Cases' },
  { key: 'strengths' as const, label: 'Strengths' },
  { key: 'limitations' as const, label: 'Limitations' },
]

export function NNComparison() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="nn-comparison" ref={ref} className="py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-3xl font-semibold text-text-primary">Comparison</h2>
        <p className="mt-2 text-lg text-text-secondary max-w-2xl">
          Six architectures, each designed for different types of data and problems.
          Here's how they stack up.
        </p>

        {/* Comparison table */}
        <GlassCard className="mt-8 p-8 overflow-x-auto">
          <table className="w-full text-sm min-w-[700px]">
            <thead>
              <tr>
                <th className="text-left text-text-tertiary font-medium pb-4 pr-4 w-36" />
                {ARCHITECTURES.map((arch) => (
                  <th key={arch.name} className="text-left pb-4 px-2">
                    <div className="flex items-center gap-1.5">
                      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: arch.color }} />
                      <span className="text-text-primary font-medium text-xs">{arch.name}</span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TABLE_ROWS.map((row) => (
                <tr key={row.key} className="border-t border-obsidian-border">
                  <td className="py-3 pr-4 text-text-tertiary font-medium text-xs align-top">
                    {row.label}
                  </td>
                  {ARCHITECTURES.map((arch) => (
                    <td key={arch.name} className="py-3 px-2 text-text-secondary text-xs align-top">
                      {arch[row.key]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </GlassCard>

        {/* Selection flowchart */}
        <GlassCard className="mt-6 p-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Which architecture should you use?</h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="font-bold mt-0.5" style={{ color: COLORS.clusters[0] }}>?</span>
              <div>
                <p className="text-text-primary font-medium">What kind of data do you have?</p>
                <p className="text-text-secondary">
                  <strong>Images/spatial:</strong> CNN.{' '}
                  <strong>Sequences/time series:</strong> RNN/LSTM.{' '}
                  <strong>Tabular:</strong> MLP.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="font-bold mt-0.5" style={{ color: COLORS.clusters[1] }}>?</span>
              <div>
                <p className="text-text-primary font-medium">Is the problem linearly separable?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> A Perceptron or logistic regression is sufficient.{' '}
                  <strong>No:</strong> You need at least an MLP with hidden layers.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="font-bold mt-0.5" style={{ color: COLORS.clusters[2] }}>?</span>
              <div>
                <p className="text-text-primary font-medium">Do you need to generate new data?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> GAN (realistic samples) or Autoencoder/VAE (structured latent space).{' '}
                  <strong>No:</strong> Use a discriminative model (MLP, CNN, RNN).
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="font-bold mt-0.5" style={{ color: COLORS.clusters[3] }}>?</span>
              <div>
                <p className="text-text-primary font-medium">Do you have labels?</p>
                <p className="text-text-secondary">
                  <strong>Yes:</strong> Supervised models (Perceptron, MLP, CNN, RNN).{' '}
                  <strong>No:</strong> Autoencoders for representation learning, GANs for generation.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-lg bg-obsidian-surface">
              <span className="font-bold mt-0.5" style={{ color: COLORS.clusters[4] }}>?</span>
              <div>
                <p className="text-text-primary font-medium">Do you need to handle very long sequences?</p>
                <p className="text-text-secondary">
                  <strong>Short/medium sequences:</strong> RNN/LSTM works well.{' '}
                  <strong>Long sequences:</strong> Consider Transformers (covered in the next section).
                </p>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </section>
  )
}
