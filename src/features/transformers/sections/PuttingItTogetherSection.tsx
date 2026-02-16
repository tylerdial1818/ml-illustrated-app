import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { FullPipeline } from '../visualizations/FullPipeline'
import { BERTvsGPT } from '../visualizations/BERTvsGPT'
import { ScaleVisualization } from '../visualizations/ScaleVisualization'

const KEY_CONCEPTS = [
  {
    name: 'Token',
    definition:
      'A subword unit produced by the tokenizer. It is the atomic input to the model. Common words are single tokens; rare words get split into multiple tokens.',
    sectionId: 'tokenization',
  },
  {
    name: 'Embedding',
    definition:
      'A dense vector of real numbers that represents a token. Learned during training so that similar tokens have similar vectors in this high-dimensional space.',
    sectionId: 'tokenization',
  },
  {
    name: 'Positional Encoding',
    definition:
      'A signal added to each embedding to encode its position in the sequence. Without it, the Transformer treats input as an unordered set.',
    sectionId: 'positional-encoding',
  },
  {
    name: 'Query / Key / Value',
    definition:
      'Three projections of each token used in attention. The Query asks "what am I looking for?", the Key says "what do I contain?", and the Value carries "what information do I share?"',
    sectionId: 'self-attention',
  },
  {
    name: 'Attention Score',
    definition:
      'The dot product between a Query and a Key, scaled by the square root of the key dimension. Measures how relevant one token is to another.',
    sectionId: 'self-attention',
  },
  {
    name: 'Softmax',
    definition:
      'A function that converts raw attention scores into a probability distribution summing to 1. Higher scores get exponentially more weight.',
    sectionId: 'self-attention',
  },
  {
    name: 'Multi-Head Attention',
    definition:
      'Running several attention operations in parallel, each with different learned projections. Lets the model attend to different types of relationships simultaneously.',
    sectionId: 'multi-head',
  },
  {
    name: 'Residual Connection',
    definition:
      'A skip connection that adds the input of a sub-layer directly to its output. Prevents signal degradation in deep networks and helps gradients flow during backpropagation.',
    sectionId: 'transformer-block',
  },
  {
    name: 'Layer Normalization',
    definition:
      'Normalizes activations across the feature dimension for each token independently. Stabilizes training and allows higher learning rates.',
    sectionId: 'transformer-block',
  },
  {
    name: 'Feed-Forward Network',
    definition:
      'A two-layer MLP applied independently to each token position. Expands the dimension (typically 4x), applies a non-linearity, then projects back down.',
    sectionId: 'transformer-block',
  },
  {
    name: 'Causal Mask',
    definition:
      'A lower-triangular mask applied in decoder attention that prevents tokens from attending to future positions. Essential for autoregressive generation.',
    sectionId: 'encoder-decoder',
  },
  {
    name: 'Cross-Attention',
    definition:
      'Attention where Queries come from the decoder and Keys/Values come from the encoder. This is how encoder-decoder models pass information from input to output.',
    sectionId: 'encoder-decoder',
  },
]

export function PuttingItTogetherSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="putting-together" ref={ref} className="py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-3xl font-semibold text-text-primary">From Tokens to Intelligence</h2>
        <p className="mt-2 text-lg text-text-secondary max-w-2xl">
          We have covered every piece. Now let's see how they fit together into a complete
          Transformer, compare the major architectures, and appreciate the scale of modern models.
        </p>

        {/* Full pipeline visualization */}
        <div className="mt-10">
          <h3 className="text-lg font-semibold text-text-primary mb-4">End-to-End Forward Pass</h3>
          <p className="text-sm text-text-secondary mb-6 max-w-2xl">
            Text enters as raw characters. The tokenizer breaks it into subword tokens. Each token
            is embedded and given a positional encoding. The sequence passes through N Transformer
            blocks (attention + FFN), and the final output is projected into a prediction.
          </p>
          <FullPipeline />
        </div>

        {/* BERT vs GPT comparison */}
        <div className="mt-16">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Architecture Comparison</h3>
          <p className="text-sm text-text-secondary mb-6 max-w-2xl">
            Same building blocks, different wiring. The choice between encoder, decoder, and
            encoder-decoder determines what your model is good at.
          </p>
          <BERTvsGPT />
        </div>

        {/* Scale visualization */}
        <div className="mt-16">
          <h3 className="text-lg font-semibold text-text-primary mb-4">The Scale of Modern Models</h3>
          <p className="text-sm text-text-secondary mb-6 max-w-2xl">
            Transformers scale remarkably well. More parameters, more data, and more compute
            consistently produce better models. Here is how the numbers have grown.
          </p>
          <ScaleVisualization />
        </div>

        {/* Key Concepts Recap */}
        <div className="mt-16">
          <h3 className="text-lg font-semibold text-text-primary mb-2">Key Concepts Recap</h3>
          <p className="text-sm text-text-secondary mb-6 max-w-2xl">
            Every term you need to understand the Transformer, in two sentences or fewer.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {KEY_CONCEPTS.map((concept) => (
              <GlassCard
                key={concept.name}
                className="p-4 cursor-pointer transition-all hover:border-indigo-500/30 group"
                onClick={() => {
                  const el = document.getElementById(concept.sectionId)
                  if (el) el.scrollIntoView({ behavior: 'smooth' })
                }}
              >
                <div className="flex items-start justify-between gap-2">
                  <h4 className="text-sm font-medium text-text-primary mb-1">{concept.name}</h4>
                  <span className="text-[9px] text-text-tertiary group-hover:text-indigo-400 transition-colors shrink-0">
                    ↑ Jump
                  </span>
                </div>
                <p className="text-xs text-text-secondary leading-relaxed">{concept.definition}</p>
              </GlassCard>
            ))}
          </div>
        </div>
      </motion.div>
    </section>
  )
}
