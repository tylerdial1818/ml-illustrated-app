import { ModelSection } from '../../../components/ui/ModelSection'
import { PositionalEncodingHeatmap } from '../visualizations/PositionalEncodingHeatmap'
import { PositionalEncodingMath } from '../content/positionalEncodingMath'

export function PositionalEncodingSection() {
  return (
    <ModelSection
      id="positional-encoding"
      title="Positional Encoding"
      subtitle="Transformers have no built-in sense of word order. Positional encoding adds that back."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Unlike RNNs, which process words one at a time and naturally know which word came first,
            Transformers process all words simultaneously. That means "the cat sat on the mat" and
            "mat the on sat cat the" would look identical to the model. We need a way to tell the
            model where each word is in the sentence.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            <strong className="text-text-primary">Positional encoding</strong> solves this by
            adding a unique numerical pattern to each position. Think of it like a fingerprint for
            each slot in the sequence. Position 0 gets one pattern, position 1 gets a slightly
            different pattern, and so on. We add this pattern directly to the token embedding, so
            the model receives both "what the word means" and "where it sits" in a single vector.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            The original Transformer uses sine and cosine waves at different frequencies. This
            clever choice means the encoding for position 10 can be expressed as a linear
            combination of position 5's encoding, making it easy for the model to learn relative
            distances between words.
          </p>
        </div>
      }
      mechanism={<PositionalEncodingHeatmap />}
      math={<PositionalEncodingMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Required for any Transformer to understand word order</li>
                <li>Sinusoidal encoding generalizes to sequence lengths not seen during training</li>
                <li>Learned positional embeddings (BERT, GPT) can be slightly more accurate</li>
                <li>RoPE (Rotary Position Embedding) is the modern standard, combining benefits of both</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Learned embeddings are capped at a fixed maximum sequence length</li>
                <li>Sinusoidal patterns can be harder for the model to use than learned ones</li>
                <li>Long sequences still challenge all position encoding schemes</li>
                <li>Adds no learnable parameters in the sinusoidal case</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
