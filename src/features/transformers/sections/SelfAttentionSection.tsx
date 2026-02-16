import { ModelSection } from '../../../components/ui/ModelSection'
import { SelfAttentionWalkthrough } from '../visualizations/SelfAttentionWalkthrough'
import { SelfAttentionMath } from '../content/selfAttentionMath'

export function SelfAttentionSection() {
  return (
    <ModelSection
      id="self-attention"
      title="Self-Attention"
      subtitle="Each word asks 'which other words are relevant to me?' and builds a weighted mix of their information."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Self-attention is the heart of the Transformer. Here is the intuition: for every word
            in a sentence, we want to figure out which other words are most relevant to it, and
            then blend their information together.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            We do this with three projections of each word,
            called <strong className="text-text-primary">Query</strong>, <strong className="text-text-primary">Key</strong>,
            and <strong className="text-text-primary">Value</strong>. Think of it this way: the
            Query is "what am I looking for?", the Key is "what do I contain?", and the Value is
            "what information do I carry?" Each word's Query is compared against every other word's
            Key to produce a relevance score. High scores mean high relevance.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            Those scores become weights (via softmax, so they sum to 1), and the output for each
            word is a weighted sum of all the Value vectors. The result: each word's output is an
            information-rich blend, where the most relevant words contributed the most. The word
            "it" in "The animal didn't cross the street because it was too tired" can directly
            attend to "animal" to resolve the reference.
          </p>
        </div>
      }
      mechanism={<SelfAttentionWalkthrough />}
      math={<SelfAttentionMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Captures long-range dependencies in a single step</li>
                <li>Fully parallelizable across all positions</li>
                <li>Attention weights are interpretable (you can see what the model focuses on)</li>
                <li>Core mechanism of every modern language model</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>O(n^2) in sequence length: cost grows quadratically with input size</li>
                <li>No inherent notion of position (requires positional encoding)</li>
                <li>Memory-intensive for very long sequences</li>
                <li>Single-head attention can only capture one type of relationship at a time</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
