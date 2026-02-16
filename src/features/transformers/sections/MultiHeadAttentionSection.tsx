import { ModelSection } from '../../../components/ui/ModelSection'
import { MultiHeadGrid } from '../visualizations/MultiHeadGrid'
import { MultiHeadAttentionMath } from '../content/multiHeadAttentionMath'

export function MultiHeadAttentionSection() {
  return (
    <ModelSection
      id="multi-head"
      title="Multi-Head Attention"
      subtitle="Multiple attention patterns in parallel so the model can attend to different types of relationships."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            A single attention head captures one type of relationship. Maybe it learns to connect
            subjects with their verbs. But language is rich with many kinds of relationships:
            syntactic structure, coreference ("it" refers to "animal"), semantic similarity,
            proximity, and more. One set of Q/K/V weights can't capture all of these at once.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            <strong className="text-text-primary">Multi-head attention</strong> solves this by
            running several attention operations in parallel, each with its own learned Q, K, and V
            weight matrices. Each head is free to specialize. One head might focus on syntactic
            dependencies, another on positional relationships, another on semantic similarity.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            After all heads compute their outputs, we concatenate the results and project them back
            to the original dimension. The total computation is roughly the same as single-head
            attention because each head operates on a smaller slice of the embedding dimension.
            We get multiple perspectives at no extra cost.
          </p>
        </div>
      }
      mechanism={<MultiHeadGrid />}
      math={<MultiHeadAttentionMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Captures multiple relationship types simultaneously</li>
                <li>Each head can specialize in different patterns</li>
                <li>Same computational cost as single full-dimension attention</li>
                <li>The standard in virtually all Transformer architectures</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>More heads means smaller per-head dimension, which can limit expressiveness</li>
                <li>Some heads may learn redundant patterns</li>
                <li>Head count is a hyperparameter: typical values are 8 (BERT-base), 12 (BERT-large), 32+ (GPT-3)</li>
                <li>Interpreting what each head learns is still an active research area</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
