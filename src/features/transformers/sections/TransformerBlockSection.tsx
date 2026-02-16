import { ModelSection } from '../../../components/ui/ModelSection'
import { TransformerBlockDiagram } from '../visualizations/TransformerBlockDiagram'
import { TransformerBlockMath } from '../content/transformerBlockMath'

export function TransformerBlockSection() {
  return (
    <ModelSection
      id="transformer-block"
      title="The Transformer Block"
      subtitle="Attention plus a feedforward network, wrapped in residual connections and normalization."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            A single Transformer block has two main stages. First,
            the <strong className="text-text-primary">attention layer</strong> lets tokens
            communicate. Each token gathers information from every other token based on relevance.
            Think of this as the "communication step."
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            Second, the <strong className="text-text-primary">feed-forward network (FFN)</strong> processes
            what each token gathered. It applies the same two-layer neural network independently to
            every token position. Think of this as the "thinking step." The FFN expands the
            representation to a larger dimension (typically 4x), applies a non-linearity, then
            projects back down.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            Two critical ingredients make deep stacking possible. <strong className="text-text-primary">Residual
            connections</strong> add the input of each sub-layer back to its output, creating a
            shortcut path for gradients. <strong className="text-text-primary">Layer normalization</strong> stabilizes
            the scale of activations. Without these, a 96-layer model like GPT-3 would be
            impossible to train. Stack N of these blocks and you have a Transformer.
          </p>
        </div>
      }
      mechanism={<TransformerBlockDiagram />}
      math={<TransformerBlockMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>The fundamental repeating unit of all Transformers</li>
                <li>Residual connections enable training very deep networks</li>
                <li>LayerNorm stabilizes activations across layers</li>
                <li>Simple to stack: more blocks = more capacity</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Each block adds significant parameters (BERT-base N=12, GPT-3 N=96)</li>
                <li>Memory usage grows linearly with the number of blocks</li>
                <li>FFN is the largest parameter contributor in each block</li>
                <li>Diminishing returns from adding more blocks without scaling other dimensions</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
