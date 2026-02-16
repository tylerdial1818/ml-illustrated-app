import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function TransformerBlockMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Attention Sub-Layer</p>
        <p className="text-sm text-text-secondary mb-3">
          Multi-head attention with a residual connection and layer normalization:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="z = \text{LayerNorm}(x + \text{MultiHeadAttention}(x))" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          The <Eq tex="+ \, x" /> is the residual (skip) connection. It lets gradients flow directly
          through the network, making deep stacking possible.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Feed-Forward Sub-Layer</p>
        <p className="text-sm text-text-secondary mb-3">
          A position-wise feed-forward network, also with a residual connection:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{output} = \text{LayerNorm}(z + \text{FFN}(z))" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Feed-Forward Network (FFN)</p>
        <p className="text-sm text-text-secondary mb-3">
          Two linear transformations with a ReLU activation in between:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{FFN}(z) = \text{ReLU}(zW_1 + b_1)W_2 + b_2" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          <Eq tex="W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}" /> expands the
          dimension, <Eq tex="W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}" /> projects
          it back. Typically <Eq tex="d_{\text{ff}} = 4 \times d_{\text{model}}" />.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          The feedforward network expands the representation
          (<Eq tex="d_{\text{model}} \to d_{\text{ff}}" />, typically 4x), applies a non-linearity,
          then projects back. This expansion gives room for complex transformations. Think of attention
          as the "communication step" (tokens share information) and FFN as the "thinking step" (each
          token processes what it gathered).
        </p>
      </div>
    </div>
  )
}
