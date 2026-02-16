import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function MultiHeadAttentionMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Single Attention Head</p>
        <p className="text-sm text-text-secondary mb-3">
          Each head runs its own attention with its own learned projections:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{head}_i = \text{Attention}(XW_Q^i,\; XW_K^i,\; XW_V^i)" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Multi-Head Attention</p>
        <p className="text-sm text-text-secondary mb-3">
          Concatenate all heads and project back to the model dimension:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          <Eq tex="W_O \in \mathbb{R}^{hd_v \times d_{\text{model}}}" /> is a learned output projection
          that combines information from all heads.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Head Dimensions</p>
        <p className="text-sm text-text-secondary mb-3">
          Each head operates on a smaller slice of the total dimension:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="d_k = d_v = \frac{d_{\text{model}}}{h}" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          Example: if <Eq tex="d_{\text{model}} = 512" /> and <Eq tex="h = 8" />,
          then <Eq tex="d_k = d_v = 64" /> per head.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Each head has its own Q, K, V projection matrices. The head dimension <Eq tex="d_k" /> is
          smaller (<Eq tex="d_{\text{model}} / h" />), so total computation is roughly the same as one
          full-sized attention. The key insight: multiple smaller heads learn diverse relationship
          patterns that a single large head would miss.
        </p>
      </div>
    </div>
  )
}
