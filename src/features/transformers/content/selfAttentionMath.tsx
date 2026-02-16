import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function SelfAttentionMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Linear Projections</p>
        <p className="text-sm text-text-secondary mb-3">
          We project each input embedding into three separate spaces:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="Q = XW_Q, \quad K = XW_K, \quad V = XW_V" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          <Eq tex="X" /> is the input matrix (one row per token). <Eq tex="W_Q, W_K, W_V" /> are
          learned weight matrices that produce the Query, Key, and Value representations.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Attention Scores</p>
        <p className="text-sm text-text-secondary mb-3">
          We measure how relevant each key is to each query using a scaled dot product:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{score}(Q, K) = \frac{QK^T}{\sqrt{d_k}}" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          The dot product <Eq tex="QK^T" /> measures similarity between queries and keys.
          Dividing by <Eq tex="\sqrt{d_k}" /> keeps the scale manageable so softmax doesn't saturate.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Attention Weights</p>
        <p className="text-sm text-text-secondary mb-3">
          Softmax turns raw scores into probabilities that sum to 1:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\alpha = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Scaled Dot-Product Attention</p>
        <p className="text-sm text-text-secondary mb-3">
          The full attention operation in one formula:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          Each output row is a weighted combination of Value vectors, where the weights come from
          the Query-Key similarity. High similarity means more of that token's Value gets mixed in.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Q, K, V are just the input embeddings multiplied by learned weight matrices. The dot
          product <Eq tex="QK^T" /> measures similarity between queries and keys. <Eq tex="\sqrt{d_k}" /> keeps
          the scale manageable. Softmax turns raw scores into probabilities that sum to 1.
        </p>
      </div>
    </div>
  )
}
