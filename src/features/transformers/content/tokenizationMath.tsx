import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function TokenizationMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Embedding Lookup</p>
        <p className="text-sm text-text-secondary mb-3">
          Each token is mapped to a dense vector by selecting a row from a learned matrix:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="e(\text{token}) = W_{\text{embed}}[\text{token\_id}]" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          <Eq tex="W_{\text{embed}} \in \mathbb{R}^{V \times d}" /> where <Eq tex="V" /> is the vocabulary
          size and <Eq tex="d" /> is the embedding dimension.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Each token gets a unique ID. That ID selects one row from a learned matrix. The row IS the
          token's vector representation. Nothing fancy here. It is just a table lookup, but the table
          is learned during training so that similar tokens end up with similar vectors.
        </p>
      </div>
    </div>
  )
}
