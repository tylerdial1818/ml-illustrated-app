import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function RidgeMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Loss function</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Closed-form solution</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\boldsymbol{\beta} = (X^T X + \alpha I)^{-1} X^T \mathbf{y}" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          The <Eq tex="\alpha \sum \beta_j^2" /> term penalizes big coefficients. Larger α = more penalty =
          smoother fit. The <Eq tex="\alpha I" /> added to <Eq tex="X^T X" /> is what makes this "ridge" —
          it adds a constant ridge along the diagonal, ensuring the matrix is always invertible.
        </p>
      </div>
    </div>
  )
}
