import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function LassoMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Loss function</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |\beta_j|" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Why L1 produces sparsity</p>
        <p className="text-sm text-text-secondary">
          The absolute value penalty creates a diamond-shaped constraint region. The loss function contours
          (ellipses) tend to first touch the diamond at its corners, which sit on the axes, meaning one
          or more coefficients are exactly zero.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="|\beta_j|" /> instead of <Eq tex="\beta_j^2" />. This sharp corner at zero is why
          coefficients get zeroed out. The L1 norm has "corners" that the optimization naturally gravitates
          toward, unlike the smooth L2 circle.
        </p>
      </div>
    </div>
  )
}
