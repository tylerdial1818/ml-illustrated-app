import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function ElasticNetMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Loss function</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \left[ (1 - l_1) \sum \beta_j^2 + l_1 \sum |\beta_j| \right]" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="l_1" /> is the mixing dial. Turn it up (toward 1) for more feature selection (Lasso).
          Turn it down (toward 0) for more shrinkage (Ridge). The constraint shape you see morphs from a
          diamond to a circle as <Eq tex="l_1" /> goes from 1 to 0.
        </p>
      </div>
    </div>
  )
}
