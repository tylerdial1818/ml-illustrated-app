import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function LinearMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Model</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="y = \beta_0 + \beta_1 x + \varepsilon" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Loss function</p>
        <p className="text-sm text-text-secondary mb-3">Minimize the sum of squared residuals:</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Closed-form solution</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\boldsymbol{\beta} = (X^T X)^{-1} X^T \mathbf{y}" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="\beta_1" /> is the slope you see. <Eq tex="\varepsilon" /> is the distance each point
          sits from the line. The closed-form solution gives us the exact best line without any iterative
          optimization.
        </p>
      </div>
    </div>
  )
}
