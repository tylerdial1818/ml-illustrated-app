import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function LogisticMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Sigmoid function</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{where } z = \boldsymbol{\beta}^T \mathbf{x}" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Loss function (binary cross-entropy)</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="L = -\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="z" /> is the linear combination, same as linear regression. <Eq tex="\sigma" /> squishes
          it into a probability between 0 and 1. The loss function heavily penalizes confident wrong
          predictions (predicting 0.99 when the true label is 0).
        </p>
      </div>
    </div>
  )
}
