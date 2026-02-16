import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function OverfittingMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Polynomial Model</p>
        <p className="text-sm text-text-secondary mb-3">
          A polynomial of degree <Eq tex="p" /> is a flexible function with <Eq tex="p + 1" /> parameters:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_p x^p" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Training Loss</p>
        <p className="text-sm text-text-secondary mb-3">
          Measures how well the model fits the data it learned from:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="L_{\text{train}} = \frac{1}{n_{\text{train}}} \sum_{i \in \text{train}} (y_i - \hat{y}_i)^2" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Test Loss</p>
        <p className="text-sm text-text-secondary mb-3">
          Measures how well the model predicts data it has never seen:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="L_{\text{test}} = \frac{1}{n_{\text{test}}} \sum_{i \in \text{test}} (y_i - \hat{y}_i)^2" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Training loss measures how well the model fits the data it learned from. Test loss measures
          how well it predicts data it has never seen. When training loss is low but test loss is high,
          the model is overfitting. The test loss is the number that actually matters.
        </p>
      </div>
    </div>
  )
}
