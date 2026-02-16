import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function LossFunctionsMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Mean Squared Error (MSE)</p>
        <p className="text-sm text-text-secondary mb-3">
          Squares each residual, then averages. Large errors get penalized disproportionately.
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Mean Absolute Error (MAE)</p>
        <p className="text-sm text-text-secondary mb-3">
          Takes the absolute value of each residual, then averages. Treats all errors linearly.
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Huber Loss</p>
        <p className="text-sm text-text-secondary mb-3">
          Behaves like MSE for small errors and MAE for large ones. The threshold <Eq tex="\delta" /> controls where it switches.
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq
            tex={String.raw`L_\delta(r) = \begin{cases} \frac{1}{2}r^2 & \text{if } |r| \le \delta \\ \delta(|r| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}`}
            display
          />
        </div>
      </div>

      <div>
        <p className="text-sm text-text-secondary mb-3">
          In all cases, <Eq tex="\hat{y}_i = \beta_0 + \beta_1 x_i" /> is the model prediction (slope times
          feature plus intercept), and <Eq tex="r_i = y_i - \hat{y}_i" /> is the residual.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          MSE is differentiable everywhere, which makes optimization smooth. MAE is not differentiable
          at <Eq tex="r = 0" />, but is more robust to outliers. Huber gives you the best of both worlds
          at the cost of an extra hyperparameter <Eq tex="\delta" />.
        </p>
      </div>
    </div>
  )
}
