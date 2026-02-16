import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function GradientDescentMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">The Update Rule</p>
        <p className="text-sm text-text-secondary mb-3">
          At each step, we move the parameters in the direction that decreases the loss the fastest:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          Where <Eq tex="\theta" /> represents our parameters (slope, intercept),{' '}
          <Eq tex="\alpha" /> is the learning rate, and <Eq tex="\nabla L" /> is the gradient of the loss.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">MSE Gradient for Linear Regression</p>
        <p className="text-sm text-text-secondary mb-3">
          For a linear model <Eq tex="\hat{y} = \beta_1 x + \beta_0" />, the partial derivatives of MSE are:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-3 text-center">
          <Eq tex="\frac{\partial L}{\partial \beta_1} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i) \cdot x_i" display />
          <Eq tex="\frac{\partial L}{\partial \beta_0} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Variants</p>
        <div className="space-y-3">
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-sm text-text-primary font-medium mb-1">Batch Gradient Descent</p>
            <p className="text-sm text-text-secondary">
              Uses all <Eq tex="n" /> training examples to compute the gradient. Stable but slow for large datasets.
            </p>
          </div>
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-sm text-text-primary font-medium mb-1">Stochastic Gradient Descent (SGD)</p>
            <p className="text-sm text-text-secondary">
              Uses a single random example per step. Noisy but fast. The noise can help escape local minima.
            </p>
          </div>
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-sm text-text-primary font-medium mb-1">Mini-Batch</p>
            <p className="text-sm text-text-secondary">
              Uses a small random subset of size <Eq tex="B" />. A practical compromise between batch and SGD.
            </p>
          </div>
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          The learning rate <Eq tex="\alpha" /> is the most important hyperparameter. Too large and the
          parameters overshoot the minimum. Too small and convergence takes forever. In practice, adaptive
          methods like Adam adjust the learning rate automatically per parameter.
        </p>
      </div>
    </div>
  )
}
