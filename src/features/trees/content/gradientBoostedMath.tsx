import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function GradientBoostedMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Ensemble Prediction</p>
        <p className="text-sm text-text-secondary mb-3">
          The model is built additively — each new tree corrects the errors of all previous trees:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          <Eq tex="F_m" /> is the ensemble after <Eq tex="m" /> trees, <Eq tex="h_m" /> is the new
          tree, and <Eq tex="\eta" /> is the learning rate that controls how much each tree contributes.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Residuals (Pseudo-Residuals)</p>
        <p className="text-sm text-text-secondary mb-3">
          Each new tree is fit to the negative gradient of the loss function — the "residuals" that tell
          each tree what to fix:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="r_{im} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          For squared error loss, this simplifies to <Eq tex="r_{im} = y_i - F_{m-1}(x_i)" /> — literally
          the difference between the true value and the current prediction.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Loss Function</p>
        <p className="text-sm text-text-secondary mb-3">
          The algorithm minimizes a differentiable loss function:
        </p>
        <div className="space-y-3">
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-xs text-text-tertiary mb-1">Regression (squared error)</p>
            <Eq tex="L(y, F) = \frac{1}{2}(y - F(x))^2" display />
          </div>
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-xs text-text-tertiary mb-1">Classification (log loss)</p>
            <Eq tex="L(y, F) = -[y \log(\sigma(F)) + (1-y)\log(1 - \sigma(F))]" display />
          </div>
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          The learning rate <Eq tex="\eta" /> (typically 0.01–0.3) controls the trade-off between
          number of trees and contribution per tree. Smaller <Eq tex="\eta" /> requires more trees
          but generally yields better generalization. This is gradient descent in function space —
          each tree takes a small step toward the optimal prediction.
        </p>
      </div>
    </div>
  )
}
