import { ModelSection } from '../../../components/ui/ModelSection'
import { RidgeViz } from '../visualizations/RidgeViz'
import { RidgeMath } from '../content/ridgeMath'

export function RidgeSection() {
  return (
    <ModelSection
      id="ridge"
      title="Ridge Regression (L2)"
      subtitle="Shrink all coefficients toward zero to prevent overfitting."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Regular linear regression can go wild — huge coefficients that overfit the noise. Ridge adds a
            penalty: <strong className="text-text-primary">"minimize error, but also keep your coefficients small."</strong>{' '}
            The bigger the penalty (α), the more the coefficients shrink toward zero.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            Watch the coefficient trace plot as you increase α: all coefficients shrink smoothly toward zero,
            but none of them actually reach zero. Ridge shrinks but never selects.
          </p>
        </div>
      }
      mechanism={<RidgeViz />}
      math={<RidgeMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li>Handles multicollinearity well</li>
                <li>Prevents overfitting</li>
                <li>Closed-form solution</li>
                <li>Keeps all features (no sparsity)</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li>Doesn't do feature selection</li>
                <li>All features stay in the model</li>
                <li>Need to tune α</li>
                <li>Less interpretable than sparse models</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
