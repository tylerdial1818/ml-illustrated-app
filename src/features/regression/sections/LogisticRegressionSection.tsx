import { ModelSection } from '../../../components/ui/ModelSection'
import { LogisticRegressionViz } from '../visualizations/LogisticRegressionViz'
import { LogisticMath } from '../content/logisticMath'

export function LogisticRegressionSection() {
  return (
    <ModelSection
      id="logistic-regression"
      title="Logistic Regression"
      subtitle="Predict probabilities by squishing linear output through a sigmoid."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Linear regression predicts a number. Logistic regression predicts a{' '}
            <strong className="text-text-primary">probability</strong> — it squishes the output through an
            S-curve (sigmoid) so predictions always land between 0 and 1. The decision boundary is where the
            probability crosses your chosen threshold.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            The gradient field shows the probability landscape: one class dominates on each side, with a
            smooth transition zone in between. Move the threshold slider to see how it trades off precision
            and recall.
          </p>
        </div>
      }
      mechanism={<LogisticRegressionViz />}
      math={<LogisticMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Probabilistic output</li>
                <li>Interpretable coefficients</li>
                <li>Works well for linear boundaries</li>
                <li>Fast training</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Only linear decision boundaries</li>
                <li>Struggles with imbalanced classes</li>
                <li>Assumes independence of features</li>
                <li>Can underfit complex patterns</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
