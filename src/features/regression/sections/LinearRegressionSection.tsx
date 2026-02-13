import { ModelSection } from '../../../components/ui/ModelSection'
import { LinearRegressionViz } from '../visualizations/LinearRegressionViz'
import { LinearMath } from '../content/linearMath'

export function LinearRegressionSection() {
  return (
    <ModelSection
      id="linear-regression"
      title="Linear Regression (OLS)"
      subtitle="Find the straight line that minimizes total squared error."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Find the straight line that minimizes the total squared distance from every point to the line.
            Squaring means <strong className="text-text-primary">big misses are penalized way more</strong> than
            small ones — a point that's 4 units away costs 16, not 4.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            The red squares you see in the visualization are literal — their area is the squared error for each
            point. OLS minimizes the total area of all those squares.
          </p>
        </div>
      }
      mechanism={<LinearRegressionViz />}
      math={<LinearMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li>Simple and interpretable</li>
                <li>Closed-form solution (fast)</li>
                <li>Good baseline model</li>
                <li>Coefficients have clear meaning</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li>Assumes linear relationship</li>
                <li>Sensitive to outliers</li>
                <li>No regularization (can overfit)</li>
                <li>Multicollinearity issues</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
