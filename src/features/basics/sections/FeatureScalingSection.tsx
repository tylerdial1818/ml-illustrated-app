import { ModelSection } from '../../../components/ui/ModelSection'
import { FeatureScalingViz } from '../visualizations/FeatureScalingViz'
import { FeatureScalingMath } from '../content/featureScalingMath'

export function FeatureScalingSection() {
  return (
    <ModelSection
      id="feature-scaling"
      title="Feature Scaling"
      subtitle="When features have wildly different scales, models behave badly. Scaling puts everything on equal footing."
      intuition={
        <div className="max-w-2xl space-y-3">
          <p className="text-text-secondary leading-relaxed">
            Imagine predicting house prices using two features: square footage (ranges from 500 to
            5,000) and number of bedrooms (ranges from 1 to 6). Without scaling, the model treats a
            1-unit change in square footage the same as a 1-unit change in bedrooms. But going from
            1,000 to 1,001 square feet is negligible, while going from 2 to 3 bedrooms is significant.
          </p>
          <p className="text-text-secondary leading-relaxed">
            Worse, gradient descent takes uneven steps. It makes big updates for the feature with the
            large range and tiny updates for the one with the small range. The optimization path
            zigzags instead of heading straight for the minimum. Scaling both features to similar
            ranges fixes this.
          </p>
        </div>
      }
      mechanism={<FeatureScalingViz />}
      math={<FeatureScalingMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">When to Scale</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Gradient-based models (linear regression, neural networks, SVMs)</li>
                <li>Distance-based models (K-Means, KNN, PCA)</li>
                <li>When features have very different units or ranges</li>
                <li>Use <strong className="text-text-primary">Robust scaling</strong> when outliers are present</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">When Not to Scale</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Tree-based models (decision trees, random forests, gradient boosting) only care about ordering</li>
                <li>When the raw scale carries meaningful information</li>
                <li>Always fit the scaler on training data, then apply to test data</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
