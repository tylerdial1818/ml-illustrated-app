import { ModelSection } from '../../../components/ui/ModelSection'
import { LossLandscapeViz } from '../visualizations/LossLandscapeViz'
import { LossFunctionsMath } from '../content/lossFunctionsMath'

export function LossFunctionsSection() {
  return (
    <ModelSection
      id="loss-functions"
      title="Loss Functions"
      subtitle="What we are optimizing and why. A loss function measures how wrong the model is."
      intuition={
        <div className="max-w-2xl space-y-3">
          <p className="text-text-secondary leading-relaxed">
            Imagine you are throwing darts at a target. You need a way to measure how badly you
            are missing. That is what a loss function does for a model. It takes the model's
            predictions, compares them to the actual answers, and produces a single number:
            the <strong className="text-text-primary">loss</strong>. A lower loss means the model
            is doing better.
          </p>
          <p className="text-text-secondary leading-relaxed">
            The choice of loss function matters. <strong className="text-text-primary">MSE</strong>{' '}
            (Mean Squared Error) squares each error, so one big miss hurts more than several small
            ones. <strong className="text-text-primary">MAE</strong> (Mean Absolute Error) treats all
            errors proportionally. <strong className="text-text-primary">Huber</strong> blends both:
            it acts like MSE for small errors and MAE for large ones, making it robust to outliers
            without losing smoothness.
          </p>
          <p className="text-text-secondary leading-relaxed">
            Try toggling the outlier below and watch how MSE reacts compared to MAE. That
            sensitivity difference drives real-world model choices.
          </p>
        </div>
      }
      mechanism={<LossLandscapeViz />}
      math={<LossFunctionsMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Choosing a Loss Function</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li><strong className="text-text-primary">MSE</strong> when outliers are rare and you want smooth optimization</li>
                <li><strong className="text-text-primary">MAE</strong> when outliers are common or you want median-like behavior</li>
                <li><strong className="text-text-primary">Huber</strong> when you want robustness without losing differentiability</li>
                <li>Classification uses different losses (cross-entropy, hinge)</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Watch Out For</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>MSE amplifies outliers: one extreme point can drag the fit</li>
                <li>MAE gradient is constant, which can make optimization slower near the minimum</li>
                <li>Huber requires choosing the delta threshold</li>
                <li>The "best" loss depends on your data and what errors cost in your domain</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
