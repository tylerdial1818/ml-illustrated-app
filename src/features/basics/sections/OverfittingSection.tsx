import { ModelSection } from '../../../components/ui/ModelSection'
import { OverfittingViz } from '../visualizations/OverfittingViz'
import { OverfittingMath } from '../content/overfittingMath'

export function OverfittingSection() {
  return (
    <ModelSection
      id="overfitting"
      title="Overfitting & the Train/Test Split"
      subtitle="A model that memorizes the training data perfectly will fail on new data. Complexity is not free."
      intuition={
        <div className="max-w-2xl space-y-3">
          <p className="text-text-secondary leading-relaxed">
            Imagine a student studying for an exam by memorizing every question and answer from
            last year's test. On last year's test, they would score 100%. On this year's test with
            new questions, they would bomb. They did not learn the subject. They memorized the answers.
          </p>
          <p className="text-text-secondary leading-relaxed">
            Models do the same thing. A sufficiently complex model can memorize every quirk and noise
            pattern in the training data, achieving near-zero training error. But when you show it
            new data it has never seen, it performs terribly. This is{' '}
            <strong className="text-text-primary">overfitting</strong>: the model learned the noise
            instead of the signal.
          </p>
          <p className="text-text-secondary leading-relaxed">
            The fix is simple. Before training, set aside a portion of your data that the model never
            sees during training. Train on the rest. Then evaluate on the held-out data. If the model
            performs well on data it has never seen, it actually learned something generalizable.
          </p>
        </div>
      }
      mechanism={<OverfittingViz />}
      math={<OverfittingMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Key Takeaways</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Always evaluate on data the model has not seen during training</li>
                <li>More model complexity does not always mean better performance</li>
                <li>The gap between train and test loss signals overfitting</li>
                <li>Regularization, early stopping, and cross-validation help prevent it</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Bias-Variance Tradeoff</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li><strong className="text-text-primary">Underfitting</strong> = high bias. The model is too simple to capture the pattern.</li>
                <li><strong className="text-text-primary">Overfitting</strong> = high variance. The model is so complex it fits the noise.</li>
                <li>The sweet spot balances bias and variance. This tradeoff appears everywhere in ML.</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
