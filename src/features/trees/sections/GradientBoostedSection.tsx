import { ModelSection } from '../../../components/ui/ModelSection'
import { GradientBoostedViz } from '../visualizations/GradientBoostedViz'
import { GradientBoostedMath } from '../content/gradientBoostedMath'

export function GradientBoostedSection() {
  return (
    <ModelSection
      id="gradient-boosted"
      title="Gradient Boosted Trees"
      subtitle="Build trees one at a time, where each new tree specifically corrects the mistakes of all previous trees combined."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Imagine studying for an exam with a series of <strong className="text-text-primary">specialized tutors</strong>.
            The first tutor covers the basics and you take a practice test. The second tutor looks
            only at the questions you got wrong and focuses there. The third tutor focuses on what
            the first two still couldn't fix. Each tutor is a specialist in your remaining weaknesses.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            Gradient Boosted Trees work the same way. Each new tree is trained not on the original
            targets, but on the <em>residual errors</em> left by the ensemble so far. The learning
            rate controls how much each tree contributes — small steps lead to better generalization,
            just like careful, incremental studying beats cramming.
          </p>
        </div>
      }
      mechanism={<GradientBoostedViz />}
      math={<GradientBoostedMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>State-of-the-art for structured/tabular data</li>
                <li>Often wins ML competitions (XGBoost, LightGBM)</li>
                <li>Handles mixed feature types well</li>
                <li>Built-in regularization (learning rate, tree depth)</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Can overfit if not carefully tuned</li>
                <li>Requires hyperparameter tuning (learning rate, depth, n_trees)</li>
                <li>Sequential training — harder to parallelize than Random Forest</li>
                <li>Less interpretable than single trees or Random Forests</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
