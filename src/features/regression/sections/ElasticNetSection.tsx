import { ModelSection } from '../../../components/ui/ModelSection'
import { ElasticNetViz } from '../visualizations/ElasticNetViz'
import { ElasticNetMath } from '../content/elasticnetMath'

export function ElasticNetSection() {
  return (
    <ModelSection
      id="elasticnet"
      title="ElasticNet"
      subtitle="The best of both worlds — blend Ridge and Lasso for stability plus sparsity."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            ElasticNet blends Ridge and Lasso. The{' '}
            <strong className="text-text-primary">l1_ratio</strong> lets you control: more Ridge-like
            (shrink everything) or more Lasso-like (zero some out). Watch the constraint shape morph from
            a diamond (Lasso) to a circle (Ridge) as you adjust the slider.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            This is especially useful when features are correlated — Lasso tends to randomly pick one,
            while ElasticNet can include groups of correlated features together.
          </p>
        </div>
      }
      mechanism={<ElasticNetViz />}
      math={<ElasticNetMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Feature selection + stability</li>
                <li>Handles correlated features well</li>
                <li>Flexible via l1_ratio tuning</li>
                <li>Subsumes Ridge and Lasso</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Two hyperparameters to tune</li>
                <li>Computationally heavier</li>
                <li>Can be overkill for simple problems</li>
                <li>Less interpretable than pure Lasso</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
