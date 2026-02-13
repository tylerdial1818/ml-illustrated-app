import { ModelSection } from '../../../components/ui/ModelSection'
import { LassoViz } from '../visualizations/LassoViz'
import { LassoMath } from '../content/lassoMath'

export function LassoSection() {
  return (
    <ModelSection
      id="lasso"
      title="Lasso Regression (L1)"
      subtitle="Shrink some coefficients all the way to zero — automatic feature selection."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Like Ridge, but with a <strong className="text-text-primary">sharper penalty</strong>. Lasso
            doesn't just shrink coefficients — it drives some of them all the way to zero, effectively
            selecting which features matter. It's regression + automatic feature selection.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            Compare the two plots: Ridge coefficients shrink smoothly but never hit zero. Lasso coefficients
            hit zero one by one as α increases. This is the key visual that makes L1 vs. L2 click.
          </p>
        </div>
      }
      mechanism={<LassoViz />}
      math={<LassoMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li>Automatic feature selection</li>
                <li>Produces sparse models</li>
                <li>More interpretable than Ridge</li>
                <li>Good for high-dimensional data</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li>Unstable with correlated features</li>
                <li>Selects at most n features (n = samples)</li>
                <li>No closed-form solution</li>
                <li>Can be slow for very large problems</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
