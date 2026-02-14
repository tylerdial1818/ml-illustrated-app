import { ModelSection } from '../../../components/ui/ModelSection'
import { RandomForestViz } from '../visualizations/RandomForestViz'
import { RandomForestMath } from '../content/randomForestMath'

export function RandomForestSection() {
  return (
    <ModelSection
      id="random-forest"
      title="Random Forest"
      subtitle="Train many different trees on random subsets of data and features, then let them vote."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            <strong className="text-text-primary">One tree is smart but unreliable.</strong> Imagine asking
            a single expert for a diagnosis — they might be brilliant but also biased by their experience.
            Now imagine asking 100 experts, each trained on slightly different cases and looking at
            different symptoms. Their majority vote will almost always be better than any individual.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            That's the wisdom-of-crowds principle behind Random Forests. Each tree sees a random
            bootstrap sample of the data and can only consider a random subset of features at
            each split. This forced diversity means individual trees make different mistakes —
            and those mistakes cancel out when you aggregate.
          </p>
        </div>
      }
      mechanism={<RandomForestViz />}
      math={<RandomForestMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Excellent "strong baseline" — works well out of the box</li>
                <li>Handles high-dimensional data with many features</li>
                <li>Robust to outliers and noisy features</li>
                <li>Built-in feature importance estimation</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Less interpretable than a single tree</li>
                <li>Cannot extrapolate beyond training data range</li>
                <li>Larger memory footprint (stores many trees)</li>
                <li>Can be slow to predict with very large forests</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
