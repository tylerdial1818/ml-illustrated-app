import { ModelSection } from '../../../components/ui/ModelSection'
import { PerceptronViz } from '../visualizations/PerceptronViz'
import { PerceptronMath } from '../content/perceptronMath'

export function PerceptronSection() {
  return (
    <ModelSection
      id="perceptron"
      title="The Perceptron"
      subtitle="The simplest neural network: one neuron, one decision boundary, one straight line."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Think of a perceptron as a <strong className="text-text-primary">bouncer at a club</strong>.
            It looks at a few things about you (maybe your age and dress code), multiplies each by
            how much it cares about that factor (the weights), adds them up, and makes a single
            yes-or-no decision. If the total score crosses a threshold, you're in. Otherwise, you're out.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            The decision boundary is always a straight line (or hyperplane in higher dimensions).
            The perceptron can only solve problems where the two classes can be separated by a single
            straight cut. These are called <em>linearly separable</em> problems.
          </p>
        </div>
      }
      mechanism={<PerceptronViz />}
      math={<PerceptronMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Dead simple to understand and implement</li>
                <li>Fast to train: just one pass per update</li>
                <li>Foundation for understanding all neural networks</li>
                <li>Guaranteed to converge for linearly separable data</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Only works on linearly separable problems</li>
                <li>Cannot learn XOR or other non-linear patterns</li>
                <li>Single decision boundary, no nuance</li>
                <li>No hidden layers means no feature composition</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
