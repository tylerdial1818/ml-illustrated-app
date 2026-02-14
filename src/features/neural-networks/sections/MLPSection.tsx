import { ModelSection } from '../../../components/ui/ModelSection'
import { MLPViz } from '../visualizations/MLPViz'
import { MLPMath } from '../content/mlpMath'

export function MLPSection() {
  return (
    <ModelSection
      id="mlp"
      title="Multi-Layer Perceptron (MLP)"
      subtitle="Stack layers of neurons to learn non-linear decision boundaries — the workhorse feedforward network."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            One neuron draws <strong className="text-text-primary">one line</strong>. A layer of neurons
            draws <strong className="text-text-primary">many lines</strong>. Stack another layer on top
            and those lines combine into curves, regions, and complex shapes. Each layer builds on the
            features detected by the previous one.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            This is the key insight: by composing simple linear operations with non-linear activations,
            an MLP can approximate any continuous function, a result known as the{' '}
            <em>universal approximation theorem</em>. The network doesn't need to be told what features
            to look for; it discovers them during training through backpropagation.
          </p>
        </div>
      }
      mechanism={<MLPViz />}
      math={<MLPMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Learns non-linear decision boundaries</li>
                <li>Universal function approximator</li>
                <li>Works well on tabular and structured data</li>
                <li>Flexible architecture — add layers and neurons as needed</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Requires more data than simpler models</li>
                <li>Harder to interpret than linear models</li>
                <li>Prone to overfitting without regularization</li>
                <li>No built-in structure for spatial or sequential data</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
