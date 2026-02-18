import { ModelSection } from '../../../components/ui/ModelSection'
import { DatasetExplorerViz } from '../visualizations/DatasetExplorerViz'
import { FeaturesLabelsMath } from '../content/featuresLabelsMath'

export function FeaturesLabelsSection() {
  return (
    <ModelSection
      id="features-labels"
      title="Features, Labels, and Datasets"
      subtitle="The vocabulary of ML: what your data looks like and what you're trying to predict."
      intuition={
        <div className="max-w-2xl space-y-3">
          <p className="text-text-secondary leading-relaxed">
            Before any model can learn, you need data organized in a specific way. Think of a
            spreadsheet. Each row is one example (one house, one patient, one email). Each column
            is a <strong className="text-text-primary">feature</strong>: a measurable property of
            that example (square footage, age, word count). One special column is the{' '}
            <strong className="text-text-primary">label</strong>: the thing you want to predict
            (sale price, diagnosis, spam or not).
          </p>
          <p className="text-text-secondary leading-relaxed">
            Features are the inputs. The label is the output. A dataset is the full collection of
            examples. That is the main vocabulary you need to understand every model on this site.
          </p>
        </div>
      }
      mechanism={<DatasetExplorerViz />}
      math={<FeaturesLabelsMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Key Takeaways</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Every ML model takes features as input and produces a prediction</li>
                <li>Feature engineering (choosing good features) often matters more than model choice</li>
                <li>Always hold out a test set to check if the model generalizes</li>
                <li>Categorical features need encoding (numbers) before most models can use them</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Common Pitfalls</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Including the label (or a proxy for it) as a feature causes data leakage</li>
                <li>Too few examples relative to features leads to overfitting</li>
                <li>Missing values need explicit handling (imputation or removal)</li>
                <li>Features on wildly different scales can bias distance-based models</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
