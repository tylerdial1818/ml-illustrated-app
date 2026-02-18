import { ModelSection } from '../../../components/ui/ModelSection'
import { DecisionTreeViz } from '../visualizations/DecisionTreeViz'
import { DecisionTreeMath } from '../content/decisionTreeMath'

export function DecisionTreeSection() {
  return (
    <ModelSection
      id="decision-tree"
      title="Decision Tree (CART)"
      subtitle="Learn a flowchart of yes/no questions that splits data into increasingly pure groups."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Imagine playing <strong className="text-text-primary">20 Questions</strong> with your data.
            At each step, the tree asks the single best yes/no question that separates the data into
            purer groups. "Is feature X greater than 5.3?" Left if yes, right if no. Repeat until the
            leaves are pure enough to make a prediction.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            The tree is greedy: it picks the <em>locally</em> best split at each node without looking
            ahead. This makes it fast to build but means it can miss globally optimal structures. Without
            constraints, a tree will happily memorize every training example, which is why pruning
            (or ensembling) is essential.
          </p>
        </div>
      }
      mechanism={<DecisionTreeViz />}
      math={<DecisionTreeMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Highly interpretable: you can visualize the rules</li>
                <li>Handles both numerical and categorical features</li>
                <li>Captures non-linear relationships naturally</li>
                <li>No feature scaling required</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>High variance: small data changes can alter the tree</li>
                <li>Axis-aligned splits only (can't learn diagonal boundaries)</li>
                <li>Prone to overfitting without pruning</li>
                <li>Unstable: sensitive to training data perturbations</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
