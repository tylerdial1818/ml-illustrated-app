import { ModelSection } from '../../../components/ui/ModelSection'
import { GradientDescentViz } from '../visualizations/GradientDescentViz'
import { GradientDescentMath } from '../content/gradientDescentMath'

export function GradientDescentSection() {
  return (
    <ModelSection
      id="gradient-descent"
      title="Gradient Descent"
      subtitle="How we optimize. Taking small steps downhill to find the best parameters."
      intuition={
        <div className="max-w-2xl space-y-3">
          <p className="text-text-secondary leading-relaxed">
            Imagine you are blindfolded on a hilly landscape and you want to reach the lowest
            valley. You cannot see, but you can feel the slope under your feet. At each step you
            move in the direction that goes most steeply downhill. That is gradient descent.
          </p>
          <p className="text-text-secondary leading-relaxed">
            The "landscape" is the loss surface from the previous section. Every combination of
            slope and intercept has a loss value. The <strong className="text-text-primary">gradient</strong>{' '}
            is a vector pointing uphill (the direction of steepest increase). We move in the
            opposite direction to decrease the loss. The <strong className="text-text-primary">learning rate</strong>{' '}
            controls how big each step is.
          </p>
          <p className="text-text-secondary leading-relaxed">
            Too small a learning rate and you will take forever to arrive. Too large and you will
            overshoot the valley and bounce around. The right learning rate gets you there
            efficiently. Press Play below to watch it happen.
          </p>
        </div>
      }
      mechanism={<GradientDescentViz />}
      math={<GradientDescentMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Key Takeaways</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Gradient descent is the foundation of nearly all modern ML training</li>
                <li>The learning rate is the single most important hyperparameter</li>
                <li>Mini-batch SGD is the practical default for large datasets</li>
                <li>Adaptive optimizers (Adam, RMSProp) adjust learning rates per parameter</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Watch Out For</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Non-convex surfaces have local minima where GD can get stuck</li>
                <li>Vanishing/exploding gradients in deep networks need careful initialization</li>
                <li>Feature scaling matters: unscaled features warp the loss surface</li>
                <li>SGD noise helps escape shallow local minima but makes convergence noisy</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
