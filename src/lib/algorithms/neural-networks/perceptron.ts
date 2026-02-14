import { createRng, normalRandom } from '../../math/random'

// ── Types ────────────────────────────────────────────────────────────

export interface PerceptronSnapshot {
  step: number
  weights: number[] // [w1, w2]
  bias: number
  predictions: number[]
  accuracy: number
  loss: number
  decisionBoundary: { slope: number; intercept: number } | null
  highlightedPoint?: number // index of misclassified point being corrected
  weightUpdates?: number[] // delta applied this step
}

// ── Activation helpers ───────────────────────────────────────────────

/**
 * Numerically stable sigmoid: sigma(z) = 1 / (1 + e^{-z}).
 */
function sigmoid(z: number): number {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z))
  }
  const expZ = Math.exp(z)
  return expZ / (1 + expZ)
}

function stepActivation(z: number): number {
  return z > 0 ? 1 : 0
}

// ── Loss helpers ─────────────────────────────────────────────────────

/**
 * Binary cross-entropy loss (used with sigmoid activation).
 */
function binaryCrossEntropy(labels: number[], predictions: number[]): number {
  const n = labels.length
  const eps = 1e-15
  let loss = 0
  for (let i = 0; i < n; i++) {
    const p = Math.max(eps, Math.min(1 - eps, predictions[i]))
    loss -= labels[i] * Math.log(p) + (1 - labels[i]) * Math.log(1 - p)
  }
  return loss / n
}

/**
 * Misclassification rate (used with step activation).
 */
function misclassificationLoss(labels: number[], predictions: number[]): number {
  let wrong = 0
  for (let i = 0; i < labels.length; i++) {
    if ((predictions[i] >= 0.5 ? 1 : 0) !== labels[i]) wrong++
  }
  return wrong / labels.length
}

// ── Decision boundary ────────────────────────────────────────────────

/**
 * Compute the decision boundary line from weights and bias.
 *
 * The boundary satisfies  w1*x + w2*y + b = 0
 * => y = -(w1/w2)*x - (b/w2)
 *
 * Returns null when w2 ~ 0 (vertical line can't be expressed as slope/intercept).
 */
function computeDecisionBoundary(
  w: number[],
  bias: number
): { slope: number; intercept: number } | null {
  if (Math.abs(w[1]) < 1e-12) return null
  return {
    slope: -w[0] / w[1],
    intercept: -bias / w[1],
  }
}

// ── Main training function ───────────────────────────────────────────

/**
 * Train a single-neuron perceptron on 2-D labelled data.
 *
 * Two modes:
 * - **step** activation: classic perceptron learning rule.  One snapshot is
 *   emitted every time a misclassified point triggers a weight update.
 * - **sigmoid** activation: gradient-descent on binary cross-entropy.  One
 *   snapshot is emitted per epoch (full pass over all points).
 *
 * @returns An array of PerceptronSnapshots suitable for frame-by-frame animation.
 */
export function runPerceptronTraining(
  data: { x: number; y: number; label: number }[],
  config: {
    learningRate?: number
    epochs?: number
    activation?: 'step' | 'sigmoid'
    seed?: number
  } = {}
): PerceptronSnapshot[] {
  const {
    learningRate = 0.1,
    epochs = 50,
    activation = 'step',
    seed = 42,
  } = config

  const rng = createRng(seed)
  const n = data.length
  if (n === 0) return []

  // Xavier-ish initialisation for two weights + bias
  const weights = [normalRandom(rng, 0, 0.5), normalRandom(rng, 0, 0.5)]
  let bias = normalRandom(rng, 0, 0.1)

  const snapshots: PerceptronSnapshot[] = []
  let step = 0

  // Helper: compute all predictions with current weights
  const predict = (): number[] =>
    data.map((pt) => {
      const z = weights[0] * pt.x + weights[1] * pt.y + bias
      return activation === 'sigmoid' ? sigmoid(z) : stepActivation(z)
    })

  // Helper: compute accuracy
  const accuracy = (preds: number[]): number => {
    let correct = 0
    for (let i = 0; i < n; i++) {
      if ((preds[i] >= 0.5 ? 1 : 0) === data[i].label) correct++
    }
    return correct / n
  }

  // Helper: record a snapshot
  const snap = (
    preds: number[],
    highlighted?: number,
    deltas?: number[]
  ): void => {
    const lossFn = activation === 'sigmoid' ? binaryCrossEntropy : misclassificationLoss
    snapshots.push({
      step: step++,
      weights: [...weights],
      bias,
      predictions: [...preds],
      accuracy: accuracy(preds),
      loss: lossFn(data.map((d) => d.label), preds),
      decisionBoundary: computeDecisionBoundary(weights, bias),
      highlightedPoint: highlighted,
      weightUpdates: deltas,
    })
  }

  // ---- Initial snapshot (before any training) ----
  snap(predict())

  if (activation === 'step') {
    // ── Classic perceptron learning rule ──────────────────────────
    // For each epoch, iterate over data.  Whenever a point is misclassified,
    // apply the update and emit a snapshot highlighting that point.
    for (let epoch = 0; epoch < epochs; epoch++) {
      let anyUpdate = false
      for (let i = 0; i < n; i++) {
        const z = weights[0] * data[i].x + weights[1] * data[i].y + bias
        const yHat = stepActivation(z)
        if (yHat !== data[i].label) {
          const error = data[i].label - yHat // +1 or -1
          const dw0 = learningRate * error * data[i].x
          const dw1 = learningRate * error * data[i].y
          const db = learningRate * error
          weights[0] += dw0
          weights[1] += dw1
          bias += db
          anyUpdate = true
          snap(predict(), i, [dw0, dw1])
        }
      }
      // If no updates were made the data is linearly separable and we converged
      if (!anyUpdate) break
    }
  } else {
    // ── Sigmoid gradient descent ─────────────────────────────────
    // One snapshot per epoch (full-batch gradient descent).
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Compute gradients over the full batch
      let gw0 = 0
      let gw1 = 0
      let gb = 0
      for (let i = 0; i < n; i++) {
        const z = weights[0] * data[i].x + weights[1] * data[i].y + bias
        const p = sigmoid(z)
        const err = p - data[i].label // d(BCE)/d(z)
        gw0 += err * data[i].x
        gw1 += err * data[i].y
        gb += err
      }
      const dw0 = -learningRate * (gw0 / n)
      const dw1 = -learningRate * (gw1 / n)
      const db = -learningRate * (gb / n)
      weights[0] += dw0
      weights[1] += dw1
      bias += db

      // Find a misclassified point to highlight (if any)
      const preds = predict()
      let highlighted: number | undefined
      for (let i = 0; i < n; i++) {
        if ((preds[i] >= 0.5 ? 1 : 0) !== data[i].label) {
          highlighted = i
          break
        }
      }
      snap(preds, highlighted, [dw0, dw1])
    }
  }

  return snapshots
}
