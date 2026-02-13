export interface LogisticSnapshot {
  step: number
  weights: number[] // [bias, w1, w2]
  loss: number // binary cross-entropy
  predictions: number[] // probabilities
  accuracy: number
}

export interface LogisticResult {
  snapshots: LogisticSnapshot[]
  finalWeights: number[]
}

/**
 * Numerically stable sigmoid function.
 * For large positive z, returns ~1. For large negative z, returns ~0.
 */
function sigmoid(z: number): number {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z))
  }
  // For negative z, use the equivalent form to avoid overflow in exp(z)
  const expZ = Math.exp(z)
  return expZ / (1 + expZ)
}

/**
 * Compute binary cross-entropy loss with numerical clamping.
 * L = -(1/N) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
 */
function binaryCrossEntropy(y: number[], predictions: number[]): number {
  const n = y.length
  const eps = 1e-15
  let loss = 0
  for (let i = 0; i < n; i++) {
    const p = Math.max(eps, Math.min(1 - eps, predictions[i]))
    loss -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p)
  }
  return loss / n
}

/**
 * Compute classification accuracy.
 */
function computeAccuracy(y: number[], predictions: number[]): number {
  let correct = 0
  for (let i = 0; i < y.length; i++) {
    const predicted = predictions[i] >= 0.5 ? 1 : 0
    if (predicted === y[i]) correct++
  }
  return correct / y.length
}

/**
 * Run logistic regression via gradient descent on binary cross-entropy loss.
 *
 * X: N x 2 feature matrix (two features per sample).
 * y: binary labels (0 or 1).
 *
 * The model computes: p = sigmoid(bias + w1*x1 + w2*x2)
 * Weights are stored as [bias, w1, w2].
 *
 * Gradient for weight j:
 *   dL/dw_j = (1/N) * Σ (p_i - y_i) * x_ij
 * where x_i0 = 1 (for the bias term).
 *
 * Returns an array of snapshots, one per gradient descent step.
 */
export function runLogisticRegression(
  X: number[][], // N x 2 feature matrix
  y: number[], // binary labels (0/1)
  learningRate: number = 0.1,
  maxSteps: number = 200
): LogisticSnapshot[] {
  const n = X.length
  const numFeatures = X[0].length // typically 2

  // Initialize weights to zero: [bias, w1, w2, ...]
  const weights = new Array(numFeatures + 1).fill(0)

  const snapshots: LogisticSnapshot[] = []

  for (let step = 0; step <= maxSteps; step++) {
    // Forward pass: compute predictions
    const predictions = new Array(n)
    for (let i = 0; i < n; i++) {
      let z = weights[0] // bias
      for (let j = 0; j < numFeatures; j++) {
        z += weights[j + 1] * X[i][j]
      }
      predictions[i] = sigmoid(z)
    }

    // Compute loss and accuracy
    const loss = binaryCrossEntropy(y, predictions)
    const accuracy = computeAccuracy(y, predictions)

    // Record snapshot
    snapshots.push({
      step,
      weights: [...weights],
      loss,
      predictions: [...predictions],
      accuracy,
    })

    // Skip gradient update on the last step (we only needed the snapshot)
    if (step === maxSteps) break

    // Compute gradients
    const gradients = new Array(numFeatures + 1).fill(0)
    for (let i = 0; i < n; i++) {
      const error = predictions[i] - y[i]
      gradients[0] += error // bias gradient (x_i0 = 1)
      for (let j = 0; j < numFeatures; j++) {
        gradients[j + 1] += error * X[i][j]
      }
    }

    // Average gradients and update weights
    for (let j = 0; j <= numFeatures; j++) {
      weights[j] -= learningRate * (gradients[j] / n)
    }
  }

  return snapshots
}
