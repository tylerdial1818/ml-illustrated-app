import { createRng, normalRandom } from '../../math/random'

// ── Types ────────────────────────────────────────────────────────────

export interface MLPSnapshot {
  epoch: number
  weights: number[][][] // weights[layer][fromNode][toNode]
  biases: number[][] // biases[layer][node]
  trainLoss: number
  trainAccuracy: number
  decisionBoundaryGrid?: { x: number; y: number; value: number }[] // probability field
}

export interface MLPConfig {
  layerSizes: number[] // e.g. [2, 4, 4, 1] for 2 inputs, 2 hidden layers of 4, 1 output
  learningRate?: number
  activation?: 'relu' | 'sigmoid' | 'tanh'
  epochs?: number
  seed?: number
}

// ── Activation functions and their derivatives ───────────────────────

function sigmoid(z: number): number {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z))
  }
  const expZ = Math.exp(z)
  return expZ / (1 + expZ)
}

function sigmoidDeriv(output: number): number {
  return output * (1 - output)
}

function relu(z: number): number {
  return z > 0 ? z : 0
}

function reluDeriv(output: number): number {
  return output > 0 ? 1 : 0
}

function tanhActivation(z: number): number {
  return Math.tanh(z)
}

function tanhDeriv(output: number): number {
  return 1 - output * output
}

type ActivationFn = (z: number) => number
type ActivationDerivFn = (output: number) => number

function getActivation(name: 'relu' | 'sigmoid' | 'tanh'): {
  fn: ActivationFn
  deriv: ActivationDerivFn
} {
  switch (name) {
    case 'relu':
      return { fn: relu, deriv: reluDeriv }
    case 'sigmoid':
      return { fn: sigmoid, deriv: sigmoidDeriv }
    case 'tanh':
      return { fn: tanhActivation, deriv: tanhDeriv }
  }
}

// ── Loss helpers ─────────────────────────────────────────────────────

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

function computeAccuracy(labels: number[], predictions: number[]): number {
  let correct = 0
  for (let i = 0; i < labels.length; i++) {
    if ((predictions[i] >= 0.5 ? 1 : 0) === labels[i]) correct++
  }
  return correct / labels.length
}

// ── Network initialisation ───────────────────────────────────────────

/**
 * Initialise weight matrices and bias vectors for the network.
 *
 * Weights are drawn from N(0, sqrt(2 / fanIn)) (He initialisation for ReLU,
 * reasonable for sigmoid/tanh as well in this educational context).
 */
function initNetwork(
  layerSizes: number[],
  rng: () => number
): { weights: number[][][]; biases: number[][] } {
  const numLayers = layerSizes.length - 1
  const weights: number[][][] = []
  const biases: number[][] = []

  for (let l = 0; l < numLayers; l++) {
    const fanIn = layerSizes[l]
    const fanOut = layerSizes[l + 1]
    const std = Math.sqrt(2 / fanIn)

    const W: number[][] = []
    for (let i = 0; i < fanIn; i++) {
      const row: number[] = []
      for (let j = 0; j < fanOut; j++) {
        row.push(normalRandom(rng, 0, std))
      }
      W.push(row)
    }
    weights.push(W)

    const b: number[] = []
    for (let j = 0; j < fanOut; j++) {
      b.push(0)
    }
    biases.push(b)
  }

  return { weights, biases }
}

// ── Forward pass ─────────────────────────────────────────────────────

/**
 * Run the forward pass for a single sample, returning all layer activations
 * and pre-activation values (needed for backprop).
 */
function forward(
  input: number[],
  weights: number[][][],
  biases: number[][],
  activationFn: ActivationFn
): { activations: number[][]; preActivations: number[][] } {
  const numLayers = weights.length
  const activations: number[][] = [input]
  const preActivations: number[][] = []

  let current = input
  for (let l = 0; l < numLayers; l++) {
    const fanIn = weights[l].length
    const fanOut = weights[l][0].length
    const z: number[] = new Array(fanOut)
    const a: number[] = new Array(fanOut)

    for (let j = 0; j < fanOut; j++) {
      let sum = biases[l][j]
      for (let i = 0; i < fanIn; i++) {
        sum += current[i] * weights[l][i][j]
      }
      z[j] = sum

      // Output layer always uses sigmoid (binary classification)
      if (l === numLayers - 1) {
        a[j] = sigmoid(sum)
      } else {
        a[j] = activationFn(sum)
      }
    }

    preActivations.push(z)
    activations.push(a)
    current = a
  }

  return { activations, preActivations }
}

// ── Backpropagation ──────────────────────────────────────────────────

/**
 * Compute weight and bias gradients for a single sample using backprop.
 */
function backward(
  label: number,
  activations: number[][],
  _preActivations: number[][],
  weights: number[][][],
  _biases: number[][],
  activationDeriv: ActivationDerivFn
): { dWeights: number[][][]; dBiases: number[][] } {
  const numLayers = weights.length
  const dWeights: number[][][] = []
  const dBiases: number[][] = []

  // Pre-allocate gradient structures matching the weight shapes
  for (let l = 0; l < numLayers; l++) {
    const fanIn = weights[l].length
    const fanOut = weights[l][0].length
    dWeights.push(
      Array.from({ length: fanIn }, () => new Array(fanOut).fill(0))
    )
    dBiases.push(new Array(fanOut).fill(0))
  }

  // Output layer delta: d(BCE)/d(z) = (a - y) for sigmoid output
  const outputActivation = activations[numLayers]
  let delta: number[] = outputActivation.map((a) => a - label)

  // Propagate backwards through layers
  for (let l = numLayers - 1; l >= 0; l--) {
    const prevActivation = activations[l]
    const fanIn = weights[l].length
    const fanOut = weights[l][0].length

    // Accumulate gradients
    for (let i = 0; i < fanIn; i++) {
      for (let j = 0; j < fanOut; j++) {
        dWeights[l][i][j] = delta[j] * prevActivation[i]
      }
    }
    for (let j = 0; j < fanOut; j++) {
      dBiases[l][j] = delta[j]
    }

    // Compute delta for the previous layer (if not the first layer)
    if (l > 0) {
      const prevDelta: number[] = new Array(fanIn).fill(0)
      for (let i = 0; i < fanIn; i++) {
        let sum = 0
        for (let j = 0; j < fanOut; j++) {
          sum += weights[l][i][j] * delta[j]
        }
        // Multiply by activation derivative of the hidden layer
        prevDelta[i] = sum * activationDeriv(activations[l][i])
      }
      delta = prevDelta
    }
  }

  return { dWeights, dBiases }
}

// ── Decision boundary grid ───────────────────────────────────────────

function generateDecisionBoundaryGrid(
  data: { x: number; y: number }[],
  weights: number[][][],
  biases: number[][],
  activationFn: ActivationFn,
  gridSize: number = 30
): { x: number; y: number; value: number }[] {
  // Compute data range with a small margin
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const pt of data) {
    if (pt.x < minX) minX = pt.x
    if (pt.x > maxX) maxX = pt.x
    if (pt.y < minY) minY = pt.y
    if (pt.y > maxY) maxY = pt.y
  }
  const marginX = (maxX - minX) * 0.15
  const marginY = (maxY - minY) * 0.15
  minX -= marginX
  maxX += marginX
  minY -= marginY
  maxY += marginY

  const stepX = (maxX - minX) / (gridSize - 1)
  const stepY = (maxY - minY) / (gridSize - 1)

  const grid: { x: number; y: number; value: number }[] = []
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const gx = minX + col * stepX
      const gy = minY + row * stepY
      const { activations } = forward([gx, gy], weights, biases, activationFn)
      const output = activations[activations.length - 1][0]
      grid.push({ x: gx, y: gy, value: output })
    }
  }
  return grid
}

// ── Deep-copy helpers ────────────────────────────────────────────────

function cloneWeights(w: number[][][]): number[][][] {
  return w.map((layer) => layer.map((row) => [...row]))
}

function cloneBiases(b: number[][]): number[][] {
  return b.map((layer) => [...layer])
}

// ── Main training function ───────────────────────────────────────────

/**
 * Train a multi-layer perceptron on 2-D binary-classification data.
 *
 * Supports 1-4 hidden layers with ReLU, sigmoid, or tanh activations.
 * Uses full-batch gradient descent with binary cross-entropy loss.
 * The output layer always uses sigmoid activation.
 *
 * @returns An array of MLPSnapshots, one per epoch, suitable for animation.
 */
export function runMLPTraining(
  data: { x: number; y: number; label: number }[],
  config: MLPConfig
): MLPSnapshot[] {
  const {
    layerSizes,
    learningRate = 0.1,
    activation = 'relu',
    epochs = 100,
    seed = 42,
  } = config

  const n = data.length
  if (n === 0 || layerSizes.length < 2) return []

  const rng = createRng(seed)
  const { fn: activationFn, deriv: activationDeriv } = getActivation(activation)

  // Initialise network parameters
  let { weights, biases } = initNetwork(layerSizes, rng)

  const snapshots: MLPSnapshot[] = []
  const labels = data.map((d) => d.label)

  // ── Epoch 0 snapshot (before training) ──
  {
    const predictions = data.map((pt) => {
      const { activations } = forward([pt.x, pt.y], weights, biases, activationFn)
      return activations[activations.length - 1][0]
    })
    snapshots.push({
      epoch: 0,
      weights: cloneWeights(weights),
      biases: cloneBiases(biases),
      trainLoss: binaryCrossEntropy(labels, predictions),
      trainAccuracy: computeAccuracy(labels, predictions),
      decisionBoundaryGrid: generateDecisionBoundaryGrid(
        data,
        weights,
        biases,
        activationFn
      ),
    })
  }

  // ── Training loop ──────────────────────────────────────────────
  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Accumulate gradients over the full batch
    const numLayers = weights.length
    const accumWeights: number[][][] = weights.map((layer) =>
      layer.map((row) => new Array(row.length).fill(0))
    )
    const accumBiases: number[][] = biases.map((layer) =>
      new Array(layer.length).fill(0)
    )

    for (let i = 0; i < n; i++) {
      const input = [data[i].x, data[i].y]
      const { activations, preActivations } = forward(
        input,
        weights,
        biases,
        activationFn
      )
      const { dWeights, dBiases } = backward(
        data[i].label,
        activations,
        preActivations,
        weights,
        biases,
        activationDeriv
      )

      // Accumulate
      for (let l = 0; l < numLayers; l++) {
        const fanIn = weights[l].length
        const fanOut = weights[l][0].length
        for (let fi = 0; fi < fanIn; fi++) {
          for (let fj = 0; fj < fanOut; fj++) {
            accumWeights[l][fi][fj] += dWeights[l][fi][fj]
          }
        }
        for (let fj = 0; fj < fanOut; fj++) {
          accumBiases[l][fj] += dBiases[l][fj]
        }
      }
    }

    // Apply averaged gradients
    for (let l = 0; l < numLayers; l++) {
      const fanIn = weights[l].length
      const fanOut = weights[l][0].length
      for (let fi = 0; fi < fanIn; fi++) {
        for (let fj = 0; fj < fanOut; fj++) {
          weights[l][fi][fj] -= learningRate * (accumWeights[l][fi][fj] / n)
        }
      }
      for (let fj = 0; fj < fanOut; fj++) {
        biases[l][fj] -= learningRate * (accumBiases[l][fj] / n)
      }
    }

    // Compute predictions for the snapshot
    const predictions = data.map((pt) => {
      const { activations } = forward([pt.x, pt.y], weights, biases, activationFn)
      return activations[activations.length - 1][0]
    })

    // Generate decision boundary grid every epoch
    // (for smooth animation - the 30x30 grid is small enough)
    const grid = generateDecisionBoundaryGrid(
      data,
      weights,
      biases,
      activationFn
    )

    snapshots.push({
      epoch,
      weights: cloneWeights(weights),
      biases: cloneBiases(biases),
      trainLoss: binaryCrossEntropy(labels, predictions),
      trainAccuracy: computeAccuracy(labels, predictions),
      decisionBoundaryGrid: grid,
    })
  }

  return snapshots
}
