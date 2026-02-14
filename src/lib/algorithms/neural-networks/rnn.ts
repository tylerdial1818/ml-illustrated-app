import { createRng, normalRandom } from '../../math/random'

// ── Types ────────────────────────────────────────────────────────────

export interface RNNStepResult {
  timeStep: number
  input: number
  hiddenState: number[]
  output: number
  intermediates: { weightedInput: number[]; preActivation: number[] }
}

export interface RNNResult {
  steps: RNNStepResult[]
  gradientMagnitudes: number[] // gradient magnitude per time step (for vanishing gradient viz)
}

export interface LSTMStepResult {
  timeStep: number
  input: number
  forgetGate: number[]
  inputGate: number[]
  cellCandidate: number[]
  cellState: number[]
  outputGate: number[]
  hiddenState: number[]
}

export interface LSTMResult {
  steps: LSTMStepResult[]
  gradientMagnitudes: number[]
}

// ── Activation helpers ───────────────────────────────────────────────

function sigmoid(z: number): number {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z))
  }
  const expZ = Math.exp(z)
  return expZ / (1 + expZ)
}

function tanhActivation(z: number): number {
  return Math.tanh(z)
}

/**
 * L2 norm of a vector.
 */
function vecNorm(v: number[]): number {
  let sum = 0
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i]
  }
  return Math.sqrt(sum)
}

// ── Vanilla RNN ──────────────────────────────────────────────────────

/**
 * Run a vanilla (Elman) RNN forward on a 1-D input sequence.
 *
 * Architecture:
 *   h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
 *   y_t = W_ho * h_t + b_o
 *
 * After the forward pass, we compute per-time-step gradient magnitudes
 * via Backpropagation Through Time (BPTT).  For a vanilla RNN, these
 * magnitudes decrease exponentially, clearly demonstrating the vanishing
 * gradient problem.
 *
 * @param sequence  Array of scalar input values
 * @param config    hiddenSize (default 8), seed (default 42)
 */
export function runRNN(
  sequence: number[],
  config: { hiddenSize?: number; seed?: number } = {}
): RNNResult {
  const { hiddenSize = 8, seed = 42 } = config
  const T = sequence.length
  if (T === 0) return { steps: [], gradientMagnitudes: [] }

  const rng = createRng(seed)
  const inputSize = 1
  const outputSize = 1

  // ── Initialise weights (Xavier-ish) ──
  const std_ih = Math.sqrt(2 / (inputSize + hiddenSize))
  const std_hh = Math.sqrt(2 / (hiddenSize + hiddenSize))
  const std_ho = Math.sqrt(2 / (hiddenSize + outputSize))

  // W_ih: inputSize x hiddenSize
  const W_ih: number[][] = Array.from({ length: inputSize }, () =>
    Array.from({ length: hiddenSize }, () => normalRandom(rng, 0, std_ih))
  )
  // W_hh: hiddenSize x hiddenSize
  const W_hh: number[][] = Array.from({ length: hiddenSize }, () =>
    Array.from({ length: hiddenSize }, () => normalRandom(rng, 0, std_hh))
  )
  // b_h: hiddenSize
  const b_h: number[] = new Array(hiddenSize).fill(0)
  // W_ho: hiddenSize x outputSize
  const W_ho: number[][] = Array.from({ length: hiddenSize }, () =>
    Array.from({ length: outputSize }, () => normalRandom(rng, 0, std_ho))
  )
  // b_o: outputSize
  const b_o: number[] = new Array(outputSize).fill(0)

  // ── Forward pass ──
  const steps: RNNStepResult[] = []
  const hiddenStates: number[][] = [] // store all h_t for BPTT
  const preActivations: number[][] = [] // store all z_t (pre-tanh values)
  let h = new Array(hiddenSize).fill(0) // h_0

  for (let t = 0; t < T; t++) {
    const x = sequence[t]

    // Compute weighted input contributions
    const weightedInput: number[] = new Array(hiddenSize)
    const preAct: number[] = new Array(hiddenSize)
    const newH: number[] = new Array(hiddenSize)

    for (let j = 0; j < hiddenSize; j++) {
      // W_ih * x  (inputSize = 1)
      let z = W_ih[0][j] * x + b_h[j]
      weightedInput[j] = z

      // + W_hh * h_{t-1}
      for (let k = 0; k < hiddenSize; k++) {
        z += W_hh[k][j] * h[k]
      }
      preAct[j] = z
      newH[j] = tanhActivation(z)
    }

    // Output: y_t = W_ho^T * h_t + b_o
    let y = b_o[0]
    for (let j = 0; j < hiddenSize; j++) {
      y += W_ho[j][0] * newH[j]
    }

    preActivations.push(preAct)
    hiddenStates.push([...newH])

    steps.push({
      timeStep: t,
      input: x,
      hiddenState: [...newH],
      output: y,
      intermediates: {
        weightedInput: [...weightedInput],
        preActivation: [...preAct],
      },
    })

    h = newH
  }

  // ── BPTT for gradient magnitudes ──
  // We compute the gradient of the loss at the final time step with respect
  // to the hidden state at each earlier time step.  For a vanilla RNN with
  // tanh, the recurrent Jacobian is diag(1 - h_t^2) * W_hh, and its
  // repeated multiplication causes exponential decay.
  const gradientMagnitudes = computeRNNGradientMagnitudes(
    hiddenStates,
    preActivations,
    W_hh
  )

  return { steps, gradientMagnitudes }
}

/**
 * Compute per-time-step gradient magnitudes for a vanilla RNN via BPTT.
 *
 * We propagate a gradient backward from the final time step through the
 * recurrent connections.  At each step t, the gradient is multiplied by
 *   diag(1 - tanh(z_t)^2) * W_hh^T
 * which typically causes exponential shrinkage.
 */
function computeRNNGradientMagnitudes(
  hiddenStates: number[][],
  _preActivations: number[][],
  W_hh: number[][]
): number[] {
  const T = hiddenStates.length
  if (T === 0) return []

  const hiddenSize = hiddenStates[0].length
  const magnitudes: number[] = new Array(T)

  // Start with gradient = 1 at the final time step (unit upstream gradient)
  let grad: number[] = new Array(hiddenSize).fill(1)
  magnitudes[T - 1] = vecNorm(grad)

  // Propagate backward
  for (let t = T - 2; t >= 0; t--) {
    const newGrad: number[] = new Array(hiddenSize).fill(0)
    // Jacobian: d h_{t+1} / d h_t = diag(1 - tanh(z_{t+1})^2) * W_hh
    // So: grad_t = W_hh^T * (grad_{t+1} .* (1 - h_{t+1}^2))
    const h_next = hiddenStates[t + 1]
    const scaledGrad: number[] = new Array(hiddenSize)
    for (let j = 0; j < hiddenSize; j++) {
      scaledGrad[j] = grad[j] * (1 - h_next[j] * h_next[j])
    }

    for (let i = 0; i < hiddenSize; i++) {
      let sum = 0
      for (let j = 0; j < hiddenSize; j++) {
        sum += W_hh[i][j] * scaledGrad[j]
      }
      newGrad[i] = sum
    }

    grad = newGrad
    magnitudes[t] = vecNorm(grad)
  }

  return magnitudes
}

// ── LSTM ─────────────────────────────────────────────────────────────

/**
 * Run an LSTM forward on a 1-D input sequence.
 *
 * Architecture (standard LSTM with forget gate):
 *   f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)          forget gate
 *   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)          input gate
 *   g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)             cell candidate
 *   c_t = f_t * c_{t-1} + i_t * g_t                     cell state
 *   o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)           output gate
 *   h_t = o_t * tanh(c_t)                                hidden state
 *
 * The LSTM's cell state provides an additive gradient path, which keeps
 * gradient magnitudes more stable compared to the vanilla RNN.
 *
 * @param sequence  Array of scalar input values
 * @param config    hiddenSize (default 8), seed (default 42)
 */
export function runLSTM(
  sequence: number[],
  config: { hiddenSize?: number; seed?: number } = {}
): LSTMResult {
  const { hiddenSize = 8, seed = 42 } = config
  const T = sequence.length
  if (T === 0) return { steps: [], gradientMagnitudes: [] }

  const rng = createRng(seed)
  const inputSize = 1
  const concatSize = hiddenSize + inputSize

  // ── Initialise weights ──
  // Each gate has a weight matrix of shape (concatSize x hiddenSize) + bias
  const std = Math.sqrt(2 / (concatSize + hiddenSize))

  const initMatrix = (): number[][] =>
    Array.from({ length: concatSize }, () =>
      Array.from({ length: hiddenSize }, () => normalRandom(rng, 0, std))
    )

  const W_f = initMatrix() // forget gate
  const W_i = initMatrix() // input gate
  const W_g = initMatrix() // cell candidate
  const W_o = initMatrix() // output gate

  // Bias for forget gate initialised to 1 (common practice to encourage
  // remembering early in training)
  const b_f: number[] = new Array(hiddenSize).fill(1)
  const b_i: number[] = new Array(hiddenSize).fill(0)
  const b_g: number[] = new Array(hiddenSize).fill(0)
  const b_o: number[] = new Array(hiddenSize).fill(0)

  // Helper: compute gate output  gate = activation(W * concat + b)
  const computeGate = (
    W: number[][],
    b: number[],
    concat: number[],
    activation: (z: number) => number
  ): number[] => {
    const result: number[] = new Array(hiddenSize)
    for (let j = 0; j < hiddenSize; j++) {
      let z = b[j]
      for (let k = 0; k < concatSize; k++) {
        z += W[k][j] * concat[k]
      }
      result[j] = activation(z)
    }
    return result
  }

  // ── Forward pass ──
  const steps: LSTMStepResult[] = []
  let h: number[] = new Array(hiddenSize).fill(0) // h_0
  let c: number[] = new Array(hiddenSize).fill(0) // c_0

  // Store gate values for BPTT gradient computation
  const allForgetGates: number[][] = []
  const allInputGates: number[][] = []
  const allOutputGates: number[][] = []
  const allCellCandidates: number[][] = []
  const allCellStates: number[][] = []

  for (let t = 0; t < T; t++) {
    const x = sequence[t]
    const concat: number[] = [...h, x]

    const ft = computeGate(W_f, b_f, concat, sigmoid)
    const it = computeGate(W_i, b_i, concat, sigmoid)
    const gt = computeGate(W_g, b_g, concat, tanhActivation)
    const ot = computeGate(W_o, b_o, concat, sigmoid)

    // Cell state update: c_t = f_t * c_{t-1} + i_t * g_t
    const newC: number[] = new Array(hiddenSize)
    for (let j = 0; j < hiddenSize; j++) {
      newC[j] = ft[j] * c[j] + it[j] * gt[j]
    }

    // Hidden state: h_t = o_t * tanh(c_t)
    const newH: number[] = new Array(hiddenSize)
    for (let j = 0; j < hiddenSize; j++) {
      newH[j] = ot[j] * tanhActivation(newC[j])
    }

    allForgetGates.push([...ft])
    allInputGates.push([...it])
    allOutputGates.push([...ot])
    allCellCandidates.push([...gt])
    allCellStates.push([...newC])

    steps.push({
      timeStep: t,
      input: x,
      forgetGate: [...ft],
      inputGate: [...it],
      cellCandidate: [...gt],
      cellState: [...newC],
      outputGate: [...ot],
      hiddenState: [...newH],
    })

    h = newH
    c = newC
  }

  // ── BPTT for gradient magnitudes ──
  const gradientMagnitudes = computeLSTMGradientMagnitudes(
    allForgetGates,
    allInputGates,
    allOutputGates,
    allCellCandidates,
    allCellStates
  )

  return { steps, gradientMagnitudes }
}

/**
 * Compute per-time-step gradient magnitudes for the LSTM.
 *
 * The key gradient path through the cell state is:
 *   d c_t / d c_{t-1} = f_t  (the forget gate)
 *
 * Because the forget gate values are typically near 1 (especially with
 * bias initialised to 1), gradients flow much more stably through the
 * cell state compared to the vanilla RNN's multiplicative tanh path.
 *
 * We track the gradient through the cell state (the "gradient highway")
 * to show the contrast with vanilla RNN.
 */
function computeLSTMGradientMagnitudes(
  forgetGates: number[][],
  _inputGates: number[][],
  _outputGates: number[][],
  _cellCandidates: number[][],
  _cellStates: number[][]
): number[] {
  const T = forgetGates.length
  if (T === 0) return []

  const hiddenSize = forgetGates[0].length
  const magnitudes: number[] = new Array(T)

  // Start with unit gradient at the final time step
  // We track gradient through the cell state (the main gradient highway)
  let gradC: number[] = new Array(hiddenSize).fill(1)
  magnitudes[T - 1] = vecNorm(gradC)

  // Propagate backward through the cell state
  // d c_t / d c_{t-1} = f_t  (element-wise)
  // Additionally, there are paths through the gates, but the cell state
  // path is the dominant one and the one that demonstrates LSTM's advantage.
  for (let t = T - 2; t >= 0; t--) {
    const newGradC: number[] = new Array(hiddenSize)
    for (let j = 0; j < hiddenSize; j++) {
      // The gradient flows through the forget gate multiplicatively
      newGradC[j] = gradC[j] * forgetGates[t + 1][j]
    }
    gradC = newGradC
    magnitudes[t] = vecNorm(gradC)
  }

  return magnitudes
}
