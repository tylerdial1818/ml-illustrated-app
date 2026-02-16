/**
 * Scaled Dot-Product Attention implementation with all intermediate states exposed.
 *
 * Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 *
 * All intermediate results are exposed for visualization and education.
 */

import { softmax, matMul, transpose } from './softmax'

export interface AttentionResult {
  /** Raw scores: QK^T (seqLen x seqLen) */
  rawScores: number[][]
  /** Scaled scores: QK^T / sqrt(dk) (seqLen x seqLen) */
  scaledScores: number[][]
  /** Scores after mask application, if mask was provided */
  maskedScores?: number[][]
  /** Attention weights after softmax (seqLen x seqLen, rows sum to 1) */
  attentionWeights: number[][]
  /** Output: weighted sum of V (seqLen x dv) */
  output: number[][]
}

/**
 * Compute full scaled dot-product attention with all intermediate states.
 *
 * @param Q - Query matrix (seqLen x dk)
 * @param K - Key matrix (seqLen x dk)
 * @param V - Value matrix (seqLen x dv)
 * @param mask - Optional boolean mask (seqLen x seqLen). true = attend, false = mask out
 * @returns All intermediate attention computation states
 */
export function computeAttention(
  Q: number[][],
  K: number[][],
  V: number[][],
  mask?: boolean[][]
): AttentionResult {
  // Step 1: Compute raw scores QK^T
  const rawScores = computeScores(Q, K)

  // Step 2: Scale by sqrt(dk)
  const dk = K[0].length
  const scaledScores = applyScaling(rawScores, dk)

  // Step 3: Optionally apply mask
  let scoresToSoftmax = scaledScores
  let maskedScores: number[][] | undefined

  if (mask) {
    maskedScores = applyMask(scaledScores, mask)
    scoresToSoftmax = maskedScores
  }

  // Step 4: Apply softmax
  const attentionWeights = applySoftmax(scoresToSoftmax)

  // Step 5: Compute output
  const output = computeOutput(attentionWeights, V)

  return {
    rawScores,
    scaledScores,
    maskedScores,
    attentionWeights,
    output,
  }
}

/**
 * Compute raw attention scores: QK^T
 *
 * @param Q - Query matrix (seqLen x dk)
 * @param K - Key matrix (seqLen x dk)
 * @returns Score matrix (seqLen x seqLen)
 */
export function computeScores(Q: number[][], K: number[][]): number[][] {
  const KT = transpose(K)
  return matMul(Q, KT)
}

/**
 * Apply scaling factor to scores: scores / sqrt(dk)
 *
 * @param scores - Score matrix (seqLen x seqLen)
 * @param dk - Dimension of keys
 * @returns Scaled score matrix
 */
export function applyScaling(scores: number[][], dk: number): number[][] {
  const scale = Math.sqrt(dk)
  return scores.map((row) => row.map((val) => val / scale))
}

/**
 * Apply a boolean mask to scores.
 * Positions where mask is false are set to -Infinity (will become 0 after softmax).
 *
 * @param scores - Score matrix (seqLen x seqLen)
 * @param mask - Boolean mask. true = keep, false = mask out (-Infinity)
 * @returns Masked score matrix
 */
export function applyMask(
  scores: number[][],
  mask: boolean[][]
): number[][] {
  return scores.map((row, i) =>
    row.map((val, j) => (mask[i][j] ? val : -Infinity))
  )
}

/**
 * Apply softmax row-wise to produce attention weights.
 * Each row of the output sums to 1.0.
 *
 * @param scores - Score matrix (seqLen x seqLen)
 * @returns Attention weight matrix where each row is a probability distribution
 */
export function applySoftmax(scores: number[][]): number[][] {
  return scores.map((row) => softmax(row))
}

/**
 * Compute attention output: weights * V
 *
 * @param weights - Attention weight matrix (seqLen x seqLen)
 * @param V - Value matrix (seqLen x dv)
 * @returns Output matrix (seqLen x dv)
 */
export function computeOutput(
  weights: number[][],
  V: number[][]
): number[][] {
  return matMul(weights, V)
}

/**
 * Create a causal (lower-triangular) mask for autoregressive attention.
 * Each position can only attend to itself and earlier positions.
 *
 * Example for size=4:
 *   [[true,  false, false, false],
 *    [true,  true,  false, false],
 *    [true,  true,  true,  false],
 *    [true,  true,  true,  true ]]
 *
 * @param size - Sequence length
 * @returns Boolean mask matrix (size x size)
 */
export function createCausalMask(size: number): boolean[][] {
  const mask: boolean[][] = []

  for (let i = 0; i < size; i++) {
    const row: boolean[] = []
    for (let j = 0; j < size; j++) {
      row.push(j <= i)
    }
    mask.push(row)
  }

  return mask
}
