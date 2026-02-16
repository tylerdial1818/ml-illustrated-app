/**
 * Multi-Head Attention implementation with intermediate states for visualization.
 *
 * MultiHead(X) = Concat(head_1, ..., head_h) * W_O
 * where head_i = Attention(X * W_Q^i, X * W_K^i, X * W_V^i)
 */

import { computeAttention, type AttentionResult } from './attention'
import { matMul } from './softmax'

export interface MultiHeadResult {
  /** Per-head attention results with all intermediate states */
  heads: AttentionResult[]
  /** Concatenated outputs from all heads (seqLen x dModel) */
  concatenated: number[][]
  /** Final output after W_O projection (seqLen x dModel) */
  output: number[][]
  /** Per-head dimension: dModel / numHeads */
  dPerHead: number
}

/**
 * Compute multi-head attention given pre-split Q/K/V for each head.
 *
 * @param headsQ - Q matrices per head (numHeads x seqLen x dPerHead)
 * @param headsK - K matrices per head
 * @param headsV - V matrices per head
 * @param WO - Output projection matrix ((numHeads * dPerHead) x dModel)
 * @param mask - Optional attention mask
 * @returns Full multi-head result with per-head intermediates
 */
export function computeMultiHeadAttention(
  headsQ: number[][][],
  headsK: number[][][],
  headsV: number[][][],
  WO?: number[][],
  mask?: boolean[][]
): MultiHeadResult {
  const numHeads = headsQ.length
  const seqLen = headsQ[0].length
  const dPerHead = headsQ[0][0].length

  // Compute attention for each head independently
  const heads: AttentionResult[] = headsQ.map((Q, h) =>
    computeAttention(Q, headsK[h], headsV[h], mask)
  )

  // Concatenate head outputs: for each position, stack all head outputs
  const concatenated: number[][] = Array.from({ length: seqLen }, (_, pos) => {
    const row: number[] = []
    for (let h = 0; h < numHeads; h++) {
      row.push(...heads[h].output[pos])
    }
    return row
  })

  // Apply output projection W_O if provided
  const output = WO ? matMul(concatenated, WO) : concatenated

  return {
    heads,
    concatenated,
    output,
    dPerHead,
  }
}

/**
 * Average attention weights across multiple heads.
 * Useful for visualization: shows combined attention pattern.
 *
 * @param headWeights - Array of attention weight matrices (numHeads x seqLen x seqLen)
 * @returns Averaged attention matrix (seqLen x seqLen)
 */
export function averageAttentionWeights(headWeights: number[][][]): number[][] {
  const numHeads = headWeights.length
  const seqLen = headWeights[0].length

  return Array.from({ length: seqLen }, (_, i) =>
    Array.from({ length: seqLen }, (_, j) => {
      let sum = 0
      for (let h = 0; h < numHeads; h++) {
        sum += headWeights[h][i][j]
      }
      return sum / numHeads
    })
  )
}

/**
 * Split a set of attention weights into the specified number of heads
 * by grouping and averaging from a source set of weights.
 *
 * Used for visualization: show how patterns change with different head counts.
 *
 * @param sourceWeights - Source weight matrices (e.g., 4 heads)
 * @param targetHeads - Desired number of output heads (1, 2, 4, or 8)
 * @returns Weight matrices for the target number of heads
 */
export function adaptHeadCount(
  sourceWeights: number[][][],
  targetHeads: number
): number[][][] {
  const sourceCount = sourceWeights.length

  if (targetHeads === sourceCount) {
    return sourceWeights
  }

  if (targetHeads < sourceCount) {
    // Merge heads by averaging groups
    const groupSize = sourceCount / targetHeads
    const result: number[][][] = []
    for (let g = 0; g < targetHeads; g++) {
      const startIdx = Math.floor(g * groupSize)
      const endIdx = Math.floor((g + 1) * groupSize)
      const group = sourceWeights.slice(startIdx, endIdx)
      result.push(averageAttentionWeights(group))
    }
    return result
  }

  // targetHeads > sourceCount: create variants by mixing with slight noise
  const result: number[][][] = []
  for (let h = 0; h < targetHeads; h++) {
    const baseIdx = h % sourceCount
    const base = sourceWeights[baseIdx]

    // Create a variation: sharpen or soften the pattern
    const sharpen = h >= sourceCount // second pass gets sharpened versions
    const variant = base.map((row) => {
      const adjusted = row.map((w) => {
        if (sharpen) {
          // Sharpen: push high values higher, low values lower
          return Math.pow(w, 0.7)
        }
        return w
      })
      // Re-normalize so rows sum to 1
      const sum = adjusted.reduce((a, b) => a + b, 0)
      return adjusted.map((v) => v / sum)
    })

    result.push(variant)
  }
  return result
}
