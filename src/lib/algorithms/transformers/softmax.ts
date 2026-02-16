/**
 * Math utilities for transformer computations.
 * Pure TypeScript implementations - no external dependencies.
 */

/**
 * Compute softmax over an array of values.
 * Uses the numerically stable version (subtract max before exp).
 */
export function softmax(values: number[]): number[] {
  if (values.length === 0) return []

  const max = Math.max(...values)
  const exps = values.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)

  return exps.map((e) => e / sum)
}

/**
 * Compute softmax with temperature scaling.
 * Higher temperature -> more uniform distribution.
 * Lower temperature -> more peaked distribution.
 * Temperature of 1.0 is standard softmax.
 */
export function softmaxWithTemperature(
  values: number[],
  temperature: number
): number[] {
  if (temperature <= 0) {
    throw new Error('Temperature must be positive')
  }

  const scaled = values.map((v) => v / temperature)
  return softmax(scaled)
}

/**
 * Compute cosine similarity between two vectors.
 * Returns a value between -1 and 1.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vector length mismatch: ${a.length} vs ${b.length}`
    )
  }
  if (a.length === 0) return 0

  const dot = dotProduct(a, b)
  const magA = Math.sqrt(dotProduct(a, a))
  const magB = Math.sqrt(dotProduct(b, b))

  if (magA === 0 || magB === 0) return 0

  return dot / (magA * magB)
}

/**
 * Compute the dot product of two vectors.
 */
export function dotProduct(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vector length mismatch: ${a.length} vs ${b.length}`
    )
  }

  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i]
  }
  return sum
}

/**
 * Matrix multiplication: A (m x n) * B (n x p) -> C (m x p)
 */
export function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length
  if (m === 0) return []

  const n = A[0].length
  const p = B[0].length

  if (B.length !== n) {
    throw new Error(
      `Matrix dimension mismatch: A is ${m}x${n}, B is ${B.length}x${p}`
    )
  }

  const C: number[][] = Array.from({ length: m }, () =>
    new Array(p).fill(0)
  )

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      let sum = 0
      for (let k = 0; k < n; k++) {
        sum += A[i][k] * B[k][j]
      }
      C[i][j] = sum
    }
  }

  return C
}

/**
 * Transpose a matrix: A (m x n) -> A^T (n x m)
 */
export function transpose(A: number[][]): number[][] {
  if (A.length === 0) return []

  const m = A.length
  const n = A[0].length

  const result: number[][] = Array.from({ length: n }, () =>
    new Array(m).fill(0)
  )

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      result[j][i] = A[i][j]
    }
  }

  return result
}
