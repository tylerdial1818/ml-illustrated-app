/**
 * Sinusoidal Positional Encoding for Transformers.
 *
 * Implements the original positional encoding from "Attention Is All You Need":
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 */

/**
 * Compute the full positional encoding matrix.
 *
 * @param seqLength - Number of positions (sequence length)
 * @param dModel - Dimension of the model (embedding size)
 * @returns A [seqLength x dModel] matrix of positional encoding values
 */
export function computePositionalEncoding(
  seqLength: number,
  dModel: number
): number[][] {
  const pe: number[][] = []

  for (let pos = 0; pos < seqLength; pos++) {
    const row: number[] = []
    for (let i = 0; i < dModel; i++) {
      const dimIndex = Math.floor(i / 2)
      const angle = pos / Math.pow(10000, (2 * dimIndex) / dModel)

      if (i % 2 === 0) {
        // Even dimensions: sin
        row.push(Math.sin(angle))
      } else {
        // Odd dimensions: cos
        row.push(Math.cos(angle))
      }
    }
    pe.push(row)
  }

  return pe
}

/**
 * Get individual sinusoidal curves for visualization.
 * Returns each dimension's curve across all positions,
 * useful for plotting how different dimensions encode position.
 *
 * @param dModel - Dimension of the model
 * @param maxLen - Maximum sequence length to compute
 * @returns Array of curve objects, each with a dimension index and array of values
 */
export function getSinusoidalCurves(
  dModel: number,
  maxLen: number
): { dim: number; values: number[] }[] {
  const curves: { dim: number; values: number[] }[] = []

  for (let i = 0; i < dModel; i++) {
    const dimIndex = Math.floor(i / 2)
    const values: number[] = []

    for (let pos = 0; pos < maxLen; pos++) {
      const angle = pos / Math.pow(10000, (2 * dimIndex) / dModel)

      if (i % 2 === 0) {
        values.push(Math.sin(angle))
      } else {
        values.push(Math.cos(angle))
      }
    }

    curves.push({ dim: i, values })
  }

  return curves
}
