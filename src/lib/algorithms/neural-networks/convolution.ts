// ── Types ────────────────────────────────────────────────────────────

export interface ConvolutionStep {
  position: { row: number; col: number }
  inputPatch: number[][] // the patch being convolved
  kernel: number[][]
  products: number[][] // element-wise multiplication result
  outputValue: number // sum of products
}

export interface ConvolutionResult {
  steps: ConvolutionStep[]
  outputGrid: number[][]
  outputSize: { rows: number; cols: number }
}

export interface PoolingStep {
  row: number
  col: number
  patch: number[][]
  maxValue: number
}

// ── Preset kernels ───────────────────────────────────────────────────

export const PRESET_KERNELS: Record<string, number[][]> = {
  horizontal_edge: [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
  ],
  vertical_edge: [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
  ],
  diagonal: [
    [0, -1, -1],
    [1, 0, -1],
    [1, 1, 0],
  ],
  blur: [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
  emboss: [
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
  ],
}

// ── 2-D convolution ──────────────────────────────────────────────────

/**
 * Perform a 2-D convolution of `input` with `kernel`.
 *
 * Every position of the sliding kernel window is recorded as a
 * {@link ConvolutionStep} so the UI can animate the operation
 * element-by-element.
 *
 * @param input   2-D numeric grid (rows x cols)
 * @param kernel  2-D kernel (typically 3x3 or 5x5)
 * @param stride  Step size for the sliding window (default 1)
 * @param padding `'valid'` (no padding, default) or `'same'` (zero-pad to
 *                preserve input dimensions)
 */
export function convolve2D(
  input: number[][],
  kernel: number[][],
  stride: number = 1,
  padding: 'valid' | 'same' = 'valid'
): ConvolutionResult {
  const inputRows = input.length
  const inputCols = input[0].length
  const kRows = kernel.length
  const kCols = kernel[0].length

  // Compute padding amounts
  let padTop = 0
  let padLeft = 0
  let paddedInput = input

  if (padding === 'same') {
    padTop = Math.floor(kRows / 2)
    padLeft = Math.floor(kCols / 2)
    const padBottom = kRows - 1 - padTop
    const padRight = kCols - 1 - padLeft
    const paddedRows = inputRows + padTop + padBottom
    const paddedCols = inputCols + padLeft + padRight

    paddedInput = Array.from({ length: paddedRows }, (_, r) =>
      Array.from({ length: paddedCols }, (_, c) => {
        const srcR = r - padTop
        const srcC = c - padLeft
        if (srcR >= 0 && srcR < inputRows && srcC >= 0 && srcC < inputCols) {
          return input[srcR][srcC]
        }
        return 0
      })
    )
  }

  const pRows = paddedInput.length
  const pCols = paddedInput[0].length

  // Output dimensions
  const outRows = Math.floor((pRows - kRows) / stride) + 1
  const outCols = Math.floor((pCols - kCols) / stride) + 1

  const outputGrid: number[][] = Array.from({ length: outRows }, () =>
    new Array(outCols).fill(0)
  )

  const steps: ConvolutionStep[] = []

  for (let outR = 0; outR < outRows; outR++) {
    for (let outC = 0; outC < outCols; outC++) {
      const startR = outR * stride
      const startC = outC * stride

      // Extract the patch under the kernel
      const inputPatch: number[][] = []
      const products: number[][] = []
      let sum = 0

      for (let kr = 0; kr < kRows; kr++) {
        const patchRow: number[] = []
        const prodRow: number[] = []
        for (let kc = 0; kc < kCols; kc++) {
          const val = paddedInput[startR + kr][startC + kc]
          const prod = val * kernel[kr][kc]
          patchRow.push(val)
          prodRow.push(prod)
          sum += prod
        }
        inputPatch.push(patchRow)
        products.push(prodRow)
      }

      outputGrid[outR][outC] = sum

      steps.push({
        position: { row: outR, col: outC },
        inputPatch,
        kernel: kernel.map((row) => [...row]),
        products,
        outputValue: sum,
      })
    }
  }

  return {
    steps,
    outputGrid,
    outputSize: { rows: outRows, cols: outCols },
  }
}

// ── Max pooling ──────────────────────────────────────────────────────

/**
 * Apply 2-D max pooling with a square pool window.
 *
 * Stride defaults to `poolSize` (non-overlapping pooling).
 *
 * @param input    2-D numeric grid
 * @param poolSize Side length of the pooling window (default 2)
 */
export function maxPool2D(
  input: number[][],
  poolSize: number = 2
): { output: number[][]; steps: PoolingStep[] } {
  const inputRows = input.length
  const inputCols = input[0].length
  const outRows = Math.floor(inputRows / poolSize)
  const outCols = Math.floor(inputCols / poolSize)

  const output: number[][] = Array.from({ length: outRows }, () =>
    new Array(outCols).fill(0)
  )

  const steps: PoolingStep[] = []

  for (let outR = 0; outR < outRows; outR++) {
    for (let outC = 0; outC < outCols; outC++) {
      const startR = outR * poolSize
      const startC = outC * poolSize

      const patch: number[][] = []
      let maxVal = -Infinity

      for (let pr = 0; pr < poolSize; pr++) {
        const row: number[] = []
        for (let pc = 0; pc < poolSize; pc++) {
          const val = input[startR + pr][startC + pc]
          row.push(val)
          if (val > maxVal) maxVal = val
        }
        patch.push(row)
      }

      output[outR][outC] = maxVal

      steps.push({
        row: outR,
        col: outC,
        patch,
        maxValue: maxVal,
      })
    }
  }

  return { output, steps }
}
