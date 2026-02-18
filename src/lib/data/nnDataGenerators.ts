import { createRng, normalRandom } from '../math/random'

export interface ClassificationPoint {
  x: number
  y: number
  label: number
}

/** Classic XOR data - perceptron failure case */
export function makeXOR(n: number, noise = 0.15, seed = 42): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const perQuadrant = Math.floor(n / 4)

  const centers = [
    { x: -1, y: -1, label: 0 },
    { x: 1, y: 1, label: 0 },
    { x: -1, y: 1, label: 1 },
    { x: 1, y: -1, label: 1 },
  ]

  for (let q = 0; q < 4; q++) {
    const count = q === 3 ? n - points.length : perQuadrant
    for (let i = 0; i < count; i++) {
      points.push({
        x: normalRandom(rng, centers[q].x, noise * 2),
        y: normalRandom(rng, centers[q].y, noise * 2),
        label: centers[q].label,
      })
    }
  }

  return points
}

/** Moon-shaped data for MLP */
export function makeMoons(n: number, noise = 0.1, seed = 42): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    const angle = (Math.PI * i) / half
    points.push({
      x: Math.cos(angle) + normalRandom(rng, 0, noise),
      y: Math.sin(angle) + normalRandom(rng, 0, noise),
      label: 0,
    })
  }

  for (let i = 0; i < n - half; i++) {
    const angle = (Math.PI * i) / (n - half)
    points.push({
      x: 1 - Math.cos(angle) + normalRandom(rng, 0, noise),
      y: 0.5 - Math.sin(angle) + normalRandom(rng, 0, noise),
      label: 1,
    })
  }

  return points
}

/** Spiral data - hard non-linear classification */
export function makeSpirals(n: number, noise = 0.2, seed = 42): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    const r = (5 * i) / half
    const angle = (1.75 * i) / half * Math.PI + Math.PI
    points.push({
      x: r * Math.sin(angle) + normalRandom(rng, 0, noise),
      y: r * Math.cos(angle) + normalRandom(rng, 0, noise),
      label: 0,
    })
  }

  for (let i = 0; i < n - half; i++) {
    const r = (5 * i) / (n - half)
    const angle = (1.75 * i) / (n - half) * Math.PI
    points.push({
      x: r * Math.sin(angle) + normalRandom(rng, 0, noise),
      y: r * Math.cos(angle) + normalRandom(rng, 0, noise),
      label: 1,
    })
  }

  return points
}

/** Concentric circles */
export function makeConcentricCircles(
  n: number,
  noise = 0.05,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    const angle = (2 * Math.PI * i) / half
    points.push({
      x: Math.cos(angle) + normalRandom(rng, 0, noise),
      y: Math.sin(angle) + normalRandom(rng, 0, noise),
      label: 0,
    })
  }

  for (let i = 0; i < n - half; i++) {
    const angle = (2 * Math.PI * i) / (n - half)
    points.push({
      x: 0.4 * Math.cos(angle) + normalRandom(rng, 0, noise),
      y: 0.4 * Math.sin(angle) + normalRandom(rng, 0, noise),
      label: 1,
    })
  }

  return points
}

/** Linearly separable data for perceptron */
export function makeLinearSeparable(
  n: number,
  separation = 2,
  noise = 0.3,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    points.push({
      x: normalRandom(rng, -separation / 2, noise),
      y: normalRandom(rng, -separation / 4, noise),
      label: 0,
    })
  }

  for (let i = 0; i < n - half; i++) {
    points.push({
      x: normalRandom(rng, separation / 2, noise),
      y: normalRandom(rng, separation / 4, noise),
      label: 1,
    })
  }

  return points
}

/** Gaussian blobs for MLP */
export function makeGaussianBlobs(
  n: number,
  nClusters = 3,
  seed = 42
): ClassificationPoint[] {
  const rng = createRng(seed)
  const points: ClassificationPoint[] = []
  const perCluster = Math.floor(n / nClusters)

  const centers = [
    { x: 0, y: 2 },
    { x: -2, y: -1 },
    { x: 2, y: -1 },
    { x: 0, y: -2 },
  ]

  for (let c = 0; c < nClusters; c++) {
    const count = c === nClusters - 1 ? n - points.length : perCluster
    const center = centers[c % centers.length]
    for (let i = 0; i < count; i++) {
      points.push({
        x: normalRandom(rng, center.x, 0.5),
        y: normalRandom(rng, center.y, 0.5),
        label: c % 2,
      })
    }
  }

  return points
}

/** Simple digit grid for CNN demos (8x8 pixel grid) */
export function makeDigitGrid(
  digit: number,
  size = 8
): number[][] {
  const grid: number[][] = Array.from({ length: size }, () =>
    new Array(size).fill(0)
  )

  // Simple pixel patterns for digits 0-9
  const patterns: Record<number, [number, number][]> = {
    0: [[1,1],[1,2],[1,3],[1,4],[1,5],[2,1],[2,5],[3,1],[3,5],[4,1],[4,5],[5,1],[5,2],[5,3],[5,4],[5,5]],
    1: [[1,3],[2,2],[2,3],[3,3],[4,3],[5,3],[5,2],[5,4]],
    2: [[1,1],[1,2],[1,3],[1,4],[1,5],[2,5],[3,1],[3,2],[3,3],[3,4],[3,5],[4,1],[5,1],[5,2],[5,3],[5,4],[5,5]],
    3: [[1,1],[1,2],[1,3],[1,4],[1,5],[2,5],[3,2],[3,3],[3,4],[3,5],[4,5],[5,1],[5,2],[5,3],[5,4],[5,5]],
    4: [[1,1],[1,5],[2,1],[2,5],[3,1],[3,2],[3,3],[3,4],[3,5],[4,5],[5,5]],
    5: [[1,1],[1,2],[1,3],[1,4],[1,5],[2,1],[3,1],[3,2],[3,3],[3,4],[3,5],[4,5],[5,1],[5,2],[5,3],[5,4],[5,5]],
    6: [[1,1],[1,2],[1,3],[1,4],[1,5],[2,1],[3,1],[3,2],[3,3],[3,4],[3,5],[4,1],[4,5],[5,1],[5,2],[5,3],[5,4],[5,5]],
    7: [[1,1],[1,2],[1,3],[1,4],[1,5],[2,5],[3,4],[4,3],[5,3]],
    8: [[1,1],[1,2],[1,3],[1,4],[1,5],[2,1],[2,5],[3,1],[3,2],[3,3],[3,4],[3,5],[4,1],[4,5],[5,1],[5,2],[5,3],[5,4],[5,5]],
    9: [[1,1],[1,2],[1,3],[1,4],[1,5],[2,1],[2,5],[3,1],[3,2],[3,3],[3,4],[3,5],[4,5],[5,1],[5,2],[5,3],[5,4],[5,5]],
  }

  const pts = patterns[digit] || patterns[0]
  for (const [r, c] of pts) {
    if (r < size && c < size) {
      grid[r][c] = 1
    }
  }

  return grid
}

/** Sequence data for RNN demos */
export function makeSequence(
  length: number,
  type: 'sine' | 'sawtooth' | 'square' = 'sine',
  seed = 42
): number[] {
  const rng = createRng(seed)
  const seq: number[] = []

  for (let i = 0; i < length; i++) {
    const t = i / length
    let value: number

    switch (type) {
      case 'sine':
        value = Math.sin(t * Math.PI * 4) + normalRandom(rng, 0, 0.05)
        break
      case 'sawtooth':
        value = 2 * (t * 2 - Math.floor(t * 2 + 0.5)) + normalRandom(rng, 0, 0.05)
        break
      case 'square':
        value = Math.sin(t * Math.PI * 4) > 0 ? 1 : -1
        value += normalRandom(rng, 0, 0.05)
        break
    }

    seq.push(value)
  }

  return seq
}

/** 2D Gaussian data for GAN distribution matching */
export function makeGaussian2D(
  n: number,
  means: [number, number][] = [[0, 0]],
  spread = 0.5,
  seed = 42
): { x: number; y: number }[] {
  const rng = createRng(seed)
  const points: { x: number; y: number }[] = []
  const perCluster = Math.floor(n / means.length)

  for (let c = 0; c < means.length; c++) {
    const count = c === means.length - 1 ? n - points.length : perCluster
    for (let i = 0; i < count; i++) {
      points.push({
        x: normalRandom(rng, means[c][0], spread),
        y: normalRandom(rng, means[c][1], spread),
      })
    }
  }

  return points
}
