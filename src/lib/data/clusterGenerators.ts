import type { Point2D } from '../../types'
import { createRng, normalRandom } from '../math/random'

export function makeBlobs(
  n: number,
  k: number,
  spread = 1.5,
  seed = 42
): Point2D[] {
  const rng = createRng(seed)
  const points: Point2D[] = []
  const perCluster = Math.floor(n / k)

  // Generate cluster centers spread around the space
  const centers: Point2D[] = []
  for (let i = 0; i < k; i++) {
    const angle = (2 * Math.PI * i) / k + normalRandom(rng, 0, 0.3)
    const radius = 3 + rng() * 2
    centers.push({
      x: radius * Math.cos(angle),
      y: radius * Math.sin(angle),
    })
  }

  for (let c = 0; c < k; c++) {
    const count = c === k - 1 ? n - points.length : perCluster
    for (let i = 0; i < count; i++) {
      points.push({
        x: normalRandom(rng, centers[c].x, spread),
        y: normalRandom(rng, centers[c].y, spread),
      })
    }
  }

  return points
}

export function makeMoons(n: number, noise = 0.1, seed = 42): Point2D[] {
  const rng = createRng(seed)
  const points: Point2D[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    const angle = (Math.PI * i) / half
    points.push({
      x: Math.cos(angle) + normalRandom(rng, 0, noise),
      y: Math.sin(angle) + normalRandom(rng, 0, noise),
    })
  }

  for (let i = 0; i < n - half; i++) {
    const angle = (Math.PI * i) / (n - half)
    points.push({
      x: 1 - Math.cos(angle) + normalRandom(rng, 0, noise),
      y: 0.5 - Math.sin(angle) + normalRandom(rng, 0, noise),
    })
  }

  return points
}

export function makeCircles(n: number, noise = 0.05, seed = 42): Point2D[] {
  const rng = createRng(seed)
  const points: Point2D[] = []
  const half = Math.floor(n / 2)

  for (let i = 0; i < half; i++) {
    const angle = (2 * Math.PI * i) / half
    points.push({
      x: Math.cos(angle) + normalRandom(rng, 0, noise),
      y: Math.sin(angle) + normalRandom(rng, 0, noise),
    })
  }

  for (let i = 0; i < n - half; i++) {
    const angle = (2 * Math.PI * i) / (n - half)
    const r = 0.4
    points.push({
      x: r * Math.cos(angle) + normalRandom(rng, 0, noise),
      y: r * Math.sin(angle) + normalRandom(rng, 0, noise),
    })
  }

  return points
}

export function makeVaryingDensity(n: number, seed = 42): Point2D[] {
  const rng = createRng(seed)
  const points: Point2D[] = []

  // Dense cluster
  const n1 = Math.floor(n * 0.5)
  for (let i = 0; i < n1; i++) {
    points.push({
      x: normalRandom(rng, -3, 0.5),
      y: normalRandom(rng, 0, 0.5),
    })
  }

  // Sparse cluster
  const n2 = Math.floor(n * 0.3)
  for (let i = 0; i < n2; i++) {
    points.push({
      x: normalRandom(rng, 3, 2),
      y: normalRandom(rng, 0, 2),
    })
  }

  // Medium cluster
  const n3 = n - n1 - n2
  for (let i = 0; i < n3; i++) {
    points.push({
      x: normalRandom(rng, 0, 1),
      y: normalRandom(rng, 4, 1),
    })
  }

  return points
}

export function addNoise(points: Point2D[], ratio: number, seed = 99): Point2D[] {
  const rng = createRng(seed)
  const noiseCount = Math.floor(points.length * ratio)
  const result = [...points]

  // Find data bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const p of points) {
    minX = Math.min(minX, p.x)
    maxX = Math.max(maxX, p.x)
    minY = Math.min(minY, p.y)
    maxY = Math.max(maxY, p.y)
  }

  const padX = (maxX - minX) * 0.2
  const padY = (maxY - minY) * 0.2

  for (let i = 0; i < noiseCount; i++) {
    result.push({
      x: minX - padX + rng() * (maxX - minX + 2 * padX),
      y: minY - padY + rng() * (maxY - minY + 2 * padY),
    })
  }

  return result
}
