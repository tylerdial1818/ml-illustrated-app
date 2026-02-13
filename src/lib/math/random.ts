// Simple seeded PRNG (mulberry32)
export function createRng(seed: number) {
  let s = seed | 0
  return function random(): number {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Box-Muller transform for normal distribution
export function normalRandom(rng: () => number, mean = 0, std = 1): number {
  const u1 = rng()
  const u2 = rng()
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  return mean + z * std
}

// Generate n samples from a 2D normal distribution
export function normal2D(
  rng: () => number,
  n: number,
  meanX: number,
  meanY: number,
  stdX: number,
  stdY: number
): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = []
  for (let i = 0; i < n; i++) {
    points.push({
      x: normalRandom(rng, meanX, stdX),
      y: normalRandom(rng, meanY, stdY),
    })
  }
  return points
}

// Shuffle array in place
export function shuffle<T>(arr: T[], rng: () => number): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}
