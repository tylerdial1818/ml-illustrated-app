import { useState, useCallback, useRef, useEffect } from 'react'

interface AlgorithmPlayerConfig<T> {
  snapshots: T[]
  baseFps?: number
}

interface AlgorithmPlayerReturn<T> {
  currentStep: number
  currentSnapshot: T
  totalSteps: number
  isPlaying: boolean
  speed: number
  play: () => void
  pause: () => void
  togglePlay: () => void
  stepForward: () => void
  stepBack: () => void
  reset: () => void
  setSpeed: (speed: number) => void
  goToStep: (step: number) => void
  isAtStart: boolean
  isAtEnd: boolean
}

export function useAlgorithmPlayer<T>({
  snapshots,
  baseFps = 2,
}: AlgorithmPlayerConfig<T>): AlgorithmPlayerReturn<T> {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const rafRef = useRef<number>(0)
  const lastTimeRef = useRef<number>(0)

  const totalSteps = snapshots.length
  const isAtStart = currentStep === 0
  const isAtEnd = currentStep >= totalSteps - 1

  const pause = useCallback(() => {
    setIsPlaying(false)
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = 0
    }
  }, [])

  const play = useCallback(() => {
    if (currentStep >= totalSteps - 1) {
      setCurrentStep(0)
    }
    setIsPlaying(true)
    lastTimeRef.current = 0
  }, [currentStep, totalSteps])

  const togglePlay = useCallback(() => {
    if (isPlaying) {
      pause()
    } else {
      play()
    }
  }, [isPlaying, pause, play])

  const stepForward = useCallback(() => {
    pause()
    setCurrentStep((prev) => Math.min(prev + 1, totalSteps - 1))
  }, [totalSteps, pause])

  const stepBack = useCallback(() => {
    pause()
    setCurrentStep((prev) => Math.max(prev - 1, 0))
  }, [pause])

  const reset = useCallback(() => {
    pause()
    setCurrentStep(0)
  }, [pause])

  const goToStep = useCallback(
    (step: number) => {
      pause()
      setCurrentStep(Math.max(0, Math.min(step, totalSteps - 1)))
    },
    [totalSteps, pause]
  )

  useEffect(() => {
    if (!isPlaying) return

    const animate = (timestamp: number) => {
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp
      }

      const elapsed = timestamp - lastTimeRef.current
      const interval = 1000 / (baseFps * speed)

      if (elapsed >= interval) {
        setCurrentStep((prev) => {
          const next = prev + 1
          if (next >= totalSteps) {
            setIsPlaying(false)
            return totalSteps - 1
          }
          return next
        })
        lastTimeRef.current = timestamp
      }

      rafRef.current = requestAnimationFrame(animate)
    }

    rafRef.current = requestAnimationFrame(animate)

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
      }
    }
  }, [isPlaying, baseFps, speed, totalSteps])

  return {
    currentStep,
    currentSnapshot: snapshots[Math.min(currentStep, totalSteps - 1)] ?? snapshots[0],
    totalSteps,
    isPlaying,
    speed,
    play,
    pause,
    togglePlay,
    stepForward,
    stepBack,
    reset,
    setSpeed,
    goToStep,
    isAtStart,
    isAtEnd,
  }
}
