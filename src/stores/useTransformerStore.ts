import { create } from 'zustand'

export interface PrecomputedAttention {
  sentence: string
  tokens: string[]
  tokenIds: number[]
  embeddings: number[][]
  positionalEncodings: number[][]
  layers: {
    selfAttention: {
      heads: {
        queries: number[][]
        keys: number[][]
        values: number[][]
        scores: number[][]
        weights: number[][]
        output: number[][]
      }[]
      combinedOutput: number[][]
    }
    ffnOutput: number[][]
    layerOutput: number[][]
  }[]
}

interface TransformerState {
  selectedToken: number | null
  activeSentence: string
  precomputedData: PrecomputedAttention | null
  setSelectedToken: (token: number | null) => void
  setActiveSentence: (sentence: string) => void
  setPrecomputedData: (data: PrecomputedAttention | null) => void
}

export const useTransformerStore = create<TransformerState>((set) => ({
  selectedToken: null,
  activeSentence: 'The cat sat on the mat',
  precomputedData: null,
  setSelectedToken: (token) => set({ selectedToken: token }),
  setActiveSentence: (sentence) => set({ activeSentence: sentence }),
  setPrecomputedData: (data) => set({ precomputedData: data }),
}))
