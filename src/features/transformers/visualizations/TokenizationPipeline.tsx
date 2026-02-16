import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import { Select } from '../../../components/ui/Select'
import { Toggle } from '../../../components/ui/Toggle'

// ── NLP Colors ────────────────────────────────────────────────────────
const NLP = {
  value: '#FBBF24',
  tokenHighlight: '#818CF8',
}

// ── Category Colors for Embedding Space ───────────────────────────────
const CATEGORY_COLORS: Record<string, string> = {
  royalty: '#F472B6',
  animals: '#34D399',
  actions: '#FBBF24',
  food: '#FB923C',
  colors: '#A78BFA',
  numbers: '#38BDF8',
  body: '#E879F9',
  nature: '#4ADE80',
}

// ── Tokenizer Logic (inline) ──────────────────────────────────────────

const COMMON_WORDS = new Set([
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
  'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
  'up', 'about', 'into', 'through', 'during', 'before', 'after',
  'above', 'below', 'between', 'out', 'off', 'over', 'under',
  'again', 'further', 'then', 'once', 'here', 'there', 'when',
  'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more',
  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
  'own', 'same', 'so', 'than', 'too', 'very', 'just', 'because',
  'but', 'and', 'or', 'if', 'while', 'as', 'until', 'that',
  'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its',
  'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
  'she', 'her', 'they', 'them', 'their', 'what', 'am',
  'cat', 'sat', 'mat', 'dog', 'run', 'eat', 'see', 'go', 'come',
  'make', 'take', 'know', 'think', 'say', 'get', 'give', 'look',
  'find', 'want', 'tell', 'work', 'call', 'try', 'ask', 'put',
  'keep', 'let', 'begin', 'seem', 'help', 'show', 'hear', 'play',
  'move', 'live', 'believe', 'bring', 'happen', 'write', 'sit',
  'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set',
  'learn', 'change', 'lead', 'under', 'stand', 'watch', 'follow',
  'stop', 'create', 'speak', 'read', 'spend', 'grow', 'open',
  'walk', 'win', 'teach', 'offer', 'remember', 'love', 'consider',
  'appear', 'buy', 'wait', 'serve', 'die', 'send', 'build', 'stay',
  'fall', 'cut', 'reach', 'kill', 'remain', 'man', 'woman', 'king',
  'queen', 'boy', 'girl', 'child', 'people', 'day', 'time', 'year',
  'way', 'thing', 'world', 'life', 'hand', 'part', 'place', 'case',
  'week', 'company', 'system', 'program', 'question', 'government',
  'number', 'night', 'point', 'home', 'water', 'room', 'mother',
  'area', 'money', 'story', 'fact', 'month', 'lot', 'right',
  'study', 'book', 'eye', 'job', 'word', 'business', 'issue',
  'side', 'kind', 'head', 'house', 'service', 'friend', 'father',
  'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car',
  'city', 'community', 'name', 'president', 'team', 'minute',
  'idea', 'big', 'small', 'long', 'new', 'old', 'great', 'good',
  'bad', 'high', 'low', 'right', 'left', 'young', 'different',
  'important', 'black', 'white', 'red', 'blue', 'green', 'happy',
  'on', 'off',
])

const SUBWORD_SUFFIXES = ['ing', 'tion', 'able', 'ment', 'ly', 'ed', 'er', 'est', 'ness', 'ful', 'less', 'ous', 'ive', 'al', 'ize']
const SUBWORD_PREFIXES = ['un', 're', 'pre', 'dis', 'mis', 'over', 'out', 'sub', 'inter', 'trans']

function tokenizeWord(text: string): string[] {
  return text
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 0)
}

function tokenizeSubword(text: string): string[] {
  const words = text.toLowerCase().split(/\s+/).filter((w) => w.length > 0)
  const result: string[] = []

  for (const word of words) {
    if (COMMON_WORDS.has(word) || word.length <= 3) {
      result.push(word)
      continue
    }

    const remaining = word
    let found = false

    // Try prefix + rest
    for (const prefix of SUBWORD_PREFIXES) {
      if (remaining.startsWith(prefix) && remaining.length > prefix.length + 1) {
        const rest = remaining.slice(prefix.length)
        // Check if the rest or a suffix match makes sense
        for (const suffix of SUBWORD_SUFFIXES) {
          if (rest.endsWith(suffix) && rest.length > suffix.length) {
            const stem = rest.slice(0, rest.length - suffix.length)
            if (stem.length >= 2) {
              result.push(prefix + '##', stem + '##', suffix)
              found = true
              break
            }
          }
        }
        if (found) break
        // Just prefix + rest
        if (rest.length >= 2) {
          result.push(prefix + '##', rest)
          found = true
          break
        }
      }
    }

    if (!found) {
      // Try suffix split
      for (const suffix of SUBWORD_SUFFIXES) {
        if (remaining.endsWith(suffix) && remaining.length > suffix.length + 1) {
          const stem = remaining.slice(0, remaining.length - suffix.length)
          if (stem.length >= 2) {
            result.push(stem + '##', suffix)
            found = true
            break
          }
        }
      }
    }

    if (!found) {
      result.push(word)
    }
  }

  return result
}

function tokenizeCharacter(text: string): string[] {
  return text
    .toLowerCase()
    .split('')
    .filter((c) => c !== ' ')
}

function tokenize(text: string, strategy: string): string[] {
  switch (strategy) {
    case 'word':
      return tokenizeWord(text)
    case 'subword':
      return tokenizeSubword(text)
    case 'character':
      return tokenizeCharacter(text)
    default:
      return tokenizeWord(text)
  }
}

// Simple hash for vocab ID generation
function vocabId(token: string): number {
  let hash = 0
  for (let i = 0; i < token.length; i++) {
    hash = (hash * 31 + token.charCodeAt(i)) % 50000
  }
  return hash + 100 // offset to avoid low numbers
}

// Generate a pseudo-random embedding vector from a token
function generateEmbedding(token: string, dims: number): number[] {
  const id = vocabId(token)
  const vec: number[] = []
  for (let d = 0; d < dims; d++) {
    // Seeded pseudo-random from token + dimension
    const seed = id * 7 + d * 13 + 37
    const val = Math.sin(seed * 2654435761) * 0.5
    vec.push(val)
  }
  return vec
}

// ── Embedding Space Data (hardcoded 2D positions) ─────────────────────

interface EmbeddingPoint {
  word: string
  x: number
  y: number
  category: string
}

const EMBEDDING_SPACE_DATA: EmbeddingPoint[] = [
  // Royalty cluster (top-right area)
  { word: 'king', x: 0.72, y: 0.18, category: 'royalty' },
  { word: 'queen', x: 0.78, y: 0.22, category: 'royalty' },
  { word: 'prince', x: 0.70, y: 0.25, category: 'royalty' },
  { word: 'princess', x: 0.76, y: 0.28, category: 'royalty' },
  { word: 'throne', x: 0.68, y: 0.15, category: 'royalty' },
  { word: 'crown', x: 0.74, y: 0.12, category: 'royalty' },
  { word: 'royal', x: 0.65, y: 0.20, category: 'royalty' },
  { word: 'man', x: 0.58, y: 0.32, category: 'royalty' },
  { word: 'woman', x: 0.64, y: 0.36, category: 'royalty' },
  { word: 'boy', x: 0.55, y: 0.35, category: 'royalty' },
  { word: 'girl', x: 0.61, y: 0.38, category: 'royalty' },

  // Animals cluster (bottom-left area)
  { word: 'cat', x: 0.18, y: 0.72, category: 'animals' },
  { word: 'dog', x: 0.22, y: 0.68, category: 'animals' },
  { word: 'fish', x: 0.15, y: 0.78, category: 'animals' },
  { word: 'bird', x: 0.25, y: 0.75, category: 'animals' },
  { word: 'horse', x: 0.28, y: 0.65, category: 'animals' },
  { word: 'mouse', x: 0.12, y: 0.70, category: 'animals' },
  { word: 'lion', x: 0.20, y: 0.62, category: 'animals' },
  { word: 'tiger', x: 0.24, y: 0.60, category: 'animals' },

  // Actions cluster (center-left area)
  { word: 'run', x: 0.30, y: 0.42, category: 'actions' },
  { word: 'walk', x: 0.28, y: 0.48, category: 'actions' },
  { word: 'jump', x: 0.34, y: 0.40, category: 'actions' },
  { word: 'sit', x: 0.25, y: 0.45, category: 'actions' },
  { word: 'eat', x: 0.32, y: 0.50, category: 'actions' },
  { word: 'drink', x: 0.35, y: 0.52, category: 'actions' },
  { word: 'sleep', x: 0.22, y: 0.50, category: 'actions' },
  { word: 'swim', x: 0.38, y: 0.44, category: 'actions' },
  { word: 'fly', x: 0.36, y: 0.38, category: 'actions' },
  { word: 'climb', x: 0.40, y: 0.46, category: 'actions' },

  // Food cluster (bottom-right area)
  { word: 'apple', x: 0.72, y: 0.70, category: 'food' },
  { word: 'bread', x: 0.68, y: 0.72, category: 'food' },
  { word: 'cake', x: 0.75, y: 0.68, category: 'food' },
  { word: 'cheese', x: 0.70, y: 0.75, category: 'food' },
  { word: 'rice', x: 0.65, y: 0.74, category: 'food' },
  { word: 'soup', x: 0.78, y: 0.72, category: 'food' },
  { word: 'meat', x: 0.66, y: 0.68, category: 'food' },
  { word: 'fruit', x: 0.74, y: 0.65, category: 'food' },

  // Colors cluster (top-left area)
  { word: 'red', x: 0.15, y: 0.18, category: 'colors' },
  { word: 'blue', x: 0.12, y: 0.22, category: 'colors' },
  { word: 'green', x: 0.18, y: 0.15, category: 'colors' },
  { word: 'yellow', x: 0.20, y: 0.20, category: 'colors' },
  { word: 'black', x: 0.10, y: 0.25, category: 'colors' },
  { word: 'white', x: 0.16, y: 0.28, category: 'colors' },
  { word: 'purple', x: 0.22, y: 0.25, category: 'colors' },

  // Numbers cluster (center-right area)
  { word: 'one', x: 0.80, y: 0.42, category: 'numbers' },
  { word: 'two', x: 0.82, y: 0.45, category: 'numbers' },
  { word: 'three', x: 0.84, y: 0.48, category: 'numbers' },
  { word: 'four', x: 0.78, y: 0.48, category: 'numbers' },
  { word: 'five', x: 0.86, y: 0.44, category: 'numbers' },
  { word: 'ten', x: 0.82, y: 0.40, category: 'numbers' },
  { word: 'hundred', x: 0.88, y: 0.50, category: 'numbers' },

  // Body cluster (center area)
  { word: 'head', x: 0.48, y: 0.55, category: 'body' },
  { word: 'hand', x: 0.45, y: 0.58, category: 'body' },
  { word: 'eye', x: 0.50, y: 0.52, category: 'body' },
  { word: 'heart', x: 0.52, y: 0.56, category: 'body' },
  { word: 'face', x: 0.46, y: 0.52, category: 'body' },
  { word: 'foot', x: 0.44, y: 0.60, category: 'body' },

  // Nature cluster (top-center area)
  { word: 'sun', x: 0.42, y: 0.15, category: 'nature' },
  { word: 'moon', x: 0.45, y: 0.18, category: 'nature' },
  { word: 'star', x: 0.48, y: 0.12, category: 'nature' },
  { word: 'tree', x: 0.38, y: 0.20, category: 'nature' },
  { word: 'river', x: 0.50, y: 0.22, category: 'nature' },
  { word: 'mountain', x: 0.44, y: 0.25, category: 'nature' },
  { word: 'ocean', x: 0.52, y: 0.18, category: 'nature' },
  { word: 'sky', x: 0.40, y: 0.12, category: 'nature' },
]

// Build a lookup for quick matching
const EMBEDDING_LOOKUP = new Map(EMBEDDING_SPACE_DATA.map((p) => [p.word, p]))

// ── Stage 1: Tokenization ─────────────────────────────────────────────

function TokenizationStage({
  tokens,
  showIds,
  innerWidth,
  innerHeight,
}: {
  tokens: string[]
  showIds: boolean
  innerWidth: number
  innerHeight: number
}) {
  const maxTokensPerRow = Math.max(Math.floor(innerWidth / 65), 4)
  const rows = Math.ceil(tokens.length / maxTokensPerRow)
  const pillW = 54
  const pillH = 24
  const rowH = showIds ? 46 : 36
  const startY = 8

  return (
    <g>
      {/* Stage label */}
      <text
        x={innerWidth / 2}
        y={0}
        textAnchor="middle"
        className="text-[10px] font-medium uppercase tracking-wider"
        fill="#A1A1AA"
      >
        Stage 1: Tokenization
      </text>

      <AnimatePresence mode="popLayout">
        {tokens.map((token, i) => {
          const row = Math.floor(i / maxTokensPerRow)
          const col = i % maxTokensPerRow
          const tokensInRow = Math.min(
            maxTokensPerRow,
            tokens.length - row * maxTokensPerRow
          )
          const rowStartX =
            (innerWidth - tokensInRow * (pillW + 6)) / 2
          const x = rowStartX + col * (pillW + 6) + pillW / 2
          const y = startY + 16 + row * rowH

          // Cycle through a set of subtle background colors
          const tokenColors = [
            'rgba(244, 114, 182, 0.15)',
            'rgba(52, 211, 153, 0.15)',
            'rgba(251, 191, 36, 0.15)',
            'rgba(56, 189, 248, 0.15)',
            'rgba(129, 140, 248, 0.15)',
            'rgba(167, 139, 250, 0.15)',
          ]
          const borderColors = [
            'rgba(244, 114, 182, 0.3)',
            'rgba(52, 211, 153, 0.3)',
            'rgba(251, 191, 36, 0.3)',
            'rgba(56, 189, 248, 0.3)',
            'rgba(129, 140, 248, 0.3)',
            'rgba(167, 139, 250, 0.3)',
          ]

          return (
            <motion.g
              key={`${token}-${i}`}
              initial={{ opacity: 0, x: x - 20 }}
              animate={{ opacity: 1, x: x }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ duration: 0.3, delay: i * 0.04 }}
            >
              {/* Token pill */}
              <rect
                x={x - pillW / 2}
                y={y - pillH / 2}
                width={pillW}
                height={pillH}
                rx={pillH / 2}
                fill={tokenColors[i % tokenColors.length]}
                stroke={borderColors[i % borderColors.length]}
                strokeWidth={1}
              />
              <text
                x={x}
                y={y + 1}
                textAnchor="middle"
                dominantBaseline="central"
                className="text-[9px] font-mono"
                fill="#E4E4E7"
              >
                {token.length > 7 ? token.slice(0, 6) + '..' : token}
              </text>

              {/* Vocab ID */}
              {showIds && (
                <motion.text
                  x={x}
                  y={y + pillH / 2 + 10}
                  textAnchor="middle"
                  className="text-[7px] font-mono"
                  fill="#71717A"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  {vocabId(token)}
                </motion.text>
              )}
            </motion.g>
          )
        })}
      </AnimatePresence>

      {/* Token count */}
      <text
        x={innerWidth}
        y={startY + rows * rowH + 12}
        textAnchor="end"
        className="text-[8px] font-mono"
        fill="#52525B"
      >
        {tokens.length} tokens
      </text>

      {/* Available height indicator - faint line to separate from next stage */}
      <line
        x1={innerWidth * 0.2}
        x2={innerWidth * 0.8}
        y1={innerHeight - 2}
        y2={innerHeight - 2}
        stroke="rgba(255,255,255,0.05)"
        strokeWidth={1}
      />
    </g>
  )
}

// ── Stage 2: Embedding Lookup ─────────────────────────────────────────

function EmbeddingLookupStage({
  tokens,
  embeddingDims,
  innerWidth,
  innerHeight,
}: {
  tokens: string[]
  embeddingDims: number
  innerWidth: number
  innerHeight: number
}) {
  // Show a simplified embedding matrix + per-token vectors
  const maxTokensToShow = Math.min(tokens.length, 8)
  const visibleTokens = tokens.slice(0, maxTokensToShow)

  // Matrix dimensions
  const matrixRows = Math.min(12, maxTokensToShow + 4) // a few extra rows for "other" vocab
  const matrixCols = embeddingDims
  const cellSize = Math.min(
    (innerWidth * 0.35) / matrixCols,
    (innerHeight - 40) / matrixRows,
    14
  )
  const matrixW = matrixCols * cellSize
  const matrixH = matrixRows * cellSize
  const matrixX = 20
  const matrixY = 30

  // Bar charts for each token (to the right of the matrix)
  const barAreaX = matrixX + matrixW + 40
  const barMaxW = innerWidth - barAreaX - 10
  const barH = Math.min(
    (innerHeight - 40) / maxTokensToShow - 4,
    16
  )
  const barStartY = matrixY

  // Color scale for matrix cells
  const cellColorScale = d3
    .scaleLinear<string>()
    .domain([-0.5, 0, 0.5])
    .range(['#6366F1', 'rgba(255,255,255,0.03)', '#F472B6'])
    .clamp(true)

  return (
    <g>
      {/* Stage label */}
      <text
        x={innerWidth / 2}
        y={12}
        textAnchor="middle"
        className="text-[10px] font-medium uppercase tracking-wider"
        fill="#A1A1AA"
      >
        Stage 2: Embedding Lookup
      </text>

      {/* Embedding matrix */}
      <text
        x={matrixX + matrixW / 2}
        y={matrixY - 6}
        textAnchor="middle"
        className="text-[8px] font-mono"
        fill="#71717A"
      >
        Embedding Matrix (V x d)
      </text>

      {Array.from({ length: matrixRows }).map((_, row) => (
        <g key={`matrix-row-${row}`}>
          {Array.from({ length: matrixCols }).map((_, col) => {
            // Determine if this row corresponds to one of our tokens
            const tokenIdx = visibleTokens.findIndex(
              (_, ti) => ti === row
            )
            const isHighlighted = tokenIdx >= 0 && tokenIdx < maxTokensToShow

            // Generate a value for this cell
            const val =
              tokenIdx >= 0
                ? generateEmbedding(visibleTokens[tokenIdx], embeddingDims)[col]
                : Math.sin((row * 7 + col * 13) * 0.5) * 0.3

            return (
              <motion.rect
                key={`cell-${row}-${col}`}
                x={matrixX + col * cellSize}
                y={matrixY + row * cellSize}
                width={cellSize - 1}
                height={cellSize - 1}
                rx={1}
                fill={cellColorScale(val)}
                fillOpacity={isHighlighted ? 0.8 : 0.2}
                stroke={
                  isHighlighted
                    ? 'rgba(129, 140, 248, 0.4)'
                    : 'transparent'
                }
                strokeWidth={isHighlighted ? 1 : 0}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3, delay: row * 0.03 }}
              />
            )
          })}

          {/* Row label (token name for highlighted rows) */}
          {row < visibleTokens.length && (
            <text
              x={matrixX - 4}
              y={matrixY + row * cellSize + cellSize / 2 + 1}
              textAnchor="end"
              dominantBaseline="central"
              className="text-[7px] font-mono"
              fill={NLP.tokenHighlight}
            >
              {visibleTokens[row].length > 5
                ? visibleTokens[row].slice(0, 4) + '..'
                : visibleTokens[row]}
            </text>
          )}
        </g>
      ))}

      {/* Dimension labels along top */}
      {Array.from({ length: matrixCols }).map((_, col) => (
        <text
          key={`dim-label-${col}`}
          x={matrixX + col * cellSize + cellSize / 2}
          y={matrixY + matrixH + 10}
          textAnchor="middle"
          className="text-[6px] font-mono"
          fill="#52525B"
        >
          d{col}
        </text>
      ))}

      {/* Arrow from matrix to bar charts */}
      <line
        x1={matrixX + matrixW + 6}
        y1={matrixY + matrixH / 2}
        x2={barAreaX - 12}
        y2={matrixY + matrixH / 2}
        stroke="rgba(129, 140, 248, 0.3)"
        strokeWidth={1}
        strokeDasharray="4,3"
        markerEnd="url(#embed-arrow)"
      />
      <defs>
        <marker
          id="embed-arrow"
          markerWidth="5"
          markerHeight="4"
          refX="5"
          refY="2"
          orient="auto"
        >
          <path d="M0,0 L5,2 L0,4" fill="rgba(129,140,248,0.4)" />
        </marker>
      </defs>

      {/* Per-token embedding bar charts */}
      <text
        x={barAreaX + barMaxW / 2}
        y={barStartY - 6}
        textAnchor="middle"
        className="text-[8px] font-mono"
        fill="#71717A"
      >
        Embedding Vectors
      </text>

      {visibleTokens.map((token, ti) => {
        const embedding = generateEmbedding(token, embeddingDims)
        const y = barStartY + ti * (barH + 4)

        return (
          <g key={`embed-bar-${ti}`}>
            {/* Token label */}
            <text
              x={barAreaX - 4}
              y={y + barH / 2 + 1}
              textAnchor="end"
              dominantBaseline="central"
              className="text-[7px] font-mono"
              fill={NLP.tokenHighlight}
            >
              {token.length > 5 ? token.slice(0, 4) + '..' : token}
            </text>

            {/* Background bar */}
            <rect
              x={barAreaX}
              y={y}
              width={barMaxW}
              height={barH}
              rx={2}
              fill="rgba(255,255,255,0.02)"
            />

            {/* Individual dimension bars */}
            {embedding.map((val, d) => {
              const segW = barMaxW / embeddingDims
              return (
                <motion.rect
                  key={`bar-${ti}-${d}`}
                  x={barAreaX + d * segW}
                  y={y}
                  width={segW - 1}
                  height={barH}
                  rx={1}
                  fill={cellColorScale(val)}
                  fillOpacity={0.7}
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{
                    duration: 0.3,
                    delay: ti * 0.05 + d * 0.02,
                  }}
                />
              )
            })}
          </g>
        )
      })}

      {tokens.length > maxTokensToShow && (
        <text
          x={barAreaX + barMaxW / 2}
          y={barStartY + maxTokensToShow * (barH + 4) + 4}
          textAnchor="middle"
          className="text-[7px]"
          fill="#52525B"
        >
          +{tokens.length - maxTokensToShow} more tokens...
        </text>
      )}
    </g>
  )
}

// ── Stage 3: Embedding Space ──────────────────────────────────────────

function EmbeddingSpaceStage({
  tokens,
  innerWidth,
  innerHeight,
  showArithmetic,
}: {
  tokens: string[]
  innerWidth: number
  innerHeight: number
  showArithmetic: boolean
}) {
  const padding = 30
  const plotW = innerWidth - padding * 2
  const plotH = innerHeight - 40

  const xScale = d3
    .scaleLinear()
    .domain([0, 1])
    .range([padding, padding + plotW])

  const yScale = d3
    .scaleLinear()
    .domain([0, 1])
    .range([30, 30 + plotH])

  // Find which input tokens match embedding space words
  const matchedTokens = new Set(
    tokens
      .map((t) => t.replace('##', '').toLowerCase())
      .filter((t) => EMBEDDING_LOOKUP.has(t))
  )

  // Arithmetic: king - man + woman = queen
  const kingPt = EMBEDDING_LOOKUP.get('king')
  const manPt = EMBEDDING_LOOKUP.get('man')
  const womanPt = EMBEDDING_LOOKUP.get('woman')
  const queenPt = EMBEDDING_LOOKUP.get('queen')

  // Get unique categories for legend
  const categories = Array.from(
    new Set(EMBEDDING_SPACE_DATA.map((p) => p.category))
  )

  return (
    <g>
      {/* Stage label */}
      <text
        x={innerWidth / 2}
        y={14}
        textAnchor="middle"
        className="text-[10px] font-medium uppercase tracking-wider"
        fill="#A1A1AA"
      >
        Stage 3: Embedding Space (2D projection)
      </text>

      {/* Faint grid */}
      {[0.2, 0.4, 0.6, 0.8].map((v) => (
        <g key={`grid-${v}`}>
          <line
            x1={xScale(v)}
            y1={yScale(0)}
            x2={xScale(v)}
            y2={yScale(1)}
            stroke="rgba(255,255,255,0.03)"
            strokeWidth={0.5}
          />
          <line
            x1={xScale(0)}
            y1={yScale(v)}
            x2={xScale(1)}
            y2={yScale(v)}
            stroke="rgba(255,255,255,0.03)"
            strokeWidth={0.5}
          />
        </g>
      ))}

      {/* All embedding points */}
      {EMBEDDING_SPACE_DATA.map((point) => {
        const isMatched = matchedTokens.has(point.word)
        const color = CATEGORY_COLORS[point.category] || '#71717A'
        const cx = xScale(point.x)
        const cy = yScale(point.y)

        return (
          <g key={`point-${point.word}`}>
            <motion.circle
              cx={cx}
              cy={cy}
              r={isMatched ? 6 : 3}
              fill={color}
              fillOpacity={isMatched ? 0.9 : 0.4}
              stroke={isMatched ? '#fff' : 'transparent'}
              strokeWidth={isMatched ? 1.5 : 0}
              initial={isMatched ? { scale: 0 } : undefined}
              animate={
                isMatched
                  ? { scale: [1, 1.3, 1] }
                  : undefined
              }
              transition={
                isMatched
                  ? { duration: 1, repeat: Infinity, repeatDelay: 2 }
                  : undefined
              }
            />
            {/* Label */}
            <text
              x={cx}
              y={cy - (isMatched ? 9 : 5)}
              textAnchor="middle"
              className={`font-mono ${isMatched ? 'text-[8px]' : 'text-[6px]'}`}
              fill={isMatched ? '#E4E4E7' : '#71717A'}
              fillOpacity={isMatched ? 1 : 0.6}
            >
              {point.word}
            </text>
          </g>
        )
      })}

      {/* King - man + woman = queen arithmetic */}
      {showArithmetic && kingPt && manPt && womanPt && queenPt && (
        <g>
          {/* king -> man (subtract) */}
          <motion.line
            x1={xScale(kingPt.x)}
            y1={yScale(kingPt.y)}
            x2={xScale(manPt.x)}
            y2={yScale(manPt.y)}
            stroke="#F87171"
            strokeWidth={2}
            strokeDasharray="6,3"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.6 }}
          />
          <motion.text
            x={(xScale(kingPt.x) + xScale(manPt.x)) / 2 - 8}
            y={(yScale(kingPt.y) + yScale(manPt.y)) / 2 - 6}
            className="text-[9px] font-bold"
            fill="#F87171"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            - man
          </motion.text>

          {/* + woman */}
          <motion.line
            x1={xScale(manPt.x)}
            y1={yScale(manPt.y)}
            x2={xScale(womanPt.x)}
            y2={yScale(womanPt.y)}
            stroke="#4ADE80"
            strokeWidth={2}
            strokeDasharray="6,3"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          />
          <motion.text
            x={(xScale(manPt.x) + xScale(womanPt.x)) / 2 + 8}
            y={(yScale(manPt.y) + yScale(womanPt.y)) / 2 + 12}
            className="text-[9px] font-bold"
            fill="#4ADE80"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.9 }}
          >
            + woman
          </motion.text>

          {/* -> queen (result) */}
          <motion.line
            x1={xScale(womanPt.x)}
            y1={yScale(womanPt.y)}
            x2={xScale(queenPt.x)}
            y2={yScale(queenPt.y)}
            stroke={NLP.value}
            strokeWidth={2}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.6, delay: 1.2 }}
          />
          <motion.text
            x={(xScale(womanPt.x) + xScale(queenPt.x)) / 2 + 10}
            y={(yScale(womanPt.y) + yScale(queenPt.y)) / 2}
            className="text-[10px] font-bold"
            fill={NLP.value}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
          >
            = queen
          </motion.text>

          {/* Result highlight ring on queen */}
          <motion.circle
            cx={xScale(queenPt.x)}
            cy={yScale(queenPt.y)}
            r={10}
            fill="transparent"
            stroke={NLP.value}
            strokeWidth={2}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 1.5, duration: 0.4 }}
          />
        </g>
      )}

      {/* Legend */}
      {categories.map((cat, ci) => {
        const legendX = padding + 4
        const legendY = 30 + plotH + 14
        const colWidth = Math.min(innerWidth / 4, 85)
        const col = ci % 4
        const row = Math.floor(ci / 4)

        return (
          <g key={`legend-${cat}`}>
            <circle
              cx={legendX + col * colWidth}
              cy={legendY + row * 14}
              r={3}
              fill={CATEGORY_COLORS[cat]}
              fillOpacity={0.7}
            />
            <text
              x={legendX + col * colWidth + 7}
              y={legendY + row * 14 + 1}
              dominantBaseline="central"
              className="text-[7px]"
              fill="#71717A"
            >
              {cat}
            </text>
          </g>
        )
      })}
    </g>
  )
}

// ── Main Component ────────────────────────────────────────────────────

const STRATEGY_OPTIONS = [
  { value: 'word', label: 'Word-level' },
  { value: 'subword', label: 'Subword (BPE)' },
  { value: 'character', label: 'Character' },
]

const DIMS_OPTIONS = [
  { value: '4', label: '4 dims' },
  { value: '8', label: '8 dims' },
  { value: '16', label: '16 dims' },
]

export function TokenizationPipeline() {
  const [inputText, setInputText] = useState('The cat sat on the mat')
  const [strategy, setStrategy] = useState('word')
  const [showIds, setShowIds] = useState(false)
  const [embeddingDims, setEmbeddingDims] = useState('8')
  const [showArithmetic, setShowArithmetic] = useState(false)

  const tokens = useMemo(
    () => tokenize(inputText, strategy),
    [inputText, strategy]
  )

  const handleShowArithmetic = useCallback(() => {
    setShowArithmetic((prev) => !prev)
  }, [])

  // Compute token row count for dynamic SVG height
  const tokenRowCount = useMemo(() => {
    const maxPerRow = 8
    return Math.ceil(tokens.length / maxPerRow)
  }, [tokens])

  return (
    <div className="space-y-4">
      {/* Stage 1: Tokenization */}
      <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border overflow-hidden">
        <SVGContainer
          aspectRatio={16 / (4 + tokenRowCount * 1.5)}
          minHeight={120}
          maxHeight={250}
          padding={{ top: 10, right: 20, bottom: 10, left: 20 }}
        >
          {({ innerWidth, innerHeight }) => (
            <TokenizationStage
              tokens={tokens}
              showIds={showIds}
              innerWidth={innerWidth}
              innerHeight={innerHeight}
            />
          )}
        </SVGContainer>
      </div>

      {/* Stage 2: Embedding Lookup */}
      <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border overflow-hidden">
        <SVGContainer
          aspectRatio={16 / 8}
          minHeight={200}
          maxHeight={400}
          padding={{ top: 10, right: 20, bottom: 10, left: 20 }}
        >
          {({ innerWidth, innerHeight }) => (
            <EmbeddingLookupStage
              tokens={tokens}
              embeddingDims={parseInt(embeddingDims)}
              innerWidth={innerWidth}
              innerHeight={innerHeight}
            />
          )}
        </SVGContainer>
      </div>

      {/* Stage 3: Embedding Space */}
      <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border overflow-hidden">
        <SVGContainer
          aspectRatio={16 / 12}
          minHeight={300}
          maxHeight={500}
          padding={{ top: 10, right: 20, bottom: 30, left: 20 }}
        >
          {({ innerWidth, innerHeight }) => (
            <EmbeddingSpaceStage
              tokens={tokens}
              innerWidth={innerWidth}
              innerHeight={innerHeight}
              showArithmetic={showArithmetic}
            />
          )}
        </SVGContainer>
      </div>

      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          {/* Text input */}
          <div className="flex flex-col gap-1.5 flex-1 min-w-[200px]">
            <label className="text-xs font-medium text-text-secondary uppercase tracking-wider">
              Input Text
            </label>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="bg-obsidian-glass border border-obsidian-border rounded-lg px-3 py-2 text-sm text-text-primary
                focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent/50 font-mono"
              placeholder="Type a sentence..."
            />
          </div>

          {/* Tokenization strategy */}
          <Select
            label="Tokenization"
            value={strategy}
            options={STRATEGY_OPTIONS}
            onChange={setStrategy}
            className="w-36"
          />

          {/* Show vocab IDs */}
          <Toggle
            label="Vocab IDs"
            checked={showIds}
            onChange={setShowIds}
          />

          {/* Embedding dims */}
          <Select
            label="Embed Dims"
            value={embeddingDims}
            options={DIMS_OPTIONS}
            onChange={setEmbeddingDims}
            className="w-28"
          />

          {/* Arithmetic button */}
          <Button
            variant={showArithmetic ? 'primary' : 'secondary'}
            size="sm"
            active={showArithmetic}
            onClick={handleShowArithmetic}
          >
            {showArithmetic
              ? 'Hide Arithmetic'
              : 'king - man + woman = queen'}
          </Button>
        </div>
      </GlassCard>
    </div>
  )
}
