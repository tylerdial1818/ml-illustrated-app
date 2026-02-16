/**
 * Simple in-browser tokenizer for demo/educational purposes.
 * Supports word-level, simplified subword (BPE-like), and character-level tokenization.
 */

export interface TokenizerResult {
  tokens: string[]
  tokenIds: number[]
}

export type TokenizationStrategy = 'word' | 'subword' | 'character'

// Special tokens
const PAD_TOKEN = '[PAD]'
const UNK_TOKEN = '[UNK]'
const CLS_TOKEN = '[CLS]'
const SEP_TOKEN = '[SEP]'
const MASK_TOKEN = '[MASK]'

/**
 * Common subword pieces for simplified BPE-like tokenization.
 * These cover common English prefixes, suffixes, and morphemes.
 */
const SUBWORD_PIECES: string[] = [
  'un', 're', 'ing', 'tion', 'able', 'ment', 'ly', 'ed', 'er', 'est',
  'ness', 'ful', 'less', 'pre', 'dis', 'mis', 'over', 'under', 'out',
  'al', 'ous', 'ive', 'ize', 'ify', 'ish', 'en', 'ary', 'ory', 'ist',
  'ism',
]

/**
 * A pre-defined vocabulary of common words mapped to IDs.
 * Includes special tokens, all demo sentence words, and ~200+ common English words.
 */
export const VOCABULARY: Map<string, number> = new Map([
  // Special tokens (0-4)
  [PAD_TOKEN, 0],
  [UNK_TOKEN, 1],
  [CLS_TOKEN, 2],
  [SEP_TOKEN, 3],
  [MASK_TOKEN, 4],

  // Punctuation (5-14)
  ['.', 5],
  [',', 6],
  ['!', 7],
  ['?', 8],
  ["'", 9],
  ['"', 10],
  ['-', 11],
  [':', 12],
  [';', 13],
  ['(', 14],

  // Articles & determiners (15-19)
  ['the', 15],
  ['a', 16],
  ['an', 17],
  ['this', 18],
  ['that', 19],

  // Demo sentence words - "The cat sat on the mat" (20-24)
  ['cat', 20],
  ['sat', 21],
  ['on', 22],
  ['mat', 23],

  // Demo sentence words - "The animal didn't cross the street because it was too tired" (24-33)
  ['animal', 24],
  ["didn't", 25],
  ['cross', 26],
  ['street', 27],
  ['because', 28],
  ['it', 29],
  ['was', 30],
  ['too', 31],
  ['tired', 32],

  // Demo sentence words - bank sentences (33-39)
  ['i', 33],
  ['went', 34],
  ['to', 35],
  ['bank', 36],
  ['deposit', 37],
  ['money', 38],
  ['skip', 39],
  ['rocks', 40],

  // Demo sentence words - "She gave him the book that she had been reading" (41-48)
  ['she', 41],
  ['gave', 42],
  ['him', 43],
  ['book', 44],
  ['had', 45],
  ['been', 46],
  ['reading', 47],

  // Pronouns (48-63)
  ['he', 48],
  ['we', 49],
  ['they', 50],
  ['me', 51],
  ['you', 52],
  ['her', 53],
  ['his', 54],
  ['its', 55],
  ['them', 56],
  ['my', 57],
  ['your', 58],
  ['our', 59],
  ['their', 60],
  ['us', 61],
  ['who', 62],
  ['what', 63],

  // Prepositions (64-79)
  ['in', 64],
  ['at', 65],
  ['by', 66],
  ['for', 67],
  ['with', 68],
  ['from', 69],
  ['of', 70],
  ['about', 71],
  ['into', 72],
  ['through', 73],
  ['after', 74],
  ['before', 75],
  ['between', 76],
  ['under', 77],
  ['over', 78],
  ['up', 79],

  // Common verbs (80-119)
  ['is', 80],
  ['are', 81],
  ['am', 82],
  ['be', 83],
  ['have', 84],
  ['has', 85],
  ['do', 86],
  ['does', 87],
  ['did', 88],
  ['will', 89],
  ['would', 90],
  ['can', 91],
  ['could', 92],
  ['should', 93],
  ['may', 94],
  ['might', 95],
  ['must', 96],
  ['shall', 97],
  ['go', 98],
  ['get', 99],
  ['make', 100],
  ['know', 101],
  ['take', 102],
  ['come', 103],
  ['see', 104],
  ['think', 105],
  ['say', 106],
  ['give', 107],
  ['find', 108],
  ['tell', 109],
  ['want', 110],
  ['use', 111],
  ['put', 112],
  ['run', 113],
  ['walk', 114],
  ['eat', 115],
  ['write', 116],
  ['read', 117],
  ['play', 118],
  ['like', 119],

  // Common nouns (120-169)
  ['time', 120],
  ['year', 121],
  ['people', 122],
  ['way', 123],
  ['day', 124],
  ['man', 125],
  ['woman', 126],
  ['child', 127],
  ['world', 128],
  ['life', 129],
  ['hand', 130],
  ['part', 131],
  ['place', 132],
  ['case', 133],
  ['week', 134],
  ['head', 135],
  ['side', 136],
  ['water', 137],
  ['house', 138],
  ['word', 139],
  ['food', 140],
  ['dog', 141],
  ['bird', 142],
  ['fish', 143],
  ['tree', 144],
  ['city', 145],
  ['car', 146],
  ['door', 147],
  ['room', 148],
  ['name', 149],
  ['school', 150],
  ['number', 151],
  ['night', 152],
  ['home', 153],
  ['point', 154],
  ['thing', 155],
  ['story', 156],
  ['fact', 157],
  ['morning', 158],
  ['end', 159],
  ['king', 160],
  ['queen', 161],
  ['boy', 162],
  ['girl', 163],
  ['family', 164],
  ['friend', 165],
  ['eye', 166],
  ['face', 167],
  ['country', 168],
  ['earth', 169],
  ['sun', 170],

  // Common adjectives (171-199)
  ['good', 171],
  ['new', 172],
  ['first', 173],
  ['last', 174],
  ['long', 175],
  ['great', 176],
  ['little', 177],
  ['big', 178],
  ['old', 179],
  ['high', 180],
  ['small', 181],
  ['large', 182],
  ['next', 183],
  ['young', 184],
  ['right', 185],
  ['important', 186],
  ['few', 187],
  ['bad', 188],
  ['same', 189],
  ['different', 190],
  ['own', 191],
  ['other', 192],
  ['early', 193],
  ['red', 194],
  ['blue', 195],
  ['green', 196],
  ['white', 197],
  ['black', 198],
  ['happy', 199],

  // Adverbs & conjunctions (200-219)
  ['not', 200],
  ['also', 201],
  ['very', 202],
  ['just', 203],
  ['then', 204],
  ['more', 205],
  ['so', 206],
  ['now', 207],
  ['only', 208],
  ['still', 209],
  ['here', 210],
  ['there', 211],
  ['where', 212],
  ['when', 213],
  ['how', 214],
  ['all', 215],
  ['each', 216],
  ['every', 217],
  ['and', 218],
  ['but', 219],

  // More conjunctions & connectors (220-229)
  ['or', 220],
  ['if', 221],
  ['while', 222],
  ['as', 223],
  ['than', 224],
  ['which', 225],
  ['no', 226],
  ['yes', 227],
  ['some', 228],
  ['any', 229],

  // Numbers as words (230-239)
  ['one', 230],
  ['two', 231],
  ['three', 232],
  ['four', 233],
  ['five', 234],
  ['six', 235],
  ['seven', 236],
  ['eight', 237],
  ['nine', 238],
  ['ten', 239],

  // Subword pieces (240-269)
  ['##un', 240],
  ['##re', 241],
  ['##ing', 242],
  ['##tion', 243],
  ['##able', 244],
  ['##ment', 245],
  ['##ly', 246],
  ['##ed', 247],
  ['##er', 248],
  ['##est', 249],
  ['##ness', 250],
  ['##ful', 251],
  ['##less', 252],
  ['##pre', 253],
  ['##dis', 254],
  ['##mis', 255],
  ['##over', 256],
  ['##under', 257],
  ['##out', 258],
  ['##al', 259],
  ['##ous', 260],
  ['##ive', 261],
  ['##ize', 262],
  ['##ify', 263],
  ['##ish', 264],
  ['##en', 265],
  ['##ary', 266],
  ['##ory', 267],
  ['##ist', 268],
  ['##ism', 269],
])

/**
 * Reverse vocabulary: ID -> token string
 */
const ID_TO_TOKEN: Map<number, string> = new Map(
  Array.from(VOCABULARY.entries()).map(([token, id]) => [id, token])
)

/**
 * Tokenize text using the specified strategy.
 *
 * @param text - Input text to tokenize
 * @param strategy - 'word', 'subword', or 'character'
 * @returns Object with token strings and their IDs
 */
export function tokenize(
  text: string,
  strategy: TokenizationStrategy
): TokenizerResult {
  switch (strategy) {
    case 'word':
      return tokenizeWord(text)
    case 'subword':
      return tokenizeSubword(text)
    case 'character':
      return tokenizeCharacter(text)
  }
}

/**
 * Convert token IDs back to token strings.
 */
export function detokenize(tokenIds: number[]): string[] {
  return tokenIds.map((id) => ID_TO_TOKEN.get(id) ?? UNK_TOKEN)
}

/**
 * Word-level tokenization.
 * Splits on whitespace and punctuation, lowercases, looks up in vocabulary.
 */
function tokenizeWord(text: string): TokenizerResult {
  const rawTokens = splitIntoWords(text)
  const tokens: string[] = []
  const tokenIds: number[] = []

  for (const raw of rawTokens) {
    const lower = raw.toLowerCase()
    const id = VOCABULARY.get(lower)

    if (id !== undefined) {
      tokens.push(raw)
      tokenIds.push(id)
    } else {
      tokens.push(raw)
      tokenIds.push(VOCABULARY.get(UNK_TOKEN)!)
    }
  }

  return { tokens, tokenIds }
}

/**
 * Simplified BPE-like subword tokenization.
 * Known words are kept whole. Unknown words are decomposed into
 * subword pieces from the SUBWORD_PIECES list.
 */
function tokenizeSubword(text: string): TokenizerResult {
  const rawTokens = splitIntoWords(text)
  const tokens: string[] = []
  const tokenIds: number[] = []

  for (const raw of rawTokens) {
    const lower = raw.toLowerCase()
    const wholeWordId = VOCABULARY.get(lower)

    if (wholeWordId !== undefined) {
      // Known word - keep whole
      tokens.push(raw)
      tokenIds.push(wholeWordId)
    } else {
      // Unknown word - try to decompose into subword pieces
      const pieces = decomposeIntoSubwords(lower)
      for (const piece of pieces) {
        tokens.push(piece)
        const pieceId = VOCABULARY.get(piece) ?? VOCABULARY.get(UNK_TOKEN)!
        tokenIds.push(pieceId)
      }
    }
  }

  return { tokens, tokenIds }
}

/**
 * Character-level tokenization.
 * Each character becomes its own token.
 */
function tokenizeCharacter(text: string): TokenizerResult {
  const tokens: string[] = []
  const tokenIds: number[] = []

  for (const char of text) {
    tokens.push(char)
    const lower = char.toLowerCase()
    const id = VOCABULARY.get(lower)
    if (id !== undefined) {
      tokenIds.push(id)
    } else {
      // Use character code as a deterministic fallback ID
      // Offset by 1000 to avoid collisions with vocabulary IDs
      tokenIds.push(1000 + char.charCodeAt(0))
    }
  }

  return { tokens, tokenIds }
}

/**
 * Split text into word tokens, preserving punctuation as separate tokens.
 */
function splitIntoWords(text: string): string[] {
  const tokens: string[] = []
  // Match words (including contractions like didn't) and individual punctuation
  const regex = /\w+(?:'\w+)?|[^\s\w]/g
  let match: RegExpExecArray | null

  while ((match = regex.exec(text)) !== null) {
    tokens.push(match[0])
  }

  return tokens
}

/**
 * Decompose an unknown word into subword pieces.
 * Greedily matches the longest suffix/prefix from SUBWORD_PIECES.
 */
function decomposeIntoSubwords(word: string): string[] {
  const pieces: string[] = []
  let remaining = word

  // First, try to find a prefix
  let foundPrefix = false
  for (const piece of SUBWORD_PIECES) {
    if (remaining.startsWith(piece) && remaining.length > piece.length) {
      pieces.push(piece)
      remaining = remaining.slice(piece.length)
      foundPrefix = true
      break
    }
  }

  // Now greedily match suffixes from the remaining string
  while (remaining.length > 0) {
    let matched = false

    // Try to match the longest suffix piece
    let bestPiece = ''
    for (const piece of SUBWORD_PIECES) {
      if (
        remaining.endsWith(piece) &&
        piece.length > bestPiece.length &&
        remaining.length > piece.length
      ) {
        bestPiece = piece
      }
    }

    if (bestPiece) {
      // We have a suffix match - first handle what's before it
      const before = remaining.slice(0, remaining.length - bestPiece.length)
      if (before.length > 0) {
        // Try to match the before part with another piece
        let beforeMatched = false
        for (const piece of SUBWORD_PIECES) {
          if (before === piece) {
            pieces.push(foundPrefix || pieces.length > 0 ? '##' + piece : piece)
            beforeMatched = true
            break
          }
        }
        if (!beforeMatched) {
          // Keep the before part as a raw chunk
          pieces.push(
            foundPrefix || pieces.length > 0 ? '##' + before : before
          )
        }
      }
      pieces.push('##' + bestPiece)
      matched = true
      remaining = ''
    }

    if (!matched) {
      // No suffix match found - emit remaining as a single token
      pieces.push(
        foundPrefix || pieces.length > 0 ? '##' + remaining : remaining
      )
      remaining = ''
    }
  }

  // If we produced nothing useful, return the whole word as unknown
  if (pieces.length === 0) {
    return [word]
  }

  return pieces
}
