import { SectionNav } from '../../components/layout/SectionNav'
import { TransformersOverview } from './sections/TransformersOverview'
import { TokenizationSection } from './sections/TokenizationSection'
import { PositionalEncodingSection } from './sections/PositionalEncodingSection'
import { SelfAttentionSection } from './sections/SelfAttentionSection'
import { MultiHeadAttentionSection } from './sections/MultiHeadAttentionSection'
import { TransformerBlockSection } from './sections/TransformerBlockSection'
import { EncoderDecoderSection } from './sections/EncoderDecoderSection'
import { PuttingItTogetherSection } from './sections/PuttingItTogetherSection'

const SECTIONS = [
  { id: 'transformers-overview', label: 'Overview' },
  { id: 'tokenization', label: 'Tokenization' },
  { id: 'positional-encoding', label: 'Positional Encoding' },
  { id: 'self-attention', label: 'Self-Attention' },
  { id: 'multi-head', label: 'Multi-Head' },
  { id: 'transformer-block', label: 'Transformer Block' },
  { id: 'encoder-decoder', label: 'Encoder vs Decoder' },
  { id: 'putting-together', label: 'Full Picture' },
]

export function TransformersPage() {
  return (
    <div className="relative">
      <SectionNav sections={SECTIONS} />

      <div>
        {/* Page Header */}
        <div className="pt-12 lg:pt-16 pb-10">
          <h1 className="text-4xl lg:text-5xl font-bold text-text-primary">Transformers &amp; NLP</h1>
          <p className="mt-4 text-lg text-text-secondary max-w-2xl leading-relaxed">
            The architecture behind GPT, BERT, and every modern language model. We build it from the
            ground up, one piece at a time.
          </p>
        </div>

        <TransformersOverview />

        <div className="border-t border-obsidian-border" />
        <TokenizationSection />

        <div className="border-t border-obsidian-border" />
        <PositionalEncodingSection />

        <div className="border-t border-obsidian-border" />
        <SelfAttentionSection />

        <div className="border-t border-obsidian-border" />
        <MultiHeadAttentionSection />

        <div className="border-t border-obsidian-border" />
        <TransformerBlockSection />

        <div className="border-t border-obsidian-border" />
        <EncoderDecoderSection />

        <div className="border-t border-obsidian-border" />
        <PuttingItTogetherSection />

        <div className="h-20" />
      </div>
    </div>
  )
}
