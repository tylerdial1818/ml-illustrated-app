import { ModelSection } from '../../../components/ui/ModelSection'
import { TokenizationPipeline } from '../visualizations/TokenizationPipeline'
import { TokenizationMath } from '../content/tokenizationMath'

export function TokenizationSection() {
  return (
    <ModelSection
      id="tokenization"
      title="Tokenization & Embeddings"
      subtitle="Before a model can process text, it needs to convert words into numbers."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Neural networks only understand numbers. They can't read the word "cat" any more than
            a calculator can. So the very first step is to convert text into a format the model can
            work with: vectors of numbers.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            <strong className="text-text-primary">Tokenization</strong> breaks text into smaller
            pieces called tokens. These aren't always full words. Modern tokenizers use subword
            algorithms like Byte Pair Encoding (BPE), which can split uncommon words into familiar
            parts. The word "unhappiness" might become ["un", "happiness"], while common words like
            "the" stay whole.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            <strong className="text-text-primary">Embedding</strong> then maps each token to a
            dense vector (a list of numbers, typically 768 or more dimensions). These vectors are
            learned during training so that tokens with similar meanings end up with similar vectors.
            "King" and "queen" will be close in this space. "King" and "toaster" will not.
          </p>
        </div>
      }
      mechanism={<TokenizationPipeline />}
      math={<TokenizationMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Subword tokenization (BPE) handles any word, even novel ones</li>
                <li>Fixed vocabulary size keeps the model manageable</li>
                <li>Embeddings capture semantic similarity automatically</li>
                <li>Standard preprocessing step for all Transformer models</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Tokenization is language-dependent and can fragment rare words</li>
                <li>Embedding dimensions are fixed (768 for BERT, 4096+ for GPT-3/4)</li>
                <li>Vocabulary size is a trade-off: too small fragments everything, too large wastes memory</li>
                <li>Subword splits can feel arbitrary to humans</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
