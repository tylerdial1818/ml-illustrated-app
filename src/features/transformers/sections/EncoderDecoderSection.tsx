import { ModelSection } from '../../../components/ui/ModelSection'
import { MaskComparison } from '../visualizations/MaskComparison'
import { EncoderDecoderMath } from '../content/encoderDecoderMath'

export function EncoderDecoderSection() {
  return (
    <ModelSection
      id="encoder-decoder"
      title="Encoder vs. Decoder"
      subtitle="Encoders see the whole sentence at once. Decoders can only see what came before."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            There are two flavors of Transformer blocks, and the difference comes down to one
            thing: <strong className="text-text-primary">what each token is allowed to look at</strong>.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            An <strong className="text-text-primary">encoder</strong> uses bidirectional attention.
            Every token can attend to every other token, including those that come after it. This
            makes encoders great for understanding tasks, because the model has full context in
            both directions. "The bank by the river" and "the bank approved the loan" benefit from
            seeing the full sentence to disambiguate "bank."
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            A <strong className="text-text-primary">decoder</strong> uses causal (masked) attention.
            Each token can only attend to tokens at earlier positions and itself. Why the
            restriction? During text generation, future tokens don't exist yet. If the model could
            peek ahead during training, it would learn to cheat instead of learning to predict.
            The mask enforces an honest left-to-right generation process.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            The original Transformer paper used both: an encoder to read the input and a decoder
            to generate the output, connected by cross-attention. But the modern trend is clear:
            decoder-only architectures (GPT-4, Claude, LLaMA) dominate, proving that causal
            language modeling alone is remarkably powerful.
          </p>
        </div>
      }
      mechanism={<MaskComparison />}
      math={<EncoderDecoderMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Encoder-only (BERT): excels at classification, NER, and understanding tasks</li>
                <li>Decoder-only (GPT): excels at text generation and in-context learning</li>
                <li>Encoder-decoder (T5): natural fit for sequence-to-sequence tasks like translation</li>
                <li>Causal masking is simple to implement and enables autoregressive generation</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Encoder-only can't generate text natively</li>
                <li>Decoder-only can't use bidirectional context (may miss backward dependencies)</li>
                <li>Encoder-decoder doubles the parameter count for a given depth</li>
                <li>Choosing the wrong architecture for your task wastes capacity</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
