import { SectionNav } from '../../components/layout/SectionNav'
import { NNOverview } from './sections/NNOverview'
import { PerceptronSection } from './sections/PerceptronSection'
import { MLPSection } from './sections/MLPSection'
import { CNNSection } from './sections/CNNSection'
import { RNNLSTMSection } from './sections/RNNLSTMSection'
import { GANAutoencoderSection } from './sections/GANAutoencoderSection'
import { NNComparison } from './sections/NNComparison'

const SECTIONS = [
  { id: 'nn-overview', label: 'Overview' },
  { id: 'perceptron', label: 'Perceptron' },
  { id: 'mlp', label: 'MLP' },
  { id: 'cnn', label: 'CNN' },
  { id: 'rnn-lstm', label: 'RNN & LSTM' },
  { id: 'gan-autoencoder', label: 'GANs & Autoencoders' },
  { id: 'nn-comparison', label: 'Comparison' },
]

export function NeuralNetworksPage() {
  return (
    <div className="relative">
      <SectionNav sections={SECTIONS} />

      <div>
        {/* Page Header */}
        <div className="pt-12 lg:pt-16 pb-10">
          <h1 className="text-4xl lg:text-5xl font-bold text-text-primary">Neural Networks</h1>
          <p className="mt-4 text-lg text-text-secondary max-w-2xl leading-relaxed">
            From single neurons to deep architectures. Discover how layers of simple
            transformations compose into powerful learned functions.
          </p>
        </div>

        <NNOverview />

        <div className="border-t border-obsidian-border" />
        <PerceptronSection />

        <div className="border-t border-obsidian-border" />
        <MLPSection />

        <div className="border-t border-obsidian-border" />
        <CNNSection />

        <div className="border-t border-obsidian-border" />
        <RNNLSTMSection />

        <div className="border-t border-obsidian-border" />
        <GANAutoencoderSection />

        <div className="border-t border-obsidian-border" />
        <NNComparison />

        <div className="h-20" />
      </div>
    </div>
  )
}
