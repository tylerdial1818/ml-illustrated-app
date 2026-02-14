import { ModelSection } from '../../../components/ui/ModelSection'
import { RNNLSTMViz } from '../visualizations/RNNLSTMViz'
import { RNNLSTMMath } from '../content/rnnLstmMath'

export function RNNLSTMSection() {
  return (
    <ModelSection
      id="rnn-lstm"
      title="RNN & LSTM"
      subtitle="Process sequences step by step, carrying memory forward — and learn what to remember and what to forget."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            <strong className="text-text-primary">RNN — reading a sentence word by word.</strong>{' '}
            As you read each word, you carry a mental summary of everything you've read so far. That
            summary (the hidden state) gets updated at each step, blending the new word with your
            running understanding. The problem: as the sentence gets longer, early words fade from memory.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            <strong className="text-text-primary">LSTM — highlighting a textbook.</strong>{' '}
            The LSTM solves the forgetting problem by adding a separate "notebook" (cell state) that runs
            alongside. At each step, it decides what to erase from the notebook (forget gate), what new
            notes to write (input gate), and what to read back out (output gate). This lets it remember
            important information across hundreds of time steps.
          </p>
        </div>
      }
      mechanism={<RNNLSTMViz />}
      math={<RNNLSTMMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Naturally handles sequential data</li>
                <li>Variable-length inputs — no fixed size needed</li>
                <li>LSTM mitigates vanishing gradient problem</li>
                <li>Good for time series, text, and audio</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Hard to parallelize — processes one step at a time</li>
                <li>Vanilla RNN suffers from vanishing gradients</li>
                <li>Largely superseded by Transformers for NLP</li>
                <li>Slow to train on very long sequences</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
