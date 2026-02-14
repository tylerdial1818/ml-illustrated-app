import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function RNNLSTMMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Vanilla RNN</p>
        <p className="text-sm text-text-secondary mb-3">
          At each time step, the hidden state is updated by combining the current input with the previous hidden state:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="h_t = \tanh(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h)" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">LSTM gates</p>
        <p className="text-sm text-text-secondary mb-3">
          The LSTM uses four gates to control information flow through the cell state:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-2 text-center">
          <Eq tex="f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)" display />
          <Eq tex="i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)" display />
          <Eq tex="\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)" display />
          <Eq tex="C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t" display />
          <Eq tex="o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)" display />
          <Eq tex="h_t = o_t \odot \tanh(C_t)" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="f_t" /> is the forget gate — it decides what to erase from memory.{' '}
          <Eq tex="i_t" /> is the input gate — it decides what new information to store.{' '}
          <Eq tex="C_t" /> is the cell state — the long-term memory conveyor belt.{' '}
          <Eq tex="o_t" /> is the output gate — it decides what part of the cell state to expose.
          The <Eq tex="\odot" /> symbol means element-wise multiplication. Together, these gates
          let the LSTM selectively remember and forget across long sequences.
        </p>
      </div>
    </div>
  )
}
