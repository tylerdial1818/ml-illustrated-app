import { Button } from '../ui/Button'

interface TransportControlsProps {
  isPlaying: boolean
  isAtStart: boolean
  isAtEnd: boolean
  currentStep: number
  totalSteps: number
  speed: number
  onPlay: () => void
  onPause: () => void
  onTogglePlay: () => void
  onStepForward: () => void
  onStepBack: () => void
  onReset: () => void
  onSetSpeed: (speed: number) => void
  className?: string
}

const SPEEDS = [0.5, 1, 2, 4]

export function TransportControls({
  isPlaying,
  isAtStart,
  isAtEnd,
  currentStep,
  totalSteps,
  speed,
  onTogglePlay,
  onStepForward,
  onStepBack,
  onReset,
  onSetSpeed,
  className = '',
}: TransportControlsProps) {
  return (
    <div
      className={`flex items-center gap-2 bg-obsidian-surface/80 backdrop-blur-sm border border-obsidian-border rounded-xl px-3 py-2 ${className}`}
    >
      {/* Reset */}
      <Button
        variant="ghost"
        size="sm"
        onClick={onReset}
        disabled={isAtStart && !isPlaying}
        aria-label="Reset"
        title="Reset"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M2 2v4h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4.5 9.5A5 5 0 1 0 3 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </Button>

      {/* Step back */}
      <Button
        variant="ghost"
        size="sm"
        onClick={onStepBack}
        disabled={isAtStart}
        aria-label="Step back"
        title="Step back"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M3 3v8M12 3L6 7l6 4V3z" fill="currentColor" />
        </svg>
      </Button>

      {/* Play/Pause */}
      <Button
        variant="secondary"
        size="sm"
        onClick={onTogglePlay}
        aria-label={isPlaying ? 'Pause' : 'Play'}
        title={isPlaying ? 'Pause' : 'Play'}
        className="min-w-[32px]"
      >
        {isPlaying ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <rect x="3" y="2" width="3" height="10" rx="1" fill="currentColor" />
            <rect x="8" y="2" width="3" height="10" rx="1" fill="currentColor" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 2l9 5-9 5V2z" fill="currentColor" />
          </svg>
        )}
      </Button>

      {/* Step forward */}
      <Button
        variant="ghost"
        size="sm"
        onClick={onStepForward}
        disabled={isAtEnd}
        aria-label="Step forward"
        title="Step forward"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M11 3v8M2 3l6 4-6 4V3z" fill="currentColor" />
        </svg>
      </Button>

      {/* Divider */}
      <div className="w-px h-4 bg-obsidian-border mx-1" />

      {/* Step counter */}
      <span className="text-xs font-mono text-text-tertiary tabular-nums min-w-[60px] text-center">
        {currentStep + 1} / {totalSteps}
      </span>

      {/* Divider */}
      <div className="w-px h-4 bg-obsidian-border mx-1" />

      {/* Speed controls */}
      <div className="flex items-center gap-0.5">
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => onSetSpeed(s)}
            className={`px-1.5 py-0.5 text-[10px] font-mono rounded transition-colors ${
              speed === s
                ? 'text-accent bg-accent/10'
                : 'text-text-tertiary hover:text-text-secondary'
            }`}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  )
}
