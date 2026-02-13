import { useMemo } from 'react'
import * as d3 from 'd3'

interface ConvergenceChartProps {
  values: number[]
  currentIndex: number
  label?: string
  width?: number
  height?: number
  color?: string
}

export function ConvergenceChart({
  values,
  currentIndex,
  label = 'Cost',
  width = 200,
  height = 80,
  color = '#4ADE80',
}: ConvergenceChartProps) {
  const padding = { top: 8, right: 8, bottom: 20, left: 35 }
  const innerW = width - padding.left - padding.right
  const innerH = height - padding.top - padding.bottom

  const scales = useMemo(() => {
    const xScale = d3.scaleLinear().domain([0, Math.max(values.length - 1, 1)]).range([0, innerW])
    const yExtent = d3.extent(values) as [number, number]
    const yScale = d3.scaleLinear().domain([yExtent[0] * 0.95, yExtent[1] * 1.05]).range([innerH, 0])
    return { xScale, yScale }
  }, [values, innerW, innerH])

  const linePath = useMemo(() => {
    const line = d3
      .line<number>()
      .x((_, i) => scales.xScale(i))
      .y((d) => scales.yScale(d))
      .curve(d3.curveMonotoneX)
    return line(values.slice(0, currentIndex + 1)) ?? ''
  }, [values, currentIndex, scales])

  const currentValue = values[currentIndex]

  return (
    <div className="bg-obsidian-surface/50 rounded-lg border border-obsidian-border p-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] uppercase tracking-wider text-text-tertiary">{label}</span>
        <span className="text-xs font-mono text-text-secondary tabular-nums">
          {currentValue !== undefined ? currentValue.toFixed(2) : '—'}
        </span>
      </div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {/* Grid lines */}
          {scales.yScale.ticks(3).map((tick) => (
            <line
              key={tick}
              x1={0}
              x2={innerW}
              y1={scales.yScale(tick)}
              y2={scales.yScale(tick)}
              stroke="rgba(255,255,255,0.05)"
              strokeDasharray="2,2"
            />
          ))}
          {/* Line */}
          <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" />
          {/* Current dot */}
          {currentValue !== undefined && (
            <circle
              cx={scales.xScale(currentIndex)}
              cy={scales.yScale(currentValue)}
              r={3}
              fill={color}
            />
          )}
          {/* Y axis labels */}
          {scales.yScale.ticks(3).map((tick) => (
            <text
              key={tick}
              x={-4}
              y={scales.yScale(tick)}
              textAnchor="end"
              dominantBaseline="middle"
              className="text-[9px] fill-text-tertiary"
            >
              {tick.toFixed(0)}
            </text>
          ))}
        </g>
      </svg>
    </div>
  )
}
