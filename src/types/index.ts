export interface Point2D {
  x: number
  y: number
}

export interface LabeledPoint2D extends Point2D {
  label?: number
}

export interface ClusterAssignment {
  pointIndex: number
  clusterId: number
}

export interface NavItem {
  label: string
  path: string
  icon: React.ReactNode
  description: string
  available: boolean
}

export const COLORS = {
  clusters: ['#6366F1', '#F472B6', '#34D399', '#FBBF24', '#38BDF8', '#A78BFA', '#FB923C', '#E879F9'],
  accent: '#818CF8',
  error: '#F87171',
  success: '#4ADE80',
  noise: '#52525B',
} as const
