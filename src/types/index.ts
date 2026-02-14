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
  icon: string
  description: string
  available: boolean
}

export const NAV_ITEMS: NavItem[] = [
  { label: 'Home', path: '/', icon: '◈', description: 'Overview', available: true },
  { label: 'Clustering', path: '/clustering', icon: '⊕', description: 'K-Means, DBSCAN, Hierarchical, GMM', available: true },
  { label: 'Regression', path: '/regression', icon: '⟋', description: 'OLS, Logistic, Ridge, Lasso, ElasticNet', available: true },
  { label: 'Trees', path: '/trees', icon: '⊞', description: 'Decision Trees, Random Forests, GBTs', available: true },
  { label: 'Neural Nets', path: '/neural-networks', icon: '◎', description: 'Perceptrons, MLPs, CNNs, RNNs', available: true },
  { label: 'Transformers', path: '/transformers', icon: '⧫', description: 'Attention, Positional Encoding', available: false },
  { label: 'About', path: '/about', icon: '○', description: 'About this project', available: true },
]

export const COLORS = {
  clusters: ['#6366F1', '#F472B6', '#34D399', '#FBBF24', '#38BDF8', '#A78BFA', '#FB923C', '#E879F9'],
  accent: '#818CF8',
  error: '#F87171',
  success: '#4ADE80',
  noise: '#52525B',
} as const
