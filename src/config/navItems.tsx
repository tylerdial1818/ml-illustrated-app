import type { NavItem } from '../types'

export const NAV_ITEMS: NavItem[] = [
  {
    label: 'Home',
    path: '/',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <path d="M6 16L16 6l10 10" stroke="#818CF8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.8" />
        <path d="M8 14v10a2 2 0 002 2h12a2 2 0 002-2V14" stroke="#818CF8" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
      </svg>
    ),
    description: 'Overview',
    available: true,
  },
  {
    label: 'ML Basics',
    path: '/basics',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <path d="M16 4L4 28h24L16 4z" stroke="#A1A1AA" strokeWidth="2" strokeLinejoin="round" opacity="0.7" fill="none" />
        <line x1="10" y1="20" x2="22" y2="20" stroke="#A1A1AA" strokeWidth="1.5" opacity="0.4" />
        <circle cx="16" cy="14" r="2" fill="#A1A1AA" opacity="0.6" />
      </svg>
    ),
    description: 'Foundations: loss, gradients, overfitting, PCA',
    available: true,
  },
  {
    label: 'Clustering',
    path: '/clustering',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <circle cx="10" cy="10" r="3" fill="#6366F1" opacity="0.8" />
        <circle cx="22" cy="8" r="2.5" fill="#F472B6" opacity="0.8" />
        <circle cx="14" cy="22" r="3.5" fill="#34D399" opacity="0.8" />
        <circle cx="24" cy="20" r="2" fill="#FBBF24" opacity="0.8" />
        <circle cx="8" cy="16" r="2" fill="#6366F1" opacity="0.5" />
        <circle cx="18" cy="14" r="1.5" fill="#F472B6" opacity="0.5" />
      </svg>
    ),
    description: 'K-Means, DBSCAN, Hierarchical, GMM',
    available: true,
  },
  {
    label: 'Regression',
    path: '/regression',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <line x1="4" y1="26" x2="28" y2="6" stroke="#F472B6" strokeWidth="2" strokeLinecap="round" />
        <circle cx="8" cy="20" r="2" fill="#6366F1" opacity="0.6" />
        <circle cx="14" cy="18" r="2" fill="#6366F1" opacity="0.6" />
        <circle cx="18" cy="14" r="2" fill="#6366F1" opacity="0.6" />
        <circle cx="24" cy="10" r="2" fill="#6366F1" opacity="0.6" />
        <circle cx="11" cy="22" r="1.5" fill="#6366F1" opacity="0.4" />
        <circle cx="21" cy="8" r="1.5" fill="#6366F1" opacity="0.4" />
      </svg>
    ),
    description: 'OLS, Logistic, Ridge, Lasso, ElasticNet',
    available: true,
  },
  {
    label: 'Trees',
    path: '/trees',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <path d="M16 4v8M16 12l-8 8M16 12l8 8" stroke="#34D399" strokeWidth="2" strokeLinecap="round" opacity="0.8" />
        <circle cx="16" cy="4" r="2.5" fill="#34D399" opacity="0.8" />
        <circle cx="8" cy="22" r="2" fill="#34D399" opacity="0.6" />
        <circle cx="24" cy="22" r="2" fill="#34D399" opacity="0.6" />
        <circle cx="4" cy="28" r="1.5" fill="#34D399" opacity="0.4" />
        <circle cx="12" cy="28" r="1.5" fill="#34D399" opacity="0.4" />
        <circle cx="20" cy="28" r="1.5" fill="#34D399" opacity="0.4" />
        <circle cx="28" cy="28" r="1.5" fill="#34D399" opacity="0.4" />
      </svg>
    ),
    description: 'Decision Trees, Random Forests, GBTs',
    available: true,
  },
  {
    label: 'Bayesian',
    path: '/bayesian',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <path d="M4 28 Q10 8, 16 16 Q22 24, 28 4" stroke="#818CF8" strokeWidth="2" strokeLinecap="round" opacity="0.7" fill="none" />
        <path d="M4 28 Q12 12, 16 16 Q20 20, 28 4" stroke="#6366F1" strokeWidth="1.2" strokeLinecap="round" opacity="0.4" fill="none" />
        <path d="M4 28 Q8 4, 16 16 Q24 28, 28 4" stroke="#A5B4FC" strokeWidth="1.2" strokeLinecap="round" opacity="0.4" fill="none" />
        <circle cx="10" cy="18" r="2" fill="#818CF8" opacity="0.6" />
        <circle cx="16" cy="16" r="2" fill="#818CF8" opacity="0.6" />
        <circle cx="22" cy="12" r="2" fill="#818CF8" opacity="0.6" />
      </svg>
    ),
    description: "Bayes' Theorem, Bayesian Regression, GPs",
    available: true,
  },
  {
    label: 'Neural Nets',
    path: '/neural-networks',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <circle cx="6" cy="10" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="6" cy="22" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="16" cy="8" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="16" cy="16" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="16" cy="24" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="26" cy="16" r="2.5" fill="#FBBF24" opacity="0.7" />
        <line x1="8" y1="10" x2="14" y2="8" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="8" y1="10" x2="14" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="8" y1="22" x2="14" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="8" y1="22" x2="14" y2="24" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="18" y1="8" x2="24" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="18" y1="16" x2="24" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="18" y1="24" x2="24" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
      </svg>
    ),
    description: 'Perceptrons, MLPs, CNNs, RNNs',
    available: true,
  },
  {
    label: 'Transformers',
    path: '/transformers',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <rect x="4" y="12" width="5" height="8" rx="1" fill="#F472B6" opacity="0.7" />
        <rect x="13.5" y="12" width="5" height="8" rx="1" fill="#34D399" opacity="0.7" />
        <rect x="23" y="12" width="5" height="8" rx="1" fill="#FBBF24" opacity="0.7" />
        <path d="M6.5 12 C6.5 6, 16 6, 16 12" stroke="#38BDF8" strokeWidth="1.2" opacity="0.5" fill="none" />
        <path d="M6.5 12 C6.5 4, 25.5 4, 25.5 12" stroke="#38BDF8" strokeWidth="0.8" opacity="0.3" fill="none" />
        <path d="M16 12 C16 8, 25.5 8, 25.5 12" stroke="#38BDF8" strokeWidth="1" opacity="0.4" fill="none" />
      </svg>
    ),
    description: 'Attention, Positional Encoding',
    available: true,
  },
  {
    label: 'About',
    path: '/about',
    icon: (
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="11" stroke="#A1A1AA" strokeWidth="2" opacity="0.5" />
        <circle cx="16" cy="12" r="1.5" fill="#A1A1AA" opacity="0.7" />
        <line x1="16" y1="16" x2="16" y2="22" stroke="#A1A1AA" strokeWidth="2" strokeLinecap="round" opacity="0.7" />
      </svg>
    ),
    description: 'About this project',
    available: true,
  },
]
