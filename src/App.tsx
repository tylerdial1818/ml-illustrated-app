import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { HomePage } from './features/home/HomePage'
import { BasicsPage } from './features/basics/BasicsPage'
import { ClusteringPage } from './features/clustering/ClusteringPage'
import { RegressionPage } from './features/regression/RegressionPage'
import { AboutPage } from './features/about/AboutPage'

const TreesPage = lazy(() =>
  import('./features/trees/TreesPage').then((m) => ({ default: m.TreesPage }))
)
const NeuralNetworksPage = lazy(() =>
  import('./features/neural-networks/NeuralNetworksPage').then((m) => ({
    default: m.NeuralNetworksPage,
  }))
)
const BayesianPage = lazy(() =>
  import('./features/bayesian/BayesianPage').then((m) => ({
    default: m.BayesianPage,
  }))
)
const TransformersPage = lazy(() =>
  import('./features/transformers/TransformersPage').then((m) => ({
    default: m.TransformersPage,
  }))
)

function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-text-tertiary text-sm">Loading...</div>
    </div>
  )
}

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/basics" element={<BasicsPage />} />
        <Route path="/clustering" element={<ClusteringPage />} />
        <Route path="/regression" element={<RegressionPage />} />
        <Route
          path="/trees"
          element={
            <Suspense fallback={<PageLoader />}>
              <TreesPage />
            </Suspense>
          }
        />
        <Route
          path="/bayesian"
          element={
            <Suspense fallback={<PageLoader />}>
              <BayesianPage />
            </Suspense>
          }
        />
        <Route
          path="/neural-networks"
          element={
            <Suspense fallback={<PageLoader />}>
              <NeuralNetworksPage />
            </Suspense>
          }
        />
        <Route
          path="/transformers"
          element={
            <Suspense fallback={<PageLoader />}>
              <TransformersPage />
            </Suspense>
          }
        />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </AppShell>
  )
}

export default App
