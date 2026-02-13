import { Routes, Route } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { HomePage } from './features/home/HomePage'
import { ClusteringPage } from './features/clustering/ClusteringPage'
import { RegressionPage } from './features/regression/RegressionPage'
import { AboutPage } from './features/about/AboutPage'

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/clustering" element={<ClusteringPage />} />
        <Route path="/regression" element={<RegressionPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </AppShell>
  )
}

export default App
