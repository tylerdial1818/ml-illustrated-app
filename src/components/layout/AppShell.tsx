import { type ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { MobileNav } from './MobileNav'

interface AppShellProps {
  children: ReactNode
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-obsidian-bg">
      <Sidebar />
      <MobileNav />
      <main className="lg:ml-64 min-h-screen">
        <div className="mx-auto max-w-6xl px-6 sm:px-8 lg:px-12">
          {children}
        </div>
      </main>
    </div>
  )
}
