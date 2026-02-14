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
      <main className="min-h-screen lg:pl-64">
        <div className="mx-auto max-w-5xl px-6 sm:px-10 lg:px-16">
          {children}
        </div>
      </main>
    </div>
  )
}
