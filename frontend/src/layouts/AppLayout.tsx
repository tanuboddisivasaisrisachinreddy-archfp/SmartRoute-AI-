import { useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  Activity, BarChart3, BrainCircuit, Database, GitCompareArrows, Info, LayoutDashboard, Menu, Moon, Navigation, Sun, X,
} from 'lucide-react';
import { useTheme } from '../hooks/useTheme';
import { useAsync } from '../hooks/useAsync';
import { api } from '../services/api';
import { TrafficModeBadge } from '../components/TrafficModeBadge';
import { cx } from '../components/ui';

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/route', label: 'Route optimizer', icon: Navigation },
  { to: '/traffic', label: 'Traffic simulation', icon: Activity },
  { to: '/training', label: 'RL training', icon: BrainCircuit },
  { to: '/comparison', label: 'Algorithm comparison', icon: GitCompareArrows },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/about', label: 'About', icon: Info },
];

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden>
        <rect width="32" height="32" rx="8" fill="#24353c" />
        <path d="M7 23c4-1 5-7 9-8s5-6 9-7" fill="none" stroke="#e3a21a" strokeWidth="3" strokeLinecap="round" />
        <circle cx="7" cy="23" r="2.6" fill="#fff" />
        <circle cx="25" cy="8" r="2.6" fill="#fff" />
      </svg>
      <div className="leading-tight">
        <div className="font-display text-[1.05rem] font-[750] text-white">SmartRoute AI</div>
        <div className="text-xs text-granite-300">Hyderabad route lab</div>
      </div>
    </div>
  );
}

export default function AppLayout() {
  const { theme, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const health = useAsync(() => api.health(), [], 30_000);
  const traffic = useAsync(() => api.traffic(false), [loc.pathname], 20_000);
  const fullBleed = loc.pathname === '/route';

  const nav = (
    <nav className="flex flex-col gap-0.5" aria-label="Main">
      {NAV.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            cx(
              'flex items-center gap-3 rounded-lg px-3 py-2 text-[0.92rem] transition-colors',
              isActive ? 'bg-white/10 font-medium text-white' : 'text-granite-300 hover:bg-white/5 hover:text-white',
            )
          }
        >
          {({ isActive }) => (
            <>
              <Icon className={cx('h-[18px] w-[18px]', isActive && 'text-turmeric-400')} />
              {label}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );

  const h = health.data;
  const status = (
    <div className="space-y-1.5 border-t border-white/10 pt-4 text-xs text-granite-300">
      {health.error ? (
        <div className="text-red-300">Backend unreachable</div>
      ) : h ? (
        <>
          <div className="flex justify-between"><span>Road graph</span><span className="text-white">{h.graph ? `${h.graph.nodes.toLocaleString()} nodes` : 'not loaded'}</span></div>
          <div className="flex justify-between"><span>Q-Learning</span><span className="text-white">{h.models.q_learning?.loaded ? 'trained' : 'on-demand'}</span></div>
          <div className="flex justify-between"><span>DQN</span><span className="text-white">{h.models.dqn?.loaded ? 'trained' : 'not trained'}</span></div>
          <div className="flex justify-between"><span>Storage</span><span className="text-white">{h.database.type === 'mongodb' ? 'MongoDB Atlas' : 'local files'}</span></div>
        </>
      ) : (
        <div>Connecting…</div>
      )}
    </div>
  );

  return (
    <div className="flex h-full">
      <aside className="hidden w-64 shrink-0 flex-col gap-6 bg-granite-900 px-4 py-5 lg:flex dark:bg-[#0f1a1f]">
        <Brand />
        {nav}
        <div className="mt-auto">{status}</div>
      </aside>

      {open && (
        <div className="fixed inset-0 z-[1100] lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 flex w-72 flex-col gap-6 bg-granite-900 px-4 py-5">
            <div className="flex items-center justify-between">
              <Brand />
              <button aria-label="Close menu" onClick={() => setOpen(false)} className="text-granite-300"><X /></button>
            </div>
            {nav}
            <div className="mt-auto">{status}</div>
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-granite-200 bg-white/80 px-4 backdrop-blur dark:border-granite-800 dark:bg-granite-900/80">
          <button className="lg:hidden" aria-label="Open menu" onClick={() => setOpen(true)}><Menu className="h-5 w-5" /></button>
          <TrafficModeBadge mode={traffic.data?.mode} label={traffic.data?.label} />
          {traffic.data && (
            <span className="hidden text-sm text-granite-500 sm:inline dark:text-granite-300">
              {traffic.data.scenario_label}, simulation clock {traffic.data.clock}
            </span>
          )}
          <div className="ml-auto flex items-center gap-2">
            {h?.graph?.synthetic && <span className="text-xs text-red-600">Synthetic test graph loaded</span>}
            <button onClick={toggle} aria-label="Toggle dark mode" className="rounded-lg p-2 text-granite-500 hover:bg-granite-100 dark:text-granite-300 dark:hover:bg-granite-800">
              {theme === 'dark' ? <Sun className="h-[18px] w-[18px]" /> : <Moon className="h-[18px] w-[18px]" />}
            </button>
          </div>
        </header>
        <main className={cx('min-h-0 flex-1', fullBleed ? 'overflow-hidden' : 'overflow-y-auto')}>
          {health.data && !health.data.graph_loaded && !fullBleed && (
            <div className="mx-4 mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 lg:mx-8 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
              The backend is running but the Hyderabad road graph is not loaded: {health.data.startup_error}
            </div>
          )}
          <div className={fullBleed ? 'h-full' : 'mx-auto max-w-[1400px] px-4 py-6 lg:px-8'}>
            <Outlet context={{ health }} />
          </div>
        </main>
      </div>
    </div>
  );
}
