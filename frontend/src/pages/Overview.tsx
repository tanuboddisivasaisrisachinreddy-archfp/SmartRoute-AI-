import { Link } from 'react-router-dom';
import { Navigation } from 'lucide-react';
import { useAsync } from '../hooks/useAsync';
import { api } from '../services/api';
import { ALGO, Button, Chip, Empty, ErrorState, Loading, PageHeader, Panel, Stat, fmt } from '../components/ui';
import { AreaTrend, BarsChart, TrendChart } from '../charts/Charts';
import { TrafficModeBadge } from '../components/TrafficModeBadge';
import type { AlgoSummary } from '../services/types';

const clockFmt = (v: unknown) => {
  const m = Number(v);
  return `${String(Math.floor((m / 60) % 24)).padStart(2, '0')}:${String(Math.floor(m % 60)).padStart(2, '0')}`;
};

function modelLine(name: string, s?: { loaded: boolean; reason?: string; tables?: number; episodes?: number; rolling_success?: number }) {
  return (
    <div className="flex items-start justify-between gap-3 py-2.5 text-sm">
      <span className="text-granite-600 dark:text-granite-300">{name}</span>
      <span className="text-right">
        {s?.loaded ? (
          <Chip tone="ok">{s.tables !== undefined ? `${s.tables} Q-tables` : `${s.episodes ?? '?'} episodes, ${fmt((s.rolling_success ?? 0) * 100, 0)}% success`}</Chip>
        ) : (
          <Chip tone="warn">{s?.reason ?? 'not loaded'}</Chip>
        )}
      </span>
    </div>
  );
}

export default function Overview() {
  const analytics = useAsync(() => api.analytics(), [], 30_000);
  const history = useAsync(() => api.trainingHistory(), []);

  if (analytics.loading && !analytics.data) return <Loading label="Loading dashboard…" />;
  if (analytics.error && !analytics.data) return <ErrorState message={analytics.error} onRetry={() => void analytics.reload()} />;
  const a = analytics.data!;
  const algo = (id: string): AlgoSummary | undefined => a.algorithms.find((x) => x.algorithm === id);
  const ql = algo('q_learning');
  const dq = algo('dqn');
  const hasRoutes = a.total_comparisons > 0;

  const curveData = (() => {
    const q = history.data?.q_learning?.curve ?? [];
    const d = history.data?.dqn?.curve ?? [];
    const n = Math.max(q.length, d.length);
    const qStep = q.length ? q[q.length - 1].episode / q.length : 1;
    const dStep = d.length ? d[d.length - 1].episode / d.length : 1;
    return Array.from({ length: n }, (_, i) => ({
      progress: Math.round(((i + 1) / n) * 100),
      q_learning: q[Math.floor((i / n) * q.length)]?.avg_reward ?? null,
      dqn: d[Math.floor((i / n) * d.length)]?.avg_reward ?? null,
      qEp: Math.round((i + 1) * qStep),
      dEp: Math.round((i + 1) * dStep),
    }));
  })();

  return (
    <div>
      <PageHeader
        title="Hyderabad routing at a glance"
        subtitle="Every figure on this page is computed by the backend from stored route comparisons, experiments and training logs."
        actions={<Link to="/route"><Button variant="primary" icon={<Navigation className="h-4 w-4" />}>Plan a route</Button></Link>}
      />

      <Panel className="mb-5">
        <div className="grid grid-cols-2 gap-6 md:grid-cols-4 xl:grid-cols-8">
          <Stat label="Routes evaluated" value={a.total_routes_evaluated.toLocaleString()} hint={`${a.total_comparisons} comparisons`} />
          <Stat label="Avg travel time" value={fmt(a.avg_travel_time_min, 1)} hint="minutes, successful routes" />
          <Stat label="Avg distance" value={fmt(a.avg_distance_km, 2)} hint="km" />
          <Stat label="Avg congestion" value={fmt(a.avg_congestion, 2)} hint="time-weighted, 0–1" />
          <Stat label="Q-Learning" accent={ALGO.q_learning.color} value={ql?.success_rate != null ? `${fmt(ql.success_rate * 100, 0)}%` : '–'} hint={ql?.avg_score != null ? `avg score ${fmt(ql.avg_score, 0)}` : 'no runs yet'} />
          <Stat label="DQN" accent={ALGO.dqn.color} value={dq?.success_rate != null ? `${fmt(dq.success_rate * 100, 0)}%` : '–'} hint={dq?.avg_score != null ? `avg score ${fmt(dq.avg_score, 0)}` : 'no runs yet'} />
          <Stat label="Training" value={<span className="text-[1.2rem] capitalize">{a.training}</span>} hint={<Link className="underline" to="/training">open training</Link>} />
          <div>
            <div className="text-sm text-granite-500 dark:text-granite-300">Traffic mode</div>
            <div className="mt-2"><TrafficModeBadge mode={a.traffic?.mode} label={a.traffic?.label} /></div>
            <div className="mt-1.5 text-xs text-granite-500">{a.traffic ? `${a.traffic.scenario_label}, ${a.traffic.clock}` : ''}</div>
          </div>
        </div>
      </Panel>

      <div className="grid gap-5 lg:grid-cols-3">
        <Panel title="Route comparison" className="lg:col-span-2">
          {hasRoutes ? (
            <BarsChart
              data={a.algorithms.filter((x) => x.runs > 0).map((x) => ({ name: x.label, time: x.avg_travel_time_min, distance: x.avg_distance_km }))}
              xKey="name"
              series={[{ key: 'time', name: 'Avg travel time (min)', color: '#566a6d' }, { key: 'distance', name: 'Avg distance (km)', color: '#e3a21a' }]}
            />
          ) : (
            <Empty title="No routes compared yet" action={<Link to="/route"><Button>Compare a route</Button></Link>}>
              Plan a trip on the route optimizer and the averages for each algorithm appear here.
            </Empty>
          )}
        </Panel>
        <Panel title="Model status">
          <div className="divide-y divide-granite-100 dark:divide-granite-800">
            {modelLine('Dijkstra baseline', { loaded: true, tables: undefined, episodes: undefined, reason: 'always available' })}
            {modelLine('Q-Learning', a.models.q_learning)}
            {modelLine('Deep Q-Network', a.models.dqn)}
          </div>
          <p className="mt-3 text-xs text-granite-500">Storage: {a.database === 'mongodb' ? 'MongoDB Atlas' : 'local JSON files (set MONGODB_URI for Atlas)'}</p>
        </Panel>

        <Panel title="Travel time trend">
          {a.timeline.length > 1 ? (
            <TrendChart data={a.timeline.map((t, i) => ({ ...t, n: i + 1 }))} xKey="n" yLabel="min"
              series={[{ key: 'travel_time_min', name: 'Best route time', color: '#2563eb' }]} height={210} />
          ) : (
            <Empty title="Not enough trips yet">The trend needs at least two comparisons.</Empty>
          )}
        </Panel>
        <Panel title="Congestion trend (simulation clock)">
          {a.traffic?.history && a.traffic.history.length > 1 ? (
            <AreaTrend data={a.traffic.history as unknown as Record<string, unknown>[]} xKey="clock_min" xFormatter={clockFmt} yMax={1}
              series={[{ key: 'mean_congestion', name: 'Mean', color: '#e07a1f' }, { key: 'p90_congestion', name: '90th percentile', color: '#c2362f' }]} height={210} />
          ) : (
            <Empty title="Simulation not advanced yet" action={<Link to="/traffic"><Button>Open traffic simulation</Button></Link>}>
              Advance the simulation clock to record congestion over time.
            </Empty>
          )}
        </Panel>
        <Panel title="Training reward progression">
          {curveData.length > 1 ? (
            <TrendChart data={curveData} xKey="progress" xFormatter={(v) => `${v}%`}
              series={[{ key: 'q_learning', name: 'Q-Learning', color: ALGO.q_learning.color }, { key: 'dqn', name: 'DQN', color: ALGO.dqn.color }]} height={210} />
          ) : (
            <Empty title="No training history" action={<Link to="/training"><Button>Start training</Button></Link>}>
              Train the agents to see the average reward rise across episodes (x-axis: % of training).
            </Empty>
          )}
        </Panel>

        <Panel title="Algorithm performance" className="lg:col-span-3">
          {hasRoutes ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead className="text-left text-granite-500">
                  <tr>{['Algorithm', 'Runs', 'Success', 'Avg time', 'Avg distance', 'Avg congestion', 'Avg reward', 'Avg compute', 'Best route count'].map((h) => <th key={h} className="pb-2 font-medium">{h}</th>)}</tr>
                </thead>
                <tbody className="divide-y divide-granite-100 dark:divide-granite-800">
                  {a.algorithms.map((x) => (
                    <tr key={x.algorithm}>
                      <td className="py-2"><span className="mr-2 inline-block h-2 w-4 rounded-full" style={{ background: ALGO[x.algorithm].color }} />{x.label}</td>
                      <td>{x.runs}</td>
                      <td>{x.success_rate != null ? `${fmt(x.success_rate * 100, 0)}%` : '–'}</td>
                      <td>{fmt(x.avg_travel_time_min, 1)} min</td>
                      <td>{fmt(x.avg_distance_km, 2)} km</td>
                      <td>{fmt(x.avg_congestion ?? null, 2)}</td>
                      <td>{fmt(x.avg_reward ?? null, 1)}</td>
                      <td>{fmt(x.avg_computation_ms, 0)} ms</td>
                      <td>{x.wins}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty title="No measurements yet">Results appear after the first comparison.</Empty>
          )}
        </Panel>
      </div>
    </div>
  );
}
