import { Link } from 'react-router-dom';
import { useAsync } from '../hooks/useAsync';
import { api, http } from '../services/api';
import { ALGO, Button, Empty, ErrorState, Loading, PageHeader, Panel, Stat, fmt } from '../components/ui';
import { BarsChart, TrendChart } from '../charts/Charts';
import type { AlgorithmId } from '../services/types';

interface RecentItem {
  _id: string;
  created_at: string;
  source_name: string;
  destination_name: string;
  preference: string;
  scenario: string;
  event: string | null;
  best_algorithm: AlgorithmId | null;
  routes: { algorithm: AlgorithmId; travel_time_min?: number; success: boolean }[];
}

export default function Analytics() {
  const a = useAsync(() => api.analytics(), []);
  const recent = useAsync(() => http.get<{ items: RecentItem[] }>('/routes/recent', { params: { limit: 25 } }).then((r) => r.data.items), []);

  if (a.loading) return <Loading label="Computing analytics…" />;
  if (a.error || !a.data) return <ErrorState message={a.error ?? 'No data'} onRetry={() => void a.reload()} />;
  const d = a.data;

  return (
    <div>
      <PageHeader
        title="Historical analytics"
        subtitle={`Aggregated from ${d.database === 'mongodb' ? 'MongoDB Atlas' : 'the local JSON store'}: every comparison run through the API is recorded with its measured metrics.`}
        actions={<Button onClick={() => { void a.reload(); void recent.reload(); }}>Refresh</Button>}
      />
      {d.total_comparisons === 0 ? (
        <Empty title="Nothing recorded yet" action={<Link to="/route"><Button variant="primary">Plan a route</Button></Link>}>
          Analytics fill in as you compare routes. Each comparison stores distance, time, congestion and reward for all three algorithms.
        </Empty>
      ) : (
        <div className="space-y-5">
          <Panel>
            <div className="grid grid-cols-2 gap-6 md:grid-cols-5">
              <Stat label="Comparisons" value={d.total_comparisons} />
              <Stat label="Searches logged" value={d.total_searches} />
              <Stat label="Routes evaluated" value={d.total_routes_evaluated} />
              <Stat label="Avg successful trip" value={`${fmt(d.avg_travel_time_min, 1)} min`} hint={`${fmt(d.avg_distance_km, 2)} km`} />
              <Stat label="Avg congestion exposure" value={fmt(d.avg_congestion, 2)} />
            </div>
          </Panel>
          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Best route by algorithm">
              <BarsChart data={d.algorithms.map((x) => ({ name: x.label, wins: x.wins }))} xKey="name" series={[{ key: 'wins', name: 'times best', color: '#566a6d' }]} cellColors={d.algorithms.map((x) => ALGO[x.algorithm].color)} height={220} />
            </Panel>
            <Panel title="Average route score (100 = best in its trip)">
              <BarsChart data={d.algorithms.map((x) => ({ name: x.label, score: x.avg_score }))} xKey="name" series={[{ key: 'score', name: 'score', color: '#566a6d' }]} cellColors={d.algorithms.map((x) => ALGO[x.algorithm].color)} height={220} />
            </Panel>
            <Panel title="Most requested trips">
              <BarsChart layout="vertical" data={d.popular_pairs.map((p) => ({ pair: p.pair, count: p.count }))} xKey="pair" series={[{ key: 'count', name: 'requests', color: '#e3a21a' }]} height={Math.max(160, d.popular_pairs.length * 34)} />
            </Panel>
            <Panel title="Best-route travel time per comparison">
              {d.timeline.length > 1 ? (
                <TrendChart data={d.timeline.map((t, i) => ({ ...t, n: i + 1 }))} xKey="n" yLabel="min" height={220}
                  series={[{ key: 'travel_time_min', name: 'Travel time', color: '#2563eb' }, { key: 'congestion', name: 'Congestion exposure', color: '#c2362f', axis: 'right' }]} rightLabel="exposure" />
              ) : <Empty title="Need more comparisons">Two or more comparisons draw a trend.</Empty>}
            </Panel>
            <Panel title="Scenarios used">
              <BarsChart data={d.scenarios.map((s) => ({ name: s.scenario, count: s.count }))} xKey="name" series={[{ key: 'count', name: 'comparisons', color: '#748687' }]} height={200} />
            </Panel>
            <Panel title="Preferences used">
              <BarsChart data={d.preferences.map((s) => ({ name: s.preference, count: s.count }))} xKey="name" series={[{ key: 'count', name: 'comparisons', color: '#748687' }]} height={200} />
            </Panel>
          </div>
          <Panel title="Recent comparisons">
            {recent.loading ? <Loading /> : recent.error ? <ErrorState message={recent.error} /> : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-sm">
                  <thead className="text-left text-granite-500"><tr>{['When', 'Trip', 'Preference', 'Scenario', 'Event', 'Best', 'Dijkstra', 'Q-Learning', 'DQN'].map((h) => <th key={h} className="pb-2 font-medium">{h}</th>)}</tr></thead>
                  <tbody className="divide-y divide-granite-100 dark:divide-granite-800">
                    {(recent.data ?? []).map((r) => {
                      const t = (id: AlgorithmId) => { const x = r.routes.find((y) => y.algorithm === id); return x?.success ? `${fmt(x.travel_time_min, 1)} min` : 'failed'; };
                      return (
                        <tr key={r._id}>
                          <td className="py-1.5 text-granite-500">{new Date(r.created_at).toLocaleString()}</td>
                          <td>{r.source_name} → {r.destination_name}</td><td>{r.preference}</td><td>{r.scenario}</td><td>{r.event ?? 'none'}</td>
                          <td style={{ color: r.best_algorithm ? ALGO[r.best_algorithm].color : undefined }}>{r.best_algorithm ? ALGO[r.best_algorithm].label : '–'}</td>
                          <td>{t('dijkstra')}</td><td>{t('q_learning')}</td><td>{t('dqn')}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}
