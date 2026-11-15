import { useMemo, useState } from 'react';
import { FlaskConical } from 'lucide-react';
import { useAsync } from '../hooks/useAsync';
import { api, errorMessage } from '../services/api';
import { ALGO, Button, Empty, ErrorState, Field, Loading, PageHeader, Panel, fmt, inputClass } from '../components/ui';
import { BarsChart } from '../charts/Charts';
import type { AlgorithmId, Experiment, ExperimentRow, Preference, SimScenario, TripEvent } from '../services/types';

const SCEN: SimScenario[] = ['low', 'moderate', 'heavy', 'peak', 'random'];
const EVENTS: { v: TripEvent; label: string }[] = [
  { v: 'none', label: 'No event' },
  { v: 'incident_on_route', label: 'Accident after departure' },
  { v: 'surge', label: 'Traffic surge' },
];

function toggle<T>(arr: T[], v: T): T[] {
  return arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v];
}

function findings(rows: ExperimentRow[]): string[] {
  const key = (r: ExperimentRow) => `${r.source}|${r.destination}|${r.scenario}|${r.preference}|${r.event}`;
  const groups = new Map<string, Partial<Record<AlgorithmId, ExperimentRow>>>();
  rows.forEach((r) => groups.set(key(r), { ...(groups.get(key(r)) ?? {}), [r.algorithm]: r }));
  const out: string[] = [];
  (['q_learning', 'dqn'] as AlgorithmId[]).forEach((a) => {
    let better = 0, worse = 0, failed = 0, n = 0;
    let worst: { g: string; ratio: number } | null = null;
    groups.forEach((g, k) => {
      const d = g.dijkstra, x = g[a];
      if (!d?.success || !x) return;
      n += 1;
      if (!x.success) { failed += 1; return; }
      const ratio = (x.travel_time_min ?? 0) / (d.travel_time_min || 1);
      if (ratio < 0.995) better += 1;
      else if (ratio > 1.005) worse += 1;
      if (!worst || ratio > worst.ratio) worst = { g: k, ratio };
    });
    if (!n) return;
    const w = worst as { g: string; ratio: number } | null;
    out.push(
      `${ALGO[a].label}: faster than Dijkstra in ${better} of ${n} trips, slower in ${worse}, failed in ${failed}.` +
        (w && w.ratio > 1.005 ? ` Worst case ${w.g.split('|').slice(0, 2).join(' → ')} (${w.g.split('|')[2]}, ${w.g.split('|')[4]}): ${fmt((w.ratio - 1) * 100, 0)}% slower.` : ''),
    );
  });
  const ev = rows.filter((r) => r.event === 'incident_on_route');
  if (ev.length) {
    const wins = ev.filter((r) => r.best && r.algorithm !== 'dijkstra').length;
    const trips = new Set(ev.map(key)).size;
    out.push(`With an accident appearing after departure, a learned policy produced the best route in ${wins} of ${trips} trips; Dijkstra's static plan cannot react to it.`);
  }
  return out;
}

export default function Comparison() {
  const latest = useAsync(() => api.latestExperiment(), []);
  const [pairs, setPairs] = useState(6);
  const [scen, setScen] = useState<SimScenario[]>(['moderate', 'peak']);
  const [events, setEvents] = useState<TripEvent[]>(['none', 'incident_on_route']);
  const [pref, setPref] = useState<Preference>('fastest');
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setErr(null);
    try {
      const r = await api.runExperiment({ pairs, scenarios: scen, preferences: [pref], events });
      latest.setData(r);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setRunning(false);
    }
  };

  const exp: Experiment | null = latest.data;
  const total = pairs * scen.length * events.length;
  const notes = useMemo(() => (exp ? findings(exp.rows) : []), [exp]);
  const colors = exp?.summary.map((s) => ALGO[s.algorithm].color) ?? [];

  return (
    <div>
      <PageHeader
        title="Algorithm comparison"
        subtitle="Each trip is routed by all three algorithms on the same graph, the same traffic snapshot and the same events, then evaluated with the same time-dependent cost model. Results are measured, not assumed — RL does not always win."
      />
      <Panel title="Run an experiment" className="mb-5">
        <div className="grid gap-4 md:grid-cols-[140px_1fr_1fr_160px_auto] md:items-end">
          <Field label="Landmark pairs"><input type="number" min={1} max={30} className={inputClass} value={pairs} onChange={(e) => setPairs(Number(e.target.value))} /></Field>
          <Field label="Scenarios">
            <div className="flex flex-wrap gap-1.5">
              {SCEN.map((s) => (
                <button key={s} onClick={() => setScen(toggle(scen, s))} aria-pressed={scen.includes(s)} className={`rounded-full border px-2.5 py-1 text-xs ${scen.includes(s) ? 'border-turmeric-500 bg-turmeric-300/30' : 'border-granite-200 dark:border-granite-700'}`}>{s}</button>
              ))}
            </div>
          </Field>
          <Field label="Events">
            <div className="flex flex-wrap gap-1.5">
              {EVENTS.map((s) => (
                <button key={s.v} onClick={() => setEvents(toggle(events, s.v))} aria-pressed={events.includes(s.v)} className={`rounded-full border px-2.5 py-1 text-xs ${events.includes(s.v) ? 'border-turmeric-500 bg-turmeric-300/30' : 'border-granite-200 dark:border-granite-700'}`}>{s.label}</button>
              ))}
            </div>
          </Field>
          <Field label="Preference">
            <select className={inputClass} value={pref} onChange={(e) => setPref(e.target.value as Preference)}>
              <option value="fastest">Fastest</option><option value="balanced">Balanced</option><option value="shortest">Shortest</option>
            </select>
          </Field>
          <Button variant="primary" icon={<FlaskConical className="h-4 w-4" />} loading={running} disabled={!scen.length || !events.length || total > 60} onClick={run}>
            Run {total} trips
          </Button>
        </div>
        {total > 60 && <p className="mt-2 text-xs text-red-600">At most 60 trips per run in the browser. Use scripts/evaluate_models.py for larger experiments.</p>}
        {running && <p className="mt-2 text-xs text-granite-500">Routing {total} trips × 3 algorithms…</p>}
        {err && <div className="mt-3"><ErrorState message={err} /></div>}
      </Panel>

      {latest.loading ? (
        <Loading />
      ) : !exp ? (
        <Empty title="No experiment yet">Run an experiment above, or from the command line with <code>python scripts/evaluate_models.py</code>.</Empty>
      ) : (
        <div className="space-y-5">
          <Panel title={`Results – ${new Date(exp.created_at).toLocaleString()}`}>
            <p className="mb-4 text-sm text-granite-500">
              {new Set(exp.rows.map((r) => `${r.source}${r.destination}${r.scenario}${r.event}`)).size} trips, scenarios {exp.config.scenarios.join(', ')}, events {exp.config.events.join(', ')}. Road graph {exp.graph.nodes.toLocaleString()} nodes{exp.graph.synthetic ? ' (synthetic test graph)' : ''}. Took {exp.duration_s}s.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-sm">
                <thead className="text-left text-granite-500">
                  <tr>{['Algorithm', 'Success rate', 'Avg time', 'Avg distance', 'Congestion exposure', 'Avg reward', 'Avg compute', 'Time vs Dijkstra', 'Best route'].map((h) => <th key={h} className="pb-2 font-medium">{h}</th>)}</tr>
                </thead>
                <tbody className="divide-y divide-granite-100 dark:divide-granite-800">
                  {exp.summary.map((s) => (
                    <tr key={s.algorithm}>
                      <td className="py-2"><span className="mr-2 inline-block h-2 w-4 rounded-full" style={{ background: ALGO[s.algorithm].color }} />{s.label}</td>
                      <td>{fmt((s.success_rate ?? 0) * 100, 0)}%</td>
                      <td>{fmt(s.avg_travel_time_min, 1)} min</td>
                      <td>{fmt(s.avg_distance_km, 2)} km</td>
                      <td>{fmt(s.avg_congestion_exposure ?? null, 3)}</td>
                      <td>{fmt(s.avg_total_reward ?? null, 1)}</td>
                      <td>{fmt(s.avg_computation_ms, 1)} ms</td>
                      <td>{s.time_ratio_vs_dijkstra != null ? `${fmt(s.time_ratio_vs_dijkstra, 3)}×` : '–'}</td>
                      <td>{s.wins}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {notes.length > 0 && (
              <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-granite-700 dark:text-granite-200">
                {notes.map((n) => <li key={n}>{n}</li>)}
              </ul>
            )}
          </Panel>
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
            <Panel title="Avg travel time (min)"><BarsChart height={200} data={exp.summary.map((s) => ({ name: s.label, v: s.avg_travel_time_min }))} xKey="name" series={[{ key: 'v', name: 'minutes', color: '#566a6d' }]} cellColors={colors} /></Panel>
            <Panel title="Success rate (%)"><BarsChart height={200} data={exp.summary.map((s) => ({ name: s.label, v: Math.round((s.success_rate ?? 0) * 100) }))} xKey="name" series={[{ key: 'v', name: '%', color: '#566a6d' }]} cellColors={colors} /></Panel>
            <Panel title="Congestion exposure"><BarsChart height={200} data={exp.summary.map((s) => ({ name: s.label, v: s.avg_congestion_exposure }))} xKey="name" series={[{ key: 'v', name: 'exposure', color: '#566a6d' }]} cellColors={colors} /></Panel>
            <Panel title="Computation time (ms)"><BarsChart height={200} data={exp.summary.map((s) => ({ name: s.label, v: s.avg_computation_ms }))} xKey="name" series={[{ key: 'v', name: 'ms', color: '#566a6d' }]} cellColors={colors} /></Panel>
          </div>
          <Panel title="Every trip">
            <div className="max-h-[420px] overflow-auto">
              <table className="w-full min-w-[820px] text-sm">
                <thead className="sticky top-0 bg-white text-left text-granite-500 dark:bg-granite-900">
                  <tr>{['Trip', 'Scenario', 'Event', 'Algorithm', 'Reached', 'Time', 'Distance', 'Exposure', 'Reward', 'Compute', 'Score'].map((h) => <th key={h} className="py-2 font-medium">{h}</th>)}</tr>
                </thead>
                <tbody className="divide-y divide-granite-100 dark:divide-granite-800">
                  {exp.rows.map((r, i) => (
                    <tr key={i} className={r.best ? 'bg-emerald-50/60 dark:bg-emerald-950/20' : undefined}>
                      <td className="py-1.5">{r.source} → {r.destination}</td><td>{r.scenario}</td><td>{r.event}</td>
                      <td style={{ color: ALGO[r.algorithm].color }}>{ALGO[r.algorithm].label}</td>
                      <td>{r.success ? 'yes' : 'no'}</td><td>{fmt(r.travel_time_min, 1)}</td><td>{fmt(r.distance_km, 2)}</td>
                      <td>{fmt(r.congestion_exposure, 2)}</td><td>{fmt(r.total_reward, 1)}</td><td>{fmt(r.computation_ms, 0)} ms</td><td>{fmt(r.route_score, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}
