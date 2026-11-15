import { CheckCircle2, ExternalLink, XCircle } from 'lucide-react';
import { useAsync } from '../hooks/useAsync';
import { api } from '../services/api';
import { Chip, ErrorState, Loading, PageHeader, Panel, Stat, fmt } from '../components/ui';
import { BarsChart, TrendChart } from '../charts/Charts';

export default function Datasets() {
  const d = useAsync(() => api.datasets(), []);
  if (d.loading) return <Loading />;
  if (d.error || !d.data) return <ErrorState message={d.error ?? 'No data'} onRetry={() => void d.reload()} />;
  const { catalog, status } = d.data;
  const gs = status.graph_stats;
  const prof = status.traffic_profiles;
  const profileData = prof
    ? Array.from({ length: 24 }, (_, h) => {
        const row: Record<string, number | string | null> = { hour: `${String(h).padStart(2, '0')}:00`, simulated: prof.simulated_hyderabad_profile[h] };
        prof.datasets.forEach((ds) => {
          row[ds.dataset] = ds.hourly_congestion_weekday[h];
        });
        return row;
      })
    : [];
  // scale the simulated profile (a multiplier) to the same 0..1 axis for shape comparison
  const colors: Record<string, string> = { 'METR-LA': '#2563eb', 'PEMS-BAY': '#9333ea' };

  return (
    <div>
      <PageHeader title="Datasets" subtitle="Where every input comes from, what it covers and how it is used. Only OpenStreetMap (and TomTom, when configured) describe Hyderabad; METR-LA and PEMS-BAY are US benchmarks." />

      <div className="grid gap-5 lg:grid-cols-3">
        <Panel title="Hyderabad road graph on this server" className="lg:col-span-2">
          {gs ? (
            <>
              <div className="mb-5 grid grid-cols-2 gap-6 md:grid-cols-4">
                <Stat label="Intersections" value={gs.nodes.toLocaleString()} />
                <Stat label="Road segments" value={gs.edges.toLocaleString()} />
                <Stat label="Road length" value={`${gs.total_km.toLocaleString()} km`} hint="directed segments" />
                <Stat label="One-way share" value={`${fmt(gs.oneway_share * 100, 0)}%`} hint={`mean out-degree ${gs.mean_out_degree}`} />
              </div>
              <BarsChart data={Object.entries(gs.km_by_class).filter(([, v]) => v > 0).map(([k, v]) => ({ name: k, km: v }))} xKey="name" series={[{ key: 'km', name: 'km', color: '#566a6d' }]} height={190} />
            </>
          ) : (
            <p className="text-sm text-granite-500">Processed graph statistics not found. Run <code>python scripts/preprocess_data.py</code>.</p>
          )}
        </Panel>
        <Panel title="Data validation">
          {status.validation ? (
            <ul className="space-y-1.5 text-sm">
              {status.validation.map((c) => (
                <li key={c.check} className="flex gap-2">
                  {c.passed ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" /> : <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />}
                  <span>{c.check}{c.detail && <span className="block text-xs text-granite-500">{c.detail}</span>}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-granite-500">Run <code>python scripts/validate_data.py</code> to generate the validation report.</p>
          )}
        </Panel>
      </div>

      <Panel title="Time-of-day traffic profiles" className="mt-5">
        {prof && prof.datasets.length > 0 ? (
          <>
            <p className="mb-3 text-sm text-granite-500">
              Weekday mean congestion (1 − speed / free-flow) computed from the benchmark datasets, compared with the shape of the simulator's assumed Hyderabad profile. These are Los Angeles and Bay Area measurements, shown for benchmarking only.
            </p>
            <TrendChart data={profileData} xKey="hour" height={260}
              series={[{ key: 'simulated', name: 'Simulator profile (assumed, Hyderabad)', color: '#e3a21a', dashed: true, axis: 'right' }, ...prof.datasets.map((ds) => ({ key: ds.dataset, name: `${ds.dataset} (${ds.geographic_origin})`, color: colors[ds.dataset] ?? '#566a6d' }))]}
              yLabel="congestion" rightLabel="profile factor" />
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {prof.datasets.map((ds) => (
                <div key={ds.dataset} className="text-sm text-granite-600 dark:text-granite-300">
                  <span className="font-medium text-granite-900 dark:text-white">{ds.dataset}</span>: {ds.sensors} sensors, {ds.timesteps.toLocaleString()} timesteps, {ds.start.slice(0, 10)} to {ds.end.slice(0, 10)}, {fmt(ds.missing_rate * 100, 1)}% missing.
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-sm text-granite-500">METR-LA / PEMS-BAY are not downloaded on this server. Run <code>python scripts/download_data.py --traffic</code> then <code>python scripts/preprocess_data.py --traffic-only</code>.</p>
        )}
      </Panel>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        {catalog.map((c) => (
          <Panel key={c.id} title={c.name} action={
            c.is_hyderabad_data === true ? <Chip tone="ok">Hyderabad data</Chip> : c.is_hyderabad_data === false ? <Chip tone="warn">Not Hyderabad data</Chip> : <Chip>Coverage unverified</Chip>
          }>
            <dl className="grid grid-cols-[120px_1fr] gap-x-4 gap-y-2 text-sm">
              <dt className="text-granite-500">Coverage</dt><dd>{c.geographic_coverage}</dd>
              <dt className="text-granite-500">Features</dt><dd>{c.features.join(', ')}</dd>
              <dt className="text-granite-500">Format</dt><dd>{c.file_format}</dd>
              <dt className="text-granite-500">Licence</dt><dd>{c.license}</dd>
              <dt className="text-granite-500">Preprocessing</dt><dd>{c.preprocessing.join('; ')}</dd>
              <dt className="text-granite-500">Used for</dt><dd>{c.intended_use}</dd>
              {c.citation && (<><dt className="text-granite-500">Citation</dt><dd className="text-xs">{c.citation}</dd></>)}
              <dt className="text-granite-500">Sources</dt>
              <dd className="flex flex-col gap-1">
                {c.source_urls.map((u) => (
                  <a key={u} href={u} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-turmeric-700 underline-offset-2 hover:underline dark:text-turmeric-400">
                    {u.replace(/^https?:\/\//, '')} <ExternalLink className="h-3 w-3" />
                  </a>
                ))}
              </dd>
            </dl>
          </Panel>
        ))}
      </div>
    </div>
  );
}
