import { useEffect, useState } from 'react';
import { Crosshair, FastForward, Radio, RotateCcw, Siren, TrendingUp, X } from 'lucide-react';
import HyderabadMap from '../map/HyderabadMap';
import { Button, Chip, Empty, ErrorState, Field, Loading, PageHeader, Panel, Segmented, Stat, Toast, congestionColor, fmt, inputClass } from '../components/ui';
import { TrafficModeBadge } from '../components/TrafficModeBadge';
import { AreaTrend } from '../charts/Charts';
import { api, errorMessage } from '../services/api';
import type { NetworkData, SimScenario, TrafficState } from '../services/types';
import { useIsDark } from '../hooks/useTheme';

const clockFmt = (v: unknown) => {
  const m = Number(v);
  return `${String(Math.floor((m / 60) % 24)).padStart(2, '0')}:${String(Math.floor(m % 60)).padStart(2, '0')}`;
};

export default function TrafficSimulation() {
  const dark = useIsDark();
  const [traffic, setTraffic] = useState<TrafficState | null>(null);
  const [network, setNetwork] = useState<NetworkData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [toast, setToast] = useState<{ msg: string; tone: 'ok' | 'bad' } | null>(null);
  const [placing, setPlacing] = useState(false);
  const [severity, setSeverity] = useState(0.7);
  const [radius, setRadius] = useState(800);
  const [duration, setDuration] = useState(45);
  const [intensity, setIntensity] = useState(1);
  const [clock, setClock] = useState('09:30');

  const load = async () => {
    try {
      const [net, tr] = await Promise.all([api.network(), api.traffic(true)]);
      setNetwork(net);
      setTraffic(tr);
      setIntensity(tr.intensity_scale);
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    }
  };
  useEffect(() => {
    void load();
  }, []);

  const act = async (key: string, fn: () => Promise<TrafficState>, msg: string) => {
    setBusy(key);
    try {
      const t = await fn();
      setTraffic(t);
      setToast({ msg, tone: 'ok' });
    } catch (e) {
      setToast({ msg: errorMessage(e), tone: 'bad' });
    } finally {
      setBusy(null);
    }
  };

  if (error && !traffic) return <ErrorState message={error} onRetry={() => void load()} />;
  if (!traffic) return <Loading label="Loading Hyderabad road network and traffic…" />;

  return (
    <div>
      <PageHeader
        title="Traffic simulation"
        subtitle="A time-dependent congestion model over every road segment. Change conditions here and the route optimizer uses them under “Current simulation state”."
        actions={<TrafficModeBadge mode={traffic.mode} label={traffic.label} />}
      />
      <div className="grid gap-5 xl:grid-cols-[1fr_380px]">
        <div className="space-y-5">
          <Panel padded={false} className="overflow-hidden">
            <div className="relative h-[520px]">
              <HyderabadMap
                network={network}
                congestion={traffic.edge_congestion}
                incidents={traffic.incidents}
                dark={dark}
                onMapClick={(lat, lon) => {
                  if (!placing) return;
                  setPlacing(false);
                  void act('place', () => api.incident({ lat, lon, severity, radius_m: radius, duration_min: duration }), 'Incident added at the clicked point.');
                }}
              />
              {placing && (
                <div className="absolute left-1/2 top-3 z-[500] -translate-x-1/2 rounded-lg bg-granite-900 px-3 py-1.5 text-sm text-white shadow">
                  Click a road to place the incident
                </div>
              )}
              <div className="absolute bottom-6 left-3 z-[500] rounded-lg bg-white/95 px-3 py-2 text-xs shadow dark:bg-granite-900/95">
                {[0.1, 0.3, 0.5, 0.7].map((c, i) => (
                  <div key={c} className="flex items-center gap-2">
                    <span className="h-1.5 w-5 rounded-full" style={{ background: congestionColor(c) }} />
                    {['Free-flowing', 'Moderate', 'Heavy', 'Severe'][i]}
                  </div>
                ))}
              </div>
            </div>
          </Panel>
          <Panel title="Network congestion over simulated time">
            {traffic.history && traffic.history.length > 1 ? (
              <AreaTrend data={traffic.history as unknown as Record<string, unknown>[]} xKey="clock_min" xFormatter={clockFmt} yMax={1}
                series={[{ key: 'level', name: 'Network level', color: '#566a6d' }, { key: 'mean_congestion', name: 'Mean congestion', color: '#e07a1f' }, { key: 'p90_congestion', name: '90th percentile', color: '#c2362f' }]} />
            ) : (
              <Empty title="No history yet">Advance the simulation to record how congestion evolves.</Empty>
            )}
          </Panel>
        </div>

        <div className="space-y-5">
          <Panel>
            <div className="grid grid-cols-2 gap-5">
              <Stat label="Simulation clock" value={traffic.clock} hint={`${fmt(traffic.elapsed_min, 0)} min elapsed`} />
              <Stat label="Network level" value={fmt(traffic.level, 2)} hint={`time-of-day factor ${fmt(traffic.time_of_day_factor, 2)}`} />
              <Stat label="Mean congestion" value={fmt(traffic.mean_congestion, 2)} hint={`${fmt(traffic.heavy_share * 100, 0)}% of segments heavy`} />
              <Stat label="Active incidents" value={traffic.active_incidents} hint={traffic.surge > 0 ? `surge +${fmt(traffic.surge, 2)}` : 'no surge'} />
            </div>
          </Panel>

          <Panel title="Traffic intensity">
            <div className="space-y-4">
              <Segmented<SimScenario>
                label="Scenario"
                value={traffic.scenario}
                onChange={(s) => void act('scenario', () => api.simulate({ scenario: s }), `Scenario set to ${s}.`)}
                options={[{ value: 'low', label: 'Low' }, { value: 'moderate', label: 'Moderate' }, { value: 'heavy', label: 'Heavy' }, { value: 'peak', label: 'Peak' }, { value: 'random', label: 'Random' }]}
              />
              <Field label={`Intensity multiplier: ${intensity.toFixed(2)}×`}>
                <div className="flex items-center gap-3">
                  <input type="range" min={0} max={2} step={0.05} value={intensity} onChange={(e) => setIntensity(Number(e.target.value))} className="w-full accent-turmeric-500" />
                  <Button loading={busy === 'intensity'} onClick={() => void act('intensity', () => api.simulate({ intensity }), 'Intensity updated.')}>Apply</Button>
                </div>
              </Field>
              <Field label="Set simulation clock">
                <div className="flex gap-2">
                  <input type="time" className={inputClass} value={clock} onChange={(e) => setClock(e.target.value)} />
                  <Button loading={busy === 'clock'} onClick={() => void act('clock', () => api.simulate({ clock }), `Clock set to ${clock}.`)}>Set</Button>
                </div>
              </Field>
              <div className="flex flex-wrap gap-2">
                {[5, 15, 60].map((m) => (
                  <Button key={m} loading={busy === `adv${m}`} icon={<FastForward className="h-4 w-4" />} onClick={() => void act(`adv${m}`, () => api.advance(m), `Advanced ${m} minutes.`)}>
                    +{m} min
                  </Button>
                ))}
              </div>
              <Button className="w-full" loading={busy === 'surge'} icon={<TrendingUp className="h-4 w-4" />} onClick={() => void act('surge', () => api.simulate({ surge: 0.3 }), 'Sudden traffic increase triggered; it recovers over ~30 simulated minutes.')}>
                Trigger sudden traffic increase
              </Button>
            </div>
          </Panel>

          <Panel title="Road incidents">
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2 text-xs">
                <Field label={`Severity ${severity.toFixed(2)}`}><input type="range" min={0.2} max={0.95} step={0.05} value={severity} onChange={(e) => setSeverity(Number(e.target.value))} className="w-full accent-red-600" /></Field>
                <Field label={`Radius ${radius} m`}><input type="range" min={200} max={2500} step={100} value={radius} onChange={(e) => setRadius(Number(e.target.value))} className="w-full accent-red-600" /></Field>
                <Field label={`${duration} min`}><input type="range" min={5} max={180} step={5} value={duration} onChange={(e) => setDuration(Number(e.target.value))} className="w-full accent-red-600" /></Field>
              </div>
              <div className="flex gap-2">
                <Button className="flex-1" variant={placing ? 'primary' : 'secondary'} icon={<Crosshair className="h-4 w-4" />} onClick={() => setPlacing((p) => !p)}>
                  {placing ? 'Cancel placing' : 'Place on map'}
                </Button>
                <Button className="flex-1" loading={busy === 'rand'} icon={<Siren className="h-4 w-4" />} onClick={() => void act('rand', () => api.incident({ random: true, severity, radius_m: radius, duration_min: duration }), 'Random incident added.')}>
                  Random incident
                </Button>
              </div>
              {traffic.incidents.length === 0 ? (
                <p className="text-sm text-granite-500">No incidents. Place one on the map to see congestion spread.</p>
              ) : (
                <ul className="divide-y divide-granite-100 text-sm dark:divide-granite-800">
                  {traffic.incidents.map((i) => (
                    <li key={i.id} className="flex items-center gap-2 py-2">
                      <span className="h-2.5 w-2.5 rounded-full bg-red-600" style={{ opacity: 0.3 + 0.7 * i.current_intensity }} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate">{i.label}</span>
                        <span className="text-xs text-granite-500">{i.status}, {i.affected_edges} segments, intensity {i.current_intensity.toFixed(2)}</span>
                      </span>
                      <button aria-label={`Clear incident ${i.id}`} className="rounded p-1 text-granite-400 hover:bg-granite-100 hover:text-red-600 dark:hover:bg-granite-800" onClick={() => void act(`clr${i.id}`, () => api.clearIncident(i.id), 'Incident cleared.')}>
                        <X className="h-4 w-4" />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              {traffic.incidents.length > 0 && (
                <Button variant="danger" className="w-full" loading={busy === 'clrall'} onClick={() => void act('clrall', () => api.clearIncident(), 'All incidents cleared.')}>
                  Clear all incidents
                </Button>
              )}
            </div>
          </Panel>

          <Panel title="Live traffic">
            {traffic.tomtom_configured ? (
              <div className="space-y-3 text-sm">
                <p className="text-granite-600 dark:text-granite-300">TomTom is configured. Fetching samples arterial points across Hyderabad; edges without a nearby sample keep simulated values.</p>
                {traffic.mode === 'live' && traffic.live && <Chip tone="live">{String(traffic.live.sampled_points)} points, {String(traffic.live.covered_edges)} segments</Chip>}
                <div className="flex gap-2">
                  <Button icon={<Radio className="h-4 w-4" />} loading={busy === 'live'} onClick={() => void act('live', () => api.liveRefresh(), 'Live traffic fetched from TomTom.')}>Fetch live traffic</Button>
                  {traffic.mode === 'live' && <Button variant="ghost" onClick={() => void act('liveoff', () => api.liveDisable(), 'Back to simulation only.')}>Use simulation only</Button>}
                </div>
              </div>
            ) : (
              <p className="text-sm text-granite-600 dark:text-granite-300">
                No TomTom API key is configured on the backend, so all traffic shown is simulated. Set <code className="rounded bg-granite-100 px-1 dark:bg-granite-800">TOMTOM_API_KEY</code> in the backend environment to enable live mode.
              </p>
            )}
          </Panel>

          <Button variant="ghost" className="w-full" icon={<RotateCcw className="h-4 w-4" />} loading={busy === 'reset'} onClick={() => void act('reset', () => api.resetTraffic('moderate', '09:30'), 'Simulation reset to moderate traffic at 09:30.')}>
            Reset simulation
          </Button>
        </div>
      </div>
      {toast && <Toast message={toast.msg} tone={toast.tone} onClose={() => setToast(null)} />}
    </div>
  );
}
