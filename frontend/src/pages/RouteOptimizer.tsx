import { useEffect, useMemo, useState } from 'react';
import { ArrowDownUp, Award, Layers, Loader2, Navigation, Route as RouteIcon, TriangleAlert } from 'lucide-react';
import HyderabadMap, { type MapRoute } from '../map/HyderabadMap';
import { LocationSearch } from '../components/LocationSearch';
import { ALGO, Button, Chip, ErrorState, Field, Segmented, cx, fmt, inputClass } from '../components/ui';
import { TrafficModeBadge } from '../components/TrafficModeBadge';
import { api, errorMessage } from '../services/api';
import type { CompareResponse, Landmark, NetworkData, Place, Preference, RouteResult, Scenario, TripEvent } from '../services/types';
import { useIsDark } from '../hooks/useTheme';

const SCENARIOS: { value: Scenario; label: string }[] = [
  { value: 'current', label: 'Current simulation state' },
  { value: 'low', label: 'Low traffic' },
  { value: 'moderate', label: 'Moderate traffic' },
  { value: 'heavy', label: 'Heavy traffic' },
  { value: 'peak', label: 'Peak-hour traffic' },
  { value: 'random', label: 'Random congestion' },
  { value: 'live', label: 'Live (TomTom, if configured)' },
];

const EVENTS: { value: TripEvent; label: string; hint: string }[] = [
  { value: 'none', label: 'No event', hint: 'Traffic evolves only with the time-of-day profile.' },
  { value: 'incident_on_route', label: 'Accident after departure', hint: 'An accident appears on the fastest planned corridor a few minutes into the trip.' },
  { value: 'surge', label: 'Sudden traffic surge', hint: 'Network-wide congestion jumps at departure and then recovers.' },
];

function RouteCard({ r, best, selected, onSelect }: { r: RouteResult; best: boolean; selected: boolean; onSelect: () => void }) {
  const m = r.metrics;
  const notes = r.diagnostics?.notes as string | undefined;
  return (
    <button
      onClick={onSelect}
      aria-pressed={selected}
      className={cx(
        'w-full rounded-xl border p-3.5 text-left transition-colors',
        selected ? 'border-granite-400 bg-granite-50 dark:border-granite-500 dark:bg-granite-800' : 'border-granite-200 hover:border-granite-300 dark:border-granite-800 dark:hover:border-granite-600',
      )}
    >
      <div className="flex items-center gap-2">
        <span className="h-3 w-8 rounded-full" style={{ background: r.color }} />
        <span className="font-medium text-granite-900 dark:text-white">{r.label}</span>
        {best && (
          <Chip tone="ok">
            <Award className="h-3.5 w-3.5" /> Best
          </Chip>
        )}
        <span className="ml-auto text-sm font-semibold text-granite-700 dark:text-granite-200">{r.success ? `${r.route_score}` : '—'}</span>
      </div>
      {r.success && m ? (
        <div className="mt-3 grid grid-cols-3 gap-x-3 gap-y-2 text-sm">
          <div><div className="text-xs text-granite-500">Time</div><div className="font-medium">{fmt(m.travel_time_min, 1)} min</div></div>
          <div><div className="text-xs text-granite-500">Distance</div><div className="font-medium">{fmt(m.distance_km, 2)} km</div></div>
          <div><div className="text-xs text-granite-500">Congestion</div><div className="font-medium">{m.congestion_level} ({fmt(m.congestion_exposure, 2)})</div></div>
          <div><div className="text-xs text-granite-500">Segments</div><div className="font-medium">{m.segments}</div></div>
          <div><div className="text-xs text-granite-500">Fuel cost</div><div className="font-medium">₹{fmt(m.estimated_cost_inr, 0)}</div></div>
          <div><div className="text-xs text-granite-500">Compute</div><div className="font-medium">{fmt(r.computation_ms, r.computation_ms < 10 ? 1 : 0)} ms</div></div>
        </div>
      ) : (
        <p className="mt-2 flex gap-1.5 text-sm text-red-700 dark:text-red-300">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" /> {r.error ?? notes ?? 'Did not reach the destination.'}
        </p>
      )}
      {r.success && notes && <p className="mt-2 text-xs leading-relaxed text-granite-500 dark:text-granite-400">{notes}</p>}
    </button>
  );
}

export default function RouteOptimizer() {
  const dark = useIsDark();
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [source, setSource] = useState<Place | null>(null);
  const [dest, setDest] = useState<Place | null>(null);
  const [pref, setPref] = useState<Preference>('fastest');
  const [scenario, setScenario] = useState<Scenario>('current');
  const [event, setEvent] = useState<TripEvent>('none');
  const [pick, setPick] = useState<'source' | 'destination' | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAlts, setShowAlts] = useState(false);
  const [overlay, setOverlay] = useState(false);
  const [network, setNetwork] = useState<NetworkData | null>(null);
  const [congestion, setCongestion] = useState<number[] | null>(null);
  const [overlayMsg, setOverlayMsg] = useState<string | null>(null);

  useEffect(() => {
    api.landmarks().then((lms) => {
      setLandmarks(lms);
      const a = lms.find((l) => l.id === 'hitech-city');
      const b = lms.find((l) => l.id === 'charminar');
      if (a) setSource((s) => s ?? { name: a.name, lat: a.lat, lon: a.lon, source: 'landmark' });
      if (b) setDest((d) => d ?? { name: b.name, lat: b.lat, lon: b.lon, source: 'landmark' });
    }).catch((e) => setError(errorMessage(e)));
  }, []);

  useEffect(() => {
    if (!overlay) return;
    let cancelled = false;
    (async () => {
      try {
        setOverlayMsg('Loading road network…');
        const [net, tr] = await Promise.all([network ? Promise.resolve(network) : api.network(), api.traffic(true)]);
        if (cancelled) return;
        setNetwork(net);
        setCongestion(tr.edge_congestion ?? null);
        setOverlayMsg(null);
      } catch (e) {
        if (!cancelled) setOverlayMsg(errorMessage(e));
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlay, result]);

  const onMapClick = async (lat: number, lon: number) => {
    const target = pick ?? (!source ? 'source' : !dest ? 'destination' : null);
    if (!target) return;
    try {
      const p = await api.reverse(lat, lon);
      if (!p.routable) {
        setError('That point is too far from the Hyderabad road network. Pick a point closer to a main road.');
        return;
      }
      const place = { name: p.name, lat, lon, source: p.source };
      if (target === 'source') setSource(place);
      else setDest(place);
      setPick(target === 'source' && !dest ? 'destination' : null);
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const run = async () => {
    if (!source || !dest) {
      setError('Choose a start and a destination first.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const r = await api.compare({ source, destination: dest, preference: pref, scenario, event });
      setResult(r);
      setSelected(r.best_algorithm);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const mapRoutes: MapRoute[] = useMemo(() => {
    if (!result) return [];
    const main = result.routes.filter((r) => r.success).map((r) => ({ id: r.algorithm, color: r.color, geometry: r.geometry, label: `${r.label}: ${fmt(r.metrics?.travel_time_min)} min` }));
    const alts = showAlts ? result.alternatives.map((a) => ({ id: a.id, color: '#566a6d', geometry: a.geometry, label: `${a.label}: ${fmt(a.metrics.travel_time_min)} min`, dashed: true })) : [];
    return [...alts, ...main];
  }, [result, showAlts]);

  const best = result?.routes.find((r) => r.algorithm === result.best_algorithm);
  const incidents = result?.event?.incident ? [result.event.incident] : [];

  return (
    <div className="flex h-full flex-col lg:flex-row">
      <div className="order-2 flex w-full flex-col overflow-y-auto border-r border-granite-200 bg-white lg:order-1 lg:w-[400px] lg:shrink-0 dark:border-granite-800 dark:bg-granite-900">
        <div className="space-y-4 p-5">
          <div>
            <h1 className="text-[1.5rem] font-[750] leading-tight">Plan a Hyderabad trip</h1>
            <p className="mt-1 text-sm text-granite-500 dark:text-granite-300">Dijkstra, Q-Learning and DQN route the same trip on the same road graph and traffic.</p>
          </div>
          <LocationSearch label="Start" value={source} onChange={setSource} dotColor="#ffffff" placeholder="e.g. Gachibowli" active={pick === 'source'} onFocusPick={() => setPick('source')} />
          <div className="flex justify-center">
            <Button variant="ghost" icon={<ArrowDownUp className="h-4 w-4" />} onClick={() => { setSource(dest); setDest(source); }} title="Swap start and destination">
              Swap
            </Button>
          </div>
          <LocationSearch label="Destination" value={dest} onChange={setDest} dotColor="#e3a21a" placeholder="e.g. Charminar" active={pick === 'destination'} onFocusPick={() => setPick('destination')} />
          {landmarks.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {landmarks.slice(0, 19).map((l) => (
                <button
                  key={l.id}
                  onClick={() => {
                    const p = { name: l.name, lat: l.lat, lon: l.lon, source: 'landmark' };
                    if (pick === 'source' || !source) setSource(p);
                    else setDest(p);
                    setPick(null);
                  }}
                  className="rounded-full border border-granite-200 px-2.5 py-0.5 text-xs text-granite-600 hover:border-turmeric-500 hover:text-granite-900 dark:border-granite-700 dark:text-granite-300 dark:hover:text-white"
                >
                  {l.name}
                </button>
              ))}
            </div>
          )}
          <Field label="Travel preference">
            <Segmented label="Travel preference" value={pref} onChange={setPref} options={[{ value: 'fastest', label: 'Fastest' }, { value: 'shortest', label: 'Shortest' }, { value: 'balanced', label: 'Balanced' }]} />
          </Field>
          <Field label="Traffic scenario">
            <select className={inputClass} value={scenario} onChange={(e) => setScenario(e.target.value as Scenario)}>
              {SCENARIOS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </Field>
          <Field label="During the trip" hint={EVENTS.find((e) => e.value === event)?.hint}>
            <select className={inputClass} value={event} onChange={(e) => setEvent(e.target.value as TripEvent)}>
              {EVENTS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </Field>
          <Button variant="primary" className="w-full py-2.5" onClick={run} loading={loading} icon={<Navigation className="h-4 w-4" />}>
            {loading ? 'Finding routes…' : 'Find best route'}
          </Button>
          {loading && <p className="text-xs text-granite-500">Q-Learning trains a small on-demand Q-table when the destination is not a pre-trained landmark; this can take a few seconds.</p>}
          {error && <ErrorState message={error} />}
        </div>

        {result && (
          <div className="space-y-3 border-t border-granite-200 p-5 dark:border-granite-800">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-[1.05rem] font-[650]">Results</h2>
              <TrafficModeBadge mode={result.traffic_mode} label={result.traffic_label} />
            </div>
            <p className="text-xs text-granite-500 dark:text-granite-400">
              Departure {result.departure_clock} (simulation clock), scenario {result.scenario}, {result.preference} preference. Score 100 = lowest cost among routes that reached the destination.
            </p>
            {best && (
              <div className="rounded-xl bg-granite-900 p-4 text-sm leading-relaxed text-granite-100 dark:bg-granite-950">
                <div className="mb-1 flex items-center gap-2 font-medium text-white">
                  <RouteIcon className="h-4 w-4 text-turmeric-400" /> Why {best.label}?
                </div>
                {result.explanation}
              </div>
            )}
            {!best && <ErrorState message={result.explanation} />}
            {result.routes.map((r) => (
              <RouteCard key={r.algorithm} r={r} best={r.algorithm === result.best_algorithm} selected={selected === r.algorithm} onSelect={() => setSelected(r.algorithm)} />
            ))}
            {result.event?.description && <p className="rounded-lg bg-red-50 p-3 text-xs text-red-800 dark:bg-red-950/40 dark:text-red-200">{result.event.description}</p>}
            {result.alternatives.length > 0 && (
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={showAlts} onChange={(e) => setShowAlts(e.target.checked)} />
                Show {result.alternatives.length} Dijkstra alternative routes (dashed)
              </label>
            )}
            {best && best.road_names.length > 0 && (
              <p className="text-xs text-granite-500 dark:text-granite-400">Via {best.road_names.join(', ')}</p>
            )}
          </div>
        )}
      </div>

      <div className="relative order-1 h-[52vh] min-h-[320px] flex-1 lg:order-2 lg:h-auto">
        <HyderabadMap
          routes={mapRoutes}
          selected={selected}
          onSelectRoute={setSelected}
          source={source}
          destination={dest}
          incidents={incidents}
          network={overlay ? network : null}
          congestion={overlay ? congestion : null}
          onMapClick={onMapClick}
          fitKey={result ? `${result.id}-${result.routes.map((r) => r.edge_ids.length).join('-')}` : 'init'}
          dark={dark}
        />
        <div className="pointer-events-none absolute left-14 top-3 z-[500] flex flex-wrap gap-2">
          <div className="pointer-events-auto flex items-center gap-1 rounded-lg bg-white/95 p-1 shadow dark:bg-granite-900/95">
            {(['source', 'destination'] as const).map((p) => (
              <button key={p} onClick={() => setPick(pick === p ? null : p)} className={cx('rounded-md px-2.5 py-1 text-xs', pick === p ? 'bg-turmeric-500 text-granite-950' : 'text-granite-600 hover:bg-granite-100 dark:text-granite-300 dark:hover:bg-granite-800')}>
                {p === 'source' ? 'Set start on map' : 'Set destination on map'}
              </button>
            ))}
            <button onClick={() => setOverlay((o) => !o)} className={cx('flex items-center gap-1 rounded-md px-2.5 py-1 text-xs', overlay ? 'bg-granite-800 text-white dark:bg-granite-600' : 'text-granite-600 hover:bg-granite-100 dark:text-granite-300 dark:hover:bg-granite-800')}>
              <Layers className="h-3.5 w-3.5" /> Traffic layer
            </button>
          </div>
          {overlayMsg && <span className="pointer-events-auto flex items-center gap-1 rounded-lg bg-white/95 px-2 py-1 text-xs shadow dark:bg-granite-900/95">{overlayMsg.startsWith('Loading') && <Loader2 className="h-3 w-3 animate-spin" />}{overlayMsg}</span>}
        </div>
        <div className="absolute bottom-6 left-3 z-[500] rounded-lg bg-white/95 px-3 py-2 text-xs shadow dark:bg-granite-900/95">
          {(Object.keys(ALGO) as (keyof typeof ALGO)[]).map((k) => (
            <div key={k} className="flex items-center gap-2"><span className="h-1.5 w-5 rounded-full" style={{ background: ALGO[k].color }} />{ALGO[k].label}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
