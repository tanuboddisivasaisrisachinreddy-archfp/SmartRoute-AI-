import axios from 'axios';
import type {
  Analytics,
  CompareResponse,
  DatasetsResponse,
  Experiment,
  Health,
  Landmark,
  NetworkData,
  Place,
  Preference,
  Scenario,
  SimScenario,
  TrafficState,
  TrainingHistory,
  TrainingStatus,
  TripEvent,
  AlgorithmId,
} from './types';

const envUrl = (import.meta.env?.VITE_API_URL as string | undefined) ?? '';
// No VITE_API_URL in a production build = frontend is served by the API itself (Docker image).
const sameOrigin = Boolean(import.meta.env?.PROD) && typeof window !== 'undefined' ? window.location.origin : '';
export const API_BASE = (envUrl || sameOrigin || 'http://localhost:8000').replace(/\/+$/, '');

export const http = axios.create({ baseURL: `${API_BASE}/api`, timeout: 90_000 });

interface ValidationItem {
  msg?: string;
  loc?: (string | number)[];
}

/** Turn any axios / network error into a sentence the UI can show. */
export function errorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) {
    const detail = (e.response?.data as { detail?: string | ValidationItem[] } | undefined)?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail.map((d) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg ?? 'invalid'}`).join('; ');
    }
    if (e.code === 'ECONNABORTED') return 'The backend took too long to respond. If it is on Render free tier it may be waking up – try again in a minute.';
    if (!e.response) return `Cannot reach the backend at ${API_BASE}. Check that it is running and VITE_API_URL is correct.`;
    return `Request failed with status ${e.response.status}.`;
  }
  return e instanceof Error ? e.message : 'Unexpected error.';
}

export interface ComparePayload {
  source: Place;
  destination: Place;
  preference: Preference;
  scenario: Scenario;
  event: TripEvent;
  algorithms?: AlgorithmId[];
}

const strip = (p: Place) => ({ lat: p.lat, lon: p.lon, name: p.name });

export const api = {
  health: () => http.get<Health>('/health').then((r) => r.data),
  search: (q: string) =>
    http.get<{ query: string; results: Place[]; warning: string | null }>('/locations/search', { params: { q } }).then((r) => r.data),
  landmarks: () => http.get<{ landmarks: Landmark[] }>('/locations/landmarks').then((r) => r.data.landmarks),
  reverse: (lat: number, lon: number) =>
    http.get<Place & { routable: boolean; snap_distance_m: number }>('/locations/reverse', { params: { lat, lon } }).then((r) => r.data),
  network: () => http.get<NetworkData>('/map/network').then((r) => r.data),
  compare: (p: ComparePayload) =>
    http
      .post<CompareResponse>('/routes/compare', {
        source: strip(p.source),
        destination: strip(p.destination),
        preference: p.preference,
        scenario: p.scenario,
        event: p.event,
        algorithms: p.algorithms ?? ['dijkstra', 'q_learning', 'dqn'],
      })
      .then((r) => r.data),
  traffic: (includeEdges = false) =>
    http.get<TrafficState>('/traffic/current', { params: { include_edges: includeEdges } }).then((r) => r.data),
  simulate: (body: { scenario?: SimScenario; intensity?: number; advance_min?: number; surge?: number; clock?: string }) =>
    http.post<TrafficState>('/traffic/simulate', body).then((r) => r.data),
  advance: (minutes: number) => http.post<TrafficState>('/traffic/advance', null, { params: { minutes } }).then((r) => r.data),
  incident: (body: { lat?: number; lon?: number; radius_m?: number; severity?: number; duration_min?: number; random?: boolean; label?: string }) =>
    http.post<{ traffic: TrafficState }>('/traffic/incident', body).then((r) => r.data.traffic),
  clearIncident: (incident_id?: number) =>
    http.post<{ cleared: number; traffic: TrafficState }>('/traffic/incident/clear', { incident_id: incident_id ?? null }).then((r) => r.data.traffic),
  resetTraffic: (scenario: SimScenario = 'moderate', clock = '09:30') =>
    http.post<TrafficState>('/traffic/reset', { scenario, clock }).then((r) => r.data),
  liveRefresh: () => http.post<{ traffic: TrafficState }>('/traffic/live/refresh', null, { params: { force: true } }).then((r) => r.data.traffic),
  liveDisable: () => http.post<TrafficState>('/traffic/live/disable').then((r) => r.data),
  startTraining: (body: Record<string, unknown>) => http.post<TrainingStatus>('/training/start', body).then((r) => r.data),
  stopTraining: () => http.post<TrainingStatus>('/training/stop').then((r) => r.data),
  trainingStatus: () => http.get<TrainingStatus>('/training/status').then((r) => r.data),
  trainingHistory: () => http.get<TrainingHistory>('/training/history').then((r) => r.data),
  runExperiment: (body: { pairs: number; scenarios: SimScenario[]; preferences: Preference[]; events: TripEvent[]; seed?: number }) =>
    http.post<Experiment>('/experiments/run', body, { timeout: 300_000 }).then((r) => r.data),
  latestExperiment: () => http.get<Experiment>('/experiments/latest').then((r) => r.data),
  analytics: () => http.get<Analytics>('/analytics/summary').then((r) => r.data),
  datasets: () => http.get<DatasetsResponse>('/datasets').then((r) => r.data),
};
