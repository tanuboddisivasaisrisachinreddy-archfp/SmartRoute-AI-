export type Preference = 'fastest' | 'shortest' | 'balanced';
export type Scenario = 'current' | 'low' | 'moderate' | 'heavy' | 'peak' | 'random' | 'live';
export type SimScenario = Exclude<Scenario, 'current' | 'live'>;
export type TripEvent = 'none' | 'incident_on_route' | 'surge';
export type AlgorithmId = 'dijkstra' | 'q_learning' | 'dqn';

export interface Place {
  name: string;
  lat: number;
  lon: number;
  source?: string;
  category?: string | null;
  display_name?: string;
  id?: string;
}

export interface RouteMetrics {
  reached_destination: boolean;
  distance_km: number;
  travel_time_min: number;
  congestion_exposure: number;
  congestion_level: string;
  congested_minutes: number;
  segments: number;
  total_reward: number;
  generalized_cost: number;
  fuel_litres: number;
  estimated_cost_inr: number;
  revisits: number;
  avg_speed_kph: number;
}

export interface RouteResult {
  algorithm: AlgorithmId;
  label: string;
  color: string;
  success: boolean;
  error: string | null;
  computation_ms: number;
  route_score: number;
  metrics: RouteMetrics | null;
  geometry: [number, number][];
  edge_ids: number[];
  node_count: number;
  road_names: string[];
  diagnostics: Record<string, unknown> & { notes?: string; training_mode?: string; fallback_steps?: number };
}

export interface Endpoint {
  node: number;
  lat: number;
  lon: number;
  snap_distance_m: number;
  landmark: string | null;
  name?: string;
}

export interface CompareResponse {
  id: string | null;
  source: Endpoint;
  destination: Endpoint;
  preference: Preference;
  scenario: string;
  departure_clock: string;
  traffic_mode: 'simulation' | 'live';
  traffic_label: string;
  event: { type: string; description?: string; incident?: Incident } | null;
  routes: RouteResult[];
  best_algorithm: AlgorithmId | null;
  explanation: string;
  alternatives: { id: string; label: string; metrics: RouteMetrics; geometry: [number, number][] }[];
  cost_model: Record<string, unknown>;
  reward_config: Record<string, number>;
}

export interface Incident {
  id: number;
  lat: number;
  lon: number;
  radius_m: number;
  severity: number;
  current_intensity: number;
  duration_min: number;
  kind: string;
  label: string;
  status: string;
  affected_edges: number;
}

export interface TrafficHistoryPoint {
  clock_min: number;
  level: number;
  mean_congestion: number;
  p90_congestion: number;
  active_incidents: number;
}

export interface TrafficState {
  mode: 'simulation' | 'live';
  label: string;
  scenario: SimScenario;
  scenario_label: string;
  intensity_scale: number;
  clock: string;
  clock_min: number;
  elapsed_min: number;
  level: number;
  time_of_day_factor: number;
  mean_congestion: number;
  p90_congestion: number;
  heavy_share: number;
  active_incidents: number;
  incidents: Incident[];
  surge: number;
  live: Record<string, unknown> | null;
  edge_congestion?: number[];
  history?: TrafficHistoryPoint[];
  scenarios?: Record<string, string>;
  tomtom_configured?: boolean;
}

export interface NetworkSegment {
  ids: number[];
  c: [number, number][];
  k: number;
}

export interface NetworkData {
  segments: NetworkSegment[];
  bounds: { south: number; north: number; west: number; east: number };
  nodes: number;
  edges: number;
  meta: Record<string, unknown>;
}

export interface ModelStatus {
  loaded: boolean;
  reason?: string;
  path?: string;
  tables?: number;
  episodes?: number;
  rolling_success?: number;
  trained_at?: string;
  inference_backend?: string;
}

export interface Health {
  status: string;
  version: string;
  uptime_s: number;
  graph_loaded: boolean;
  graph: { nodes: number; edges: number; signature?: string; source?: string; synthetic?: boolean } | null;
  startup_error: string | null;
  traffic_mode: string | null;
  tomtom_configured: boolean;
  database: { type: string; connected: boolean; note?: string; database?: string };
  models: { q_learning?: ModelStatus; dqn?: ModelStatus };
  training: string;
}

export interface CurvePoint {
  episode: number;
  avg_reward: number;
  completion_rate: number;
  epsilon: number | null;
  loss: number | null;
  episode_reward: number;
}

export interface TrainingProgress {
  algorithm: string;
  state: string;
  current_episode: number;
  total_episodes: number;
  episode_reward: number | null;
  average_reward: number;
  epsilon: number | null;
  loss: number | null;
  completion_rate: number;
  elapsed_s: number;
  curve: CurvePoint[];
}

export interface TrainingStatus {
  state: 'idle' | 'starting' | 'running' | 'completed' | 'failed' | 'stopped';
  job: { id: string; algorithm: string; started_at: string; params: Record<string, unknown>; exit_code?: number } | null;
  progress: TrainingProgress | null;
  log_tail: string | null;
}

export interface TrainingHistoryEntry {
  curve: CurvePoint[];
  summary?: { preference: string; goal: string; episodes: number; final_success_rate: number; final_avg_reward: number }[];
  config: Record<string, unknown>;
  episodes: number;
  error?: string;
}

export interface TrainingHistory {
  q_learning: TrainingHistoryEntry | null;
  dqn: TrainingHistoryEntry | null;
  runs: Record<string, unknown>[];
}

export interface AlgoSummary {
  algorithm: AlgorithmId;
  label: string;
  runs: number;
  success_rate: number | null;
  avg_travel_time_min: number | null;
  avg_distance_km: number | null;
  avg_congestion?: number | null;
  avg_congestion_exposure?: number | null;
  avg_reward?: number | null;
  avg_total_reward?: number | null;
  avg_computation_ms: number | null;
  avg_score?: number | null;
  avg_route_score?: number | null;
  wins: number;
  time_ratio_vs_dijkstra?: number | null;
}

export interface ExperimentRow {
  source: string;
  destination: string;
  scenario: string;
  preference: string;
  event: string;
  algorithm: AlgorithmId;
  success: boolean;
  computation_ms: number;
  route_score: number;
  distance_km: number | null;
  travel_time_min: number | null;
  congestion_exposure: number | null;
  total_reward: number | null;
  best: boolean;
}

export interface Experiment {
  created_at: string;
  duration_s: number;
  config: { n_pairs: number; scenarios: string[]; preferences: string[]; events: string[] };
  graph: { nodes: number; edges: number; synthetic: boolean };
  summary: AlgoSummary[];
  rows: ExperimentRow[];
}

export interface Analytics {
  total_comparisons: number;
  total_routes_evaluated: number;
  total_searches: number;
  avg_travel_time_min: number | null;
  avg_distance_km: number | null;
  avg_congestion: number | null;
  algorithms: AlgoSummary[];
  popular_pairs: { pair: string; count: number }[];
  scenarios: { scenario: string; count: number }[];
  preferences: { preference: string; count: number }[];
  timeline: { created_at: string; travel_time_min: number; distance_km: number; congestion: number; best: string }[];
  database: string;
  latest_experiment: { created_at: string; summary: AlgoSummary[]; config: Experiment['config'] } | null;
  traffic: (TrafficState & { history?: TrafficHistoryPoint[] }) | null;
  models: Health['models'];
  training: string;
}

export interface DatasetEntry {
  id: string;
  name: string;
  source_urls: string[];
  geographic_coverage: string;
  is_hyderabad_data: boolean | null;
  features: string[];
  file_format: string;
  license: string;
  preprocessing: string[];
  intended_use: string;
  citation?: string;
}

export interface TrafficProfileDataset {
  dataset: string;
  geographic_origin: string;
  sensors: number;
  timesteps: number;
  start: string;
  end: string;
  missing_rate: number;
  hourly_congestion_weekday: number[];
}

export interface DatasetsResponse {
  catalog: DatasetEntry[];
  status: {
    processed_graph: { path: string; bytes: number | null };
    graph_stats: {
      nodes: number;
      edges: number;
      total_km: number;
      km_by_class: Record<string, number>;
      edges_by_class: Record<string, number>;
      mean_out_degree: number;
      oneway_share: number;
    } | null;
    landmarks_present: boolean;
    metr_la_present: boolean;
    pems_bay_present: boolean;
    traffic_profiles: { simulated_hyderabad_profile: number[]; datasets: TrafficProfileDataset[] } | null;
    validation: { check: string; passed: boolean; detail: string }[] | null;
  };
}

export interface Landmark {
  id: string;
  name: string;
  lat: number;
  lon: number;
  category: string;
  source: string;
  node: number;
  snap_distance_m: number;
}
