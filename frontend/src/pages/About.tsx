import { useAsync } from '../hooks/useAsync';
import { API_BASE, http } from '../services/api';
import { ALGO, Loading, PageHeader, Panel } from '../components/ui';

interface ModelsInfo {
  dqn_architecture: { state_features: string[]; action_features: string[]; hidden: number[]; max_actions: number; n_features: number };
  reward_preferences: Record<string, Record<string, number>>;
}

export default function About() {
  const info = useAsync(() => http.get<ModelsInfo>('/models').then((r) => r.data), []);
  const arch = info.data?.dqn_architecture;
  return (
    <div className="max-w-5xl">
      <PageHeader title="About SmartRoute AI" subtitle="An academic machine-learning project: reinforcement learning for dynamic route optimisation on the real Hyderabad road network." />
      <div className="space-y-5">
        <Panel title="How the system works">
          <ol className="list-decimal space-y-2 pl-5 text-[0.95rem] leading-relaxed text-granite-700 dark:text-granite-200">
            <li>OSMnx downloads the Hyderabad arterial road network from OpenStreetMap; intersections become nodes and road segments become directed edges.</li>
            <li>A traffic model assigns every segment a time-dependent congestion value (scenario level × time-of-day profile × road class × hotspots, plus noise and incidents). Speed drops and travel time rises with congestion.</li>
            <li>Dijkstra plans once with the traffic snapshot at departure. Q-Learning and DQN are policies that choose the next road segment at every intersection, observing traffic as the trip unfolds.</li>
            <li>Every route is replayed through the same time-dependent model to measure distance, travel time, congestion exposure, reward and fuel cost.</li>
          </ol>
        </Panel>

        <Panel title="The reinforcement-learning problem">
          <div className="space-y-3 text-[0.95rem] leading-relaxed text-granite-700 dark:text-granite-200">
            <p><strong>State</strong>: current intersection, destination, straight-line distance to it, network traffic level (4 buckets), local congestion, elapsed time and visited intersections.</p>
            <p><strong>Actions</strong>: the outgoing road segments of the current intersection. Invalid moves are never offered and the immediate U-turn is removed unless it is a dead end.</p>
            <p><strong>Reward</strong> per segment e taken at time t:</p>
            <pre className="overflow-x-auto rounded-lg bg-granite-100 p-3 text-sm dark:bg-granite-800">{`r = − w_time·tt_e(t)[min] − w_dist·len_e[km] − w_cong·c_e(t)·tt_e(t)
    + w_prog·(Φ(v) − Φ(u)),     Φ(n) = −distance(n, goal)[km]
    + 20 on arrival − 3 for revisiting − 15 on timeout / dead end`}</pre>
            <p>Time and congestion terms make the agent prefer quick, uncongested roads; the distance term implements the "shortest" preference; the potential-based progress term speeds up learning without changing the optimal policy; arrival and revisit terms enforce reaching the goal without loops.</p>
            {info.data && (
              <div className="overflow-x-auto">
                <table className="text-sm">
                  <thead className="text-left text-granite-500"><tr><th className="pr-6 font-medium">Preference</th>{['w_time', 'w_dist', 'w_cong', 'w_prog', 'goal_reward', 'gamma'].map((k) => <th key={k} className="pr-6 font-medium">{k}</th>)}</tr></thead>
                  <tbody>{Object.entries(info.data.reward_preferences).map(([p, w]) => <tr key={p}><td className="pr-6">{p}</td>{['w_time', 'w_dist', 'w_cong', 'w_prog', 'goal_reward', 'gamma'].map((k) => <td key={k} className="pr-6">{w[k]}</td>)}</tr>)}</tbody>
                </table>
              </div>
            )}
          </div>
        </Panel>

        <div className="grid gap-5 md:grid-cols-3">
          {(Object.keys(ALGO) as (keyof typeof ALGO)[]).map((k) => (
            <Panel key={k} title={<span className="flex items-center gap-2"><span className="h-2.5 w-6 rounded-full" style={{ background: ALGO[k].color }} />{ALGO[k].label}</span>}>
              <p className="text-sm leading-relaxed text-granite-600 dark:text-granite-300">
                {k === 'dijkstra' && 'NetworkX Dijkstra on edge weights from the departure-time traffic snapshot, using the same cost weights as the RL reward. Optimal for a static world; cannot react to changes during the trip.'}
                {k === 'q_learning' && 'Tabular Q(b, e) = Qbase[e] + Qres[b, e] per destination and preference (b = traffic bucket, e = outgoing segment). ε-greedy exploration with decay, visit-decaying learning rate α, undiscounted returns and backward replay of each episode. Untrained destinations get a bounded on-demand table.'}
                {k === 'dqn' && 'PyTorch Double-DQN: an MLP scores each candidate segment from state + action features, padding rows are masked. Experience replay, target network, Huber loss, Adam and checkpointing. Generalises to any origin/destination.'}
              </p>
            </Panel>
          ))}
        </div>

        <Panel title="DQN network">
          {info.loading ? <Loading /> : arch ? (
            <div className="space-y-2 text-sm text-granite-700 dark:text-granite-200">
              <p>Input {arch.n_features} features per candidate (up to {arch.max_actions} candidates) → {arch.hidden.join(' → ')} → 1 Q-value.</p>
              <p><span className="text-granite-500">State features:</span> {arch.state_features.join(', ')}</p>
              <p><span className="text-granite-500">Action features:</span> {arch.action_features.join(', ')}</p>
            </div>
          ) : <p className="text-sm text-granite-500">{info.error}</p>}
        </Panel>

        <Panel title="Honest limitations">
          <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-granite-700 dark:text-granite-200">
            <li>Traffic is simulated unless a TomTom key is configured; the simulator's profile and hotspots are modelling assumptions, not Hyderabad measurements.</li>
            <li>METR-LA and PEMS-BAY are US datasets used only for benchmarking traffic patterns.</li>
            <li>The graph keeps arterial roads (motorway to secondary) so tabular Q-learning stays tractable; trips through small lanes are snapped to the nearest arterial intersection.</li>
            <li>Dijkstra is optimal for static costs; the learned policies only have an edge when conditions change during the trip, and the comparison page reports cases where they are worse.</li>
          </ul>
        </Panel>

        <p className="text-xs text-granite-500">API: {API_BASE}/docs · Map data © OpenStreetMap contributors (ODbL).</p>
      </div>
    </div>
  );
}
